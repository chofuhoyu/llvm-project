//===- SkeletonTargetLower.cpp - Lower by skeleton.target -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers linalg.matmul ops in func.func ops marked with skeleton.target.
// CPU targets use -convert-linalg-to-loops (scf.for),
// GPU targets use -convert-linalg-to-parallel-loops (scf.parallel).
//
// Bufferization produces host memory (memref.get_global / memref.alloc).
// GPU kernels cannot access host memory, so we replace host buffers with
// device memory (gpu.alloc) and use gpu.memcpy for explicit data transfers.
// GPU ops are chained via async tokens to satisfy lowering pattern
// requirements (isAsyncWithOneDependency).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace mlir {
namespace skeleton {

#define GEN_PASS_DEF_SKELETONTARGETLOWER
#include "mlir/Dialect/Skeleton/Transforms/Passes.h.inc"

namespace {

struct SkeletonTargetLowerPass
    : public impl::SkeletonTargetLowerBase<SkeletonTargetLowerPass> {
  using SkeletonTargetLowerBase::SkeletonTargetLowerBase;

  void runOnOperation() override {
    auto module = dyn_cast<ModuleOp>(getOperation());
    if (!module) {
      signalPassFailure();
      return;
    }

    // Collect target functions before modifying IR.
    SmallVector<func::FuncOp> targets;
    module.walk([&](func::FuncOp func) {
      if (func->getAttrOfType<StringAttr>("skeleton.target"))
        targets.push_back(func);
    });

    if (targets.empty())
      return;

    // Build a set of GPU target function names (using StringSet which owns
    // copies of the strings, avoiding dangling StringRefs after IR mutations).
    llvm::StringSet<> gpuTargetNames;
    for (func::FuncOp func : targets) {
      if (auto attr = func->getAttrOfType<StringAttr>("skeleton.target")) {
        if (attr.getValue() == "GPU")
          gpuTargetNames.insert(func.getSymName().str());
      }
    }

    // Step 1: bufferize tensor ops. This pass is module-scoped and
    // requires function-boundaries mode to update function signatures.
    {
      OpPassManager pm("builtin.module", OpPassManager::Nesting::Explicit);
      bufferization::OneShotBufferizePassOptions bufOpts;
      bufOpts.bufferizeFunctionBoundaries = true;
      pm.addPass(bufferization::createOneShotBufferizePass(bufOpts));
      if (failed(runPipeline(pm, module)))
        return signalPassFailure();
    }

    // Step 1b: ensure GPU target functions receive GPU-accessible memory.
    //   a) gpu.alloc (sync, device memory) + gpu.memcpy async (H→D)
    //   b) gpu.memcpy async (D→H) + gpu.dealloc (sync)
    //   (skip copy-back + dealloc for read-only memref.get_global operands)
    //
    // gpu.alloc and gpu.dealloc now use sync versions thanks to #191661.
    // Only gpu.memcpy still requires async (isAsyncWithOneDependency).
    if (!gpuTargetNames.empty()) {
      struct GPUBridgeEntry {
        func::CallOp call;
        unsigned operandIdx;
        Value hostMemref;
        MemRefType hostMemrefType;
      };
      SmallVector<GPUBridgeEntry> bridgeWorklist;

      module.walk([&](func::CallOp call) {
        auto callee = call.getCallee();
        if (!gpuTargetNames.contains(callee))
          return;
        for (unsigned i = 0; i < call.getNumOperands(); ++i) {
          Value operand = call.getOperand(i);
          auto memrefType = dyn_cast<MemRefType>(operand.getType());
          if (!memrefType)
            continue;
          bridgeWorklist.push_back({call, i, operand, memrefType});
        }
      });

      OpBuilder builder(&getContext());
      auto asyncTokenType =
          gpu::AsyncTokenType::get(builder.getContext());

      // Phase A: sync alloc + async memcpy(H→D) before each GPU call.
      DenseMap<func::CallOp, SmallVector<std::pair<Value, Value>>>
          callGpuMemrefs; // call -> (gpuMemref, hostMemref) pairs

      for (auto &entry : bridgeWorklist) {
        builder.setInsertionPoint(entry.call);

        if (entry.hostMemrefType.getNumDynamicDims() > 0) {
          // TODO: Skipping the bridge leaves the host memref as the GPU
          // call operand, so the kernel reads/writes host memory. Fail the
          // pass (or implement dynamic-shape bridging) instead of silently
          // degrading.
          entry.call.emitWarning(
              "skipping GPU memory bridging for dynamically-shaped memref "
              "operand; static shapes required for GPU lowering");
          continue;
        }

        // Sync alloc: no async, no token.
        // TODO: The device memref type is rebuilt from shape + element
        // type only, dropping the host layout (offset/strides) and any
        // explicit address space; the cast back to hostMemrefType below
        // papers over the difference. Derive the alloc type from
        // hostMemrefType instead.
        auto allocType =
            MemRefType::get(entry.hostMemrefType.getShape(),
                            entry.hostMemrefType.getElementType());
        auto allocOp = gpu::AllocOp::create(
            builder, entry.call.getLoc(), allocType,
            /*asyncToken=*/Type(), {},
            {}, {}, /*hostShared=*/false);

        // Async memcpy host → device (only this op needs a token).
        auto waitOp = gpu::WaitOp::create(builder, entry.call.getLoc(),
                                           asyncTokenType, {});
        Value token = waitOp.getAsyncToken();
        gpu::MemcpyOp::create(builder, entry.call.getLoc(),
                               /*asyncToken=*/asyncTokenType,
                               /*asyncDependencies=*/ValueRange{token},
                               allocOp.getMemref(), entry.hostMemref);

        // Cast to strided type and replace call operand.
        auto casted = memref::CastOp::create(
            builder, entry.call.getLoc(), entry.hostMemrefType,
            allocOp.getMemref());
        entry.call->setOperand(entry.operandIdx, casted);

        callGpuMemrefs[entry.call].push_back(
            {allocOp.getMemref(), entry.hostMemref});
      }

      // Phase B: async memcpy(D→H) + sync dealloc after each GPU call.
      for (auto &kv : callGpuMemrefs) {
        func::CallOp call = kv.first;
        auto &pairs = kv.second;

        builder.setInsertionPointAfter(call);

        // Fresh async token for post-kernel memcpy chain.
        auto waitOp = gpu::WaitOp::create(builder, call.getLoc(),
                                           asyncTokenType, {});
        Value token = waitOp.getAsyncToken();
        Value outputHostMemref;

        for (auto &pair : pairs) {
          Value gpuMemref = pair.first;
          Value hostMemref = pair.second;

          // Skip copy-back for read-only constants (memref.get_global).
          Value source = hostMemref;
          while (auto castOp =
                     source.getDefiningOp<memref::CastOp>())
            source = castOp.getSource();
          bool isReadOnly =
              source.getDefiningOp<memref::GetGlobalOp>();
          if (!isReadOnly) {
            auto memcpyOut = gpu::MemcpyOp::create(
                builder, call.getLoc(),
                /*asyncToken=*/asyncTokenType,
                /*asyncDependencies=*/ValueRange{token},
                hostMemref, gpuMemref);
            token = memcpyOut.getAsyncToken();
            outputHostMemref = hostMemref;
          }

          // Sync dealloc: no async, no token.
          gpu::DeallocOp::create(builder, call.getLoc(),
                                 /*asyncToken=*/Type(),
                                 /*asyncDependencies=*/ValueRange{},
                                 gpuMemref);
        }

        // TODO: Single-output assumption: only the last copied-back host
        // memref replaces call.getResult(0) (hard-coded index 0), and the
        // read-only check above treats every memref.get_global as
        // immutable. Calls with several writable memrefs, multiple
        // results, or written globals silently lose the other copy-backs.
        // Replace call result (device memref, now dealloc'd) with the
        // host memref that received the copy-back.
        if (outputHostMemref) {
          auto retType =
              cast<MemRefType>(call.getResult(0).getType());
          auto retCast = memref::CastOp::create(
              builder, call.getLoc(), retType, outputHostMemref);
          call->getResult(0).replaceAllUsesWith(retCast.getResult());
        }
      }
    }

    // Step 2: lower linalg per function, choosing loops or parallel loops.
    for (func::FuncOp func : targets) {
      auto targetAttr = func->getAttrOfType<StringAttr>("skeleton.target");
      StringRef target = targetAttr.getValue();

      OpPassManager pm("func.func", OpPassManager::Nesting::Explicit);

      if (target == "CPU") {
        pm.addPass(createConvertLinalgToLoopsPass());
      } else if (target == "GPU") {
        pm.addPass(createConvertLinalgToParallelLoopsPass());
      } else {
        func.emitWarning("unknown skeleton.target value '")
            << target << "'; skipping lowering for this function";
        continue;
      }

      if (failed(runPipeline(pm, func)))
        return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createSkeletonTargetLowerPass() {
  return std::make_unique<SkeletonTargetLowerPass>();
}

} // namespace skeleton
} // namespace mlir
