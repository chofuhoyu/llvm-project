//===- SkeletonTargetLower.cpp - Lower by skeleton.target -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers the linalg ops of a func.func marked with skeleton.target. The
// runtime driver uses this to compile one operator unit for the device it
// placed it on. CPU targets use -convert-linalg-to-loops (scf.for), GPU
// targets use -convert-linalg-to-parallel-loops (scf.parallel).
//
// Bufferization produces host memory (memref.get_global / memref.alloc).
// GPU kernels cannot access plain host memory, so each GPU call operand is
// given a Unified Memory (UVM) buffer instead: gpu.alloc with host_shared
// lowers to cuMemAllocManaged, and the same allocation is visible to both the
// host and the kernel. The host copies the initial contents in and the results
// back out with memref.copy, which is a plain host memcpy into host-visible
// memory; no explicit device transfers (gpu.memcpy) or async streams are
// involved. The kernel launch itself is asynchronous, so the generated code
// synchronizes through the runtime before the first copy-back.
//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonDialect.h"
#include "mlir/Dialect/Skeleton/Runtime/SkeletonRuntime.h"
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
      if (func->getAttrOfType<TargetAttr>("skeleton.target"))
        targets.push_back(func);
    });

    if (targets.empty())
      return;

    // Build a set of GPU target function names (using StringSet which owns
    // copies of the strings, avoiding dangling StringRefs after IR mutations).
    // A managed target takes its operands from memory the runtime allocated, so
    // its callers pass it device-visible buffers already.
    llvm::StringSet<> gpuTargetNames;
    llvm::StringSet<> managedTargetNames;
    for (func::FuncOp func : targets) {
      if (auto attr = func->getAttrOfType<TargetAttr>("skeleton.target")) {
        if (attr.getValue() == "GPU") {
          gpuTargetNames.insert(func.getSymName().str());
          if (func->hasAttr("skeleton.managed"))
            managedTargetNames.insert(func.getSymName().str());
        }
      }
    }

    // Step 1: bufferize tensor ops. This pass is module-scoped and
    // requires function-boundaries mode to update function signatures.
    // Function boundaries use identity layout maps so that host memrefs are
    // statically contiguous; the GPU bridge below relies on that (gpu.memcpy
    // is a raw byte copy of contiguous memory).
    {
      OpPassManager pm("builtin.module", OpPassManager::Nesting::Explicit);
      bufferization::OneShotBufferizePassOptions bufOpts;
      bufOpts.bufferizeFunctionBoundaries = true;
      bufOpts.functionBoundaryTypeConversion =
          bufferization::LayoutMapOption::IdentityLayoutMap;
      pm.addPass(bufferization::createOneShotBufferizePass(bufOpts));
      if (failed(runPipeline(pm, module)))
        return signalPassFailure();
    }

    // Step 1b: ensure GPU target functions receive GPU-accessible memory.
    //   a) gpu.alloc host_shared (UVM) + memref.copy (host -> UVM) before
    //      each GPU call
    //   b) memref.copy (UVM -> host) + gpu.dealloc after each GPU call
    //   (skip copy-back for read-only memref.get_global operands)
    // The host and the kernel use the same managed buffer, so there are no
    // async device transfers and no stream/token choreography.
    if (!gpuTargetNames.empty()) {
      struct GPUBridgeEntry {
        func::CallOp call;
        unsigned operandIdx;
        Value hostMemref;
        MemRefType hostMemrefType;
      };
      SmallVector<GPUBridgeEntry> bridgeWorklist;
      // Every call into a GPU target, whether or not it moves memrefs. The
      // kernel behind it runs asynchronously, so each of them needs a
      // synchronization before the host touches anything the kernel wrote.
      SmallVector<func::CallOp> gpuCalls;

      bool multiResult = false;
      module.walk([&](func::CallOp call) {
        if (multiResult)
          return;
        auto callee = call.getCallee();
        if (!gpuTargetNames.contains(callee))
          return;
        gpuCalls.push_back(call);
        // A managed unit's operands are already memory the kernel can reach.
        // Bridging them would allocate a device buffer and copy the data into
        // it, which is exactly the cost the managed path exists to avoid.
        if (managedTargetNames.contains(callee))
          return;
        if (call->getNumResults() > 1) {
          call.emitError("cannot bridge an operator with more than one result "
                         "to the GPU");
          signalPassFailure();
          multiResult = true;
          return;
        }
        for (unsigned i = 0; i < call.getNumOperands(); ++i) {
          Value operand = call.getOperand(i);
          auto memrefType = dyn_cast<MemRefType>(operand.getType());
          if (!memrefType)
            continue;
          bridgeWorklist.push_back({call, i, operand, memrefType});
        }
      });
      if (multiResult)
        return;

      OpBuilder builder(&getContext());

      // Phase A: UVM alloc + host-to-UVM copy before each GPU call.
      DenseMap<func::CallOp, SmallVector<std::pair<Value, Value>>>
          callGpuMemrefs; // call -> (uvmMemref, hostMemref) pairs

      for (auto &entry : bridgeWorklist) {
        builder.setInsertionPoint(entry.call);

        if (entry.hostMemrefType.getNumDynamicDims() > 0) {
          // Dynamic shapes are not supported by GPU bridging: passing the
          // host memref straight through would make the kernel read/write
          // host memory. Fail explicitly instead of silently degrading.
          entry.call.emitError(
              "cannot bridge dynamically-shaped memref operand to GPU "
              "(static shapes required for GPU lowering)");
          signalPassFailure();
          return;
        }

        if (!entry.hostMemrefType.getLayout().isIdentity()) {
          // The allocation is a byte-for-byte copy of the host buffer and the
          // kernel receives it with the host memref's type. A non-contiguous
          // host buffer (e.g. a hand-written strided memref) would need its
          // strides preserved explicitly; fail instead of producing wrong
          // results.
          entry.call.emitError(
              "cannot bridge non-contiguous memref operand to GPU "
              "(identity layout required for GPU lowering)");
          signalPassFailure();
          return;
        }

        // Host-shared allocation: on CUDA this is cuMemAllocManaged, i.e.
        // Unified Memory accessible from both the host and the kernel.
        // Host memrefs are identity-layout and statically contiguous
        // (bufferize config above, plus the identity check above), so the
        // allocation type matches the host type; the cast below is a
        // type-level no-op.
        auto allocType = entry.hostMemrefType;
        auto allocOp = gpu::AllocOp::create(
            builder, entry.call.getLoc(), allocType,
            /*asyncToken=*/Type(), {}, {}, {}, /*hostShared=*/true);

        // Synchronous host memcpy into host-visible UVM memory. The kernel
        // reads the very same buffer later, so no device transfer is needed.
        memref::CopyOp::create(builder, entry.call.getLoc(), entry.hostMemref,
                               allocOp.getMemref());

        // Cast to strided type and replace call operand.
        auto casted =
            memref::CastOp::create(builder, entry.call.getLoc(),
                                   entry.hostMemrefType, allocOp.getMemref());
        entry.call->setOperand(entry.operandIdx, casted);

        callGpuMemrefs[entry.call].push_back(
            {allocOp.getMemref(), entry.hostMemref});
      }

      // The runtime entry point the generated code synchronizes through.
      if (!module.lookupSymbol<func::FuncOp>("skeleton_rt_sync")) {
        OpBuilder declarationBuilder(module.getContext());
        declarationBuilder.setInsertionPointToStart(module.getBody());
        auto declaration = func::FuncOp::create(
            declarationBuilder, module.getLoc(), "skeleton_rt_sync",
            FunctionType::get(module.getContext(),
                              {declarationBuilder.getI32Type()}, {}));
        declaration.setPrivate();
      }

      // Phase B: synchronize, copy back, and deallocate after each GPU call.
      for (func::CallOp call : gpuCalls) {
        // The launch has returned but the kernel has not necessarily finished.
        // The copy-backs below read what it wrote, so wait for the device
        // first; a kernel that faults otherwise shows up as a host read of
        // half-written memory. The device number is materialized before the
        // call so that the wait is the first thing after it.
        builder.setInsertionPoint(call);
        Value device = arith::ConstantOp::create(
            builder, call.getLoc(), builder.getI32Type(),
            builder.getI32IntegerAttr(SKELETON_DEVICE_GPU));
        builder.setInsertionPointAfter(call);
        func::CallOp::create(builder, call.getLoc(), "skeleton_rt_sync",
                             TypeRange{}, ValueRange{device});

        auto &pairs = callGpuMemrefs[call];
        Value outputHostMemref;

        for (auto &pair : pairs) {
          Value gpuMemref = pair.first;
          Value hostMemref = pair.second;

          // Skip copy-back for read-only constants (memref.get_global).
          Value source = hostMemref;
          while (auto castOp = source.getDefiningOp<memref::CastOp>())
            source = castOp.getSource();
          bool isReadOnly = source.getDefiningOp<memref::GetGlobalOp>();
          if (!isReadOnly) {
            // Synchronous host memcpy out of the UVM buffer. The kernel is
            // launched synchronously, so its results are already visible.
            memref::CopyOp::create(builder, call.getLoc(), gpuMemref,
                                   hostMemref);
            outputHostMemref = hostMemref;
          }

          gpu::DeallocOp::create(builder, call.getLoc(),
                                 /*asyncToken=*/Type(),
                                 /*asyncDependencies=*/ValueRange{}, gpuMemref);
        }

        // The pipeline accepts a single result (skeleton-outline rejects more)
        // and linalg puts the destination last, so the last writable operand
        // is the one whose copy-back becomes the call result. A void result
        // still refreshes the writable operands through the copy-backs above.
        // Read-only memref.get_global operands are assumed immutable.
        // Replace the call result (UVM memref, now dealloc'd) with the host
        // memref that received the copy-back.
        if (outputHostMemref && call->getNumResults() > 0 &&
            isa<MemRefType>(call.getResult(0).getType())) {
          auto retType = cast<MemRefType>(call.getResult(0).getType());
          auto retCast = memref::CastOp::create(builder, call.getLoc(), retType,
                                                outputHostMemref);
          call->getResult(0).replaceAllUsesWith(retCast.getResult());
        }
      }
    }

    // Step 2: lower linalg per function, choosing loops or parallel loops.
    for (func::FuncOp func : targets) {
      auto targetAttr = func->getAttrOfType<TargetAttr>("skeleton.target");
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

    // The skeleton.target attribute has served its purpose once lowering is
    // done. Remove it so it does not leak into downstream passes (e.g.
    // mlir-runner in integration tests) that do not load the skeleton
    // dialect and would fail to parse the typed #skeleton.target<...>.
    for (func::FuncOp func : targets)
      func->removeAttr("skeleton.target");
  }
};

} // namespace

std::unique_ptr<Pass> createSkeletonTargetLowerPass() {
  return std::make_unique<SkeletonTargetLowerPass>();
}

} // namespace skeleton
} // namespace mlir
