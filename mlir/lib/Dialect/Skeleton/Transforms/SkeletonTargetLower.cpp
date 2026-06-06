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
// Bufferization produces host memory (memref.alloc / memref.get_global).
// For GPU target functions we replace host buffers with gpu.alloc host_shared,
// which is GPU-accessible managed memory, and insert copy-back + dealloc
// after the call to keep host and device memory coherent.
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
    // Bufferization produces host memory (memref.get_global, memref.alloc).
    // GPU kernels cannot access host memory, so we:
    //   a) Before each GPU call: gpu.alloc host_shared + memref.copy(in) + cast
    //   b) After each GPU call:  memref.copy(out) + gpu.dealloc
    if (!gpuTargetNames.empty()) {
      // Collect GPU calls and their memref operands first, to avoid
      // iterator invalidation during IR modification.
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

      // Process each entry: insert alloc+copy before call, defer copy-back
      // and dealloc until after processing all operands of this call.
      OpBuilder builder(&getContext());
      DenseMap<func::CallOp, SmallVector<std::pair<Value, Value>>>
          callToAllocMapping; // call -> (gpuMemref, hostMemref) pairs

      for (auto &entry : bridgeWorklist) {
        builder.setInsertionPoint(entry.call);

        // Guard against dynamic memref shapes: we cannot allocate GPU memory
        // without concrete sizes. Skip with a diagnostic.
        if (entry.hostMemrefType.getNumDynamicDims() > 0) {
          entry.call.emitWarning(
              "skipping GPU memory bridging for dynamically-shaped memref "
              "operand; static shapes required for GPU lowering");
          continue;
        }

        // Allocate GPU-accessible managed memory.
        auto allocType = MemRefType::get(entry.hostMemrefType.getShape(),
                                          entry.hostMemrefType.getElementType());
        auto gpuAlloc = gpu::AllocOp::create(
            builder, entry.call.getLoc(), allocType,
            /*asyncToken=*/Type(),
            /*asyncDependencies=*/ValueRange{},
            /*dynamicSizes=*/ValueRange{},
            /*symbolOperands=*/ValueRange{},
            /*hostShared=*/true);

        // Copy data from host to GPU.
        memref::CopyOp::create(builder, entry.call.getLoc(), entry.hostMemref,
                                gpuAlloc.getMemref());

        // Cast to the expected (potentially strided) type and replace operand.
        auto casted = memref::CastOp::create(
            builder, entry.call.getLoc(), entry.hostMemrefType,
            gpuAlloc.getMemref());
        entry.call->setOperand(entry.operandIdx, casted);

        // Track for post-call copy-back and dealloc.
        callToAllocMapping[entry.call].push_back(
            {gpuAlloc.getMemref(), entry.hostMemref});
      }

      // Insert copy-back and dealloc after each GPU call.
      for (auto &kv : callToAllocMapping) {
        func::CallOp call = kv.first;
        auto &allocPairs = kv.second;

        builder.setInsertionPointAfter(call);
        for (auto &pair : allocPairs) {
          Value gpuMemref = pair.first;
          Value hostMemref = pair.second;

          // Copy results back from GPU to host memory.
          memref::CopyOp::create(builder, call.getLoc(), gpuMemref,
                                  hostMemref);

          // Deallocate GPU memory.
          gpu::DeallocOp::create(builder, call.getLoc(),
                                 /*asyncToken=*/Type(),
                                 /*asyncDependencies=*/ValueRange{},
                                 gpuMemref);
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
