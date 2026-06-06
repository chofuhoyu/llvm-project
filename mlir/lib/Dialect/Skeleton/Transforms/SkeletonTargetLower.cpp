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
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
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
    ModuleOp module = cast<ModuleOp>(getOperation());

    // Collect target functions before modifying IR.
    SmallVector<func::FuncOp> targets;
    module.walk([&](func::FuncOp func) {
      if (func->getAttrOfType<StringAttr>("skeleton.target"))
        targets.push_back(func);
    });

    if (targets.empty())
      return;

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

    // Step 2: lower linalg per function, choosing loops or parallel loops.
    for (func::FuncOp func : targets) {
      auto targetAttr = func->getAttrOfType<StringAttr>("skeleton.target");
      StringRef target = targetAttr.getValue();

      OpPassManager pm("func.func", OpPassManager::Nesting::Explicit);

      if (target == "CPU") {
        pm.addPass(createConvertLinalgToLoopsPass());
      } else if (target == "GPU") {
        pm.addPass(createConvertLinalgToParallelLoopsPass());
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
