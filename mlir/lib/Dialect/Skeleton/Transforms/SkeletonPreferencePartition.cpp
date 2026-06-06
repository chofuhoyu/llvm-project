//===- SkeletonPreferencePartition.cpp - Partition by preference -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Partitions linalg.matmul ops carrying skeleton.preference into separate
// func.func ops, replacing each with a func.call.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Skeleton/IR/SkeletonAttrs.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir {
namespace skeleton {

#define GEN_PASS_DEF_SKELETONPREFERENCEPARTITION
#include "mlir/Dialect/Skeleton/Transforms/Passes.h.inc"

namespace {

struct SkeletonPreferencePartitionPass
    : public impl::SkeletonPreferencePartitionBase<
          SkeletonPreferencePartitionPass> {
  using SkeletonPreferencePartitionBase::SkeletonPreferencePartitionBase;

  void runOnOperation() override {
    ModuleOp module = cast<ModuleOp>(getOperation());
    MLIRContext *ctx = &getContext();
    SymbolTable symbolTable(module);

    // Phase 1: collect all matmuls to outline, grouped by enclosing function.
    // We must collect first because modifying the IR during a walk is unsafe.
    struct OutlineEntry {
      func::FuncOp func;
      linalg::MatmulOp matmul;
      unsigned index;
      std::string preference;
    };
    SmallVector<OutlineEntry> worklist;

    module.walk([&](func::FuncOp func) {
      unsigned idx = 0;
      func.walk([&](linalg::MatmulOp matmul) {
        auto prefAttr = matmul->getDiscardableAttr("skeleton.preference");
        if (!prefAttr)
          return WalkResult::advance();
        auto pref = dyn_cast<PreferenceAttr>(prefAttr);
        if (!pref) {
          matmul.emitWarning("expected #skeleton.preference attribute, got "
                             "incompatible type; skipping");
          return WalkResult::advance();
        }
        worklist.push_back({func, matmul, idx++, pref.getValue()});
        return WalkResult::advance();
      });
    });

    if (worklist.empty())
      return;

    // Phase 2: outline each matmul into its own function.
    OpBuilder builder(ctx);
    for (auto &entry : worklist) {
      linalg::MatmulOp matmul = entry.matmul;
      Location loc = matmul.getLoc();

      // Gather operand types for the new function signature.
      SmallVector<Value> callOperands;
      llvm::append_range(callOperands, matmul.getInputs());
      llvm::append_range(callOperands, matmul.getOutputs());

      SmallVector<Type> operandTypes;
      for (Value v : callOperands)
        operandTypes.push_back(v.getType());

      // Build the function type.
      auto funcType =
          FunctionType::get(ctx, operandTypes, matmul.getResultTypes());

      // Create the outlined func.func.
      std::string name =
          (entry.preference + "_matmul_" + Twine(entry.index)).str();
      auto outlinedFunc =
          func::FuncOp::create(loc, name, funcType);
      outlinedFunc.setPrivate();
      outlinedFunc->setAttr("skeleton.target",
                            StringAttr::get(ctx, entry.preference));

      // Populate the entry block.
      Block *entryBlock = outlinedFunc.addEntryBlock();
      builder.setInsertionPointToStart(entryBlock);

      // Create the matmul inside the new function using block arguments.
      unsigned numInputs = matmul.getInputs().size();
      auto newMatmul = linalg::MatmulOp::create(
          builder, loc, matmul.getResultTypes(),
          /*inputs=*/entryBlock->getArguments().slice(0, numInputs),
          /*outputs=*/entryBlock->getArguments().slice(numInputs));

      // Preserve the preference attribute on the new matmul.
      // The attribute is guaranteed to exist (filtered in Phase 1).
      auto prefAttr = matmul->getDiscardableAttr("skeleton.preference");
      newMatmul->setDiscardableAttr("skeleton.preference", prefAttr);

      // Add the return op.
      func::ReturnOp::create(builder, loc, newMatmul.getResults());

      // Insert the new function into the symbol table; capture the actual
      // name in case SymbolTable renames it to avoid collisions.
      StringAttr actualName = symbolTable.insert(outlinedFunc);

      // Replace the original matmul with a func.call.
      builder.setInsertionPoint(matmul);
      auto call = func::CallOp::create(builder, loc, actualName,
                                        matmul.getResultTypes(), callOperands);
      matmul.replaceAllUsesWith(call.getResults());
      matmul.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createSkeletonPreferencePartitionPass() {
  return std::make_unique<SkeletonPreferencePartitionPass>();
}

} // namespace skeleton
} // namespace mlir
