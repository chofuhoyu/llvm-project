//===- SkeletonPreferencePartition.cpp - Partition by preference -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Partitions Linalg ops carrying skeleton.preference into separate func.func
// ops, replacing each with a func.call.  Supports any linalg::LinalgOp
// (matmul, add, generic, reduce, etc.).
//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonAttrs.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include <vector>

using namespace mlir;

namespace mlir {
namespace skeleton {

#define GEN_PASS_DEF_SKELETONPREFERENCEPARTITION
#include "mlir/Dialect/Skeleton/Transforms/Passes.h.inc"

namespace {

/// A generic entry for outlining any Linalg op.  Only the op reference and
/// naming metadata are collected in Phase 1; operands are re-read from the
/// live op in Phase 2, because a producer outlined earlier in the worklist
/// replaces its uses (RAUW) before a consumer op is handled.
struct OutlineEntry {
  linalg::LinalgOp linalgOp;
  unsigned index;
  std::string preference;
  std::string opName; // for generating function name
};

struct SkeletonPreferencePartitionPass
    : public impl::SkeletonPreferencePartitionBase<
          SkeletonPreferencePartitionPass> {
  using SkeletonPreferencePartitionBase::SkeletonPreferencePartitionBase;

  void runOnOperation() override {
    ModuleOp module = cast<ModuleOp>(getOperation());
    MLIRContext *ctx = &getContext();
    SymbolTable symbolTable(module);

    // Phase 1: collect all Linalg ops with skeleton.preference to outline.
    std::vector<OutlineEntry> worklist;

    module.walk([&](func::FuncOp func) {
      unsigned idx = 0;
      func.walk([&](linalg::LinalgOp linalgOp) {
        auto prefAttr =
            linalgOp->getDiscardableAttr("skeleton.preference");
        if (!prefAttr)
          return WalkResult::advance();

        auto pref = dyn_cast<PreferenceAttr>(prefAttr);
        if (!pref) {
          linalgOp.emitWarning(
              "expected #skeleton.preference attribute, got "
              "incompatible type; skipping");
          return WalkResult::advance();
        }

        OutlineEntry entry;
        entry.linalgOp = linalgOp;
        entry.index = idx++;
        entry.preference = pref.getValue().str();
        entry.opName = linalgOp->getName().stripDialect().str();

        worklist.push_back(entry);
        return WalkResult::advance();
      });
    });

    if (worklist.empty())
      return;

    // Phase 2: outline each op into its own function.
    OpBuilder builder(ctx);
    for (auto &entry : worklist) {
      linalg::LinalgOp op = entry.linalgOp;
      Location loc = op->getLoc();

      // Gather operands from the op's current state.  The Phase-1 snapshot
      // can go stale: a producer outlined earlier in the worklist replaces
      // its uses (RAUW) before this op is handled, so snapshot Values become
      // dangling.  Reading the live op's operands is both correct and
      // order-independent.
      SmallVector<Value> callOperands;
      SmallVector<Type> operandTypes;
      for (Value v : op.getDpsInputs()) {
        callOperands.push_back(v);
        operandTypes.push_back(v.getType());
      }
      for (Value v : op.getDpsInits()) {
        callOperands.push_back(v);
        operandTypes.push_back(v.getType());
      }

      SmallVector<Type> resultTypes = llvm::to_vector(op->getResultTypes());

      auto funcType = FunctionType::get(ctx, operandTypes, resultTypes);

      // Create the outlined func.func.
      std::string name =
          (entry.preference + "_" + entry.opName + "_" +
           Twine(entry.index))
              .str();
      auto outlinedFunc = func::FuncOp::create(loc, name, funcType);
      outlinedFunc.setPrivate();
      outlinedFunc->setAttr("skeleton.target",
                            StringAttr::get(ctx, entry.preference));

      Block *entryBlock = outlinedFunc.addEntryBlock();
      builder.setInsertionPointToStart(entryBlock);

      // Clone the op inside the new function using block arguments.
      IRMapping mapper;
      // Map block arguments to the original operands.
      for (unsigned i = 0; i < callOperands.size(); ++i)
        mapper.map(callOperands[i], entryBlock->getArgument(i));

      Operation *newOp = builder.clone(*op, mapper);

      // Preserve the preference attribute on the new op.
      auto prefAttr = op->getDiscardableAttr("skeleton.preference");
      if (prefAttr)
        newOp->setDiscardableAttr("skeleton.preference", prefAttr);

      // Add return op.
      func::ReturnOp::create(builder, loc, newOp->getResults());

      // Insert into symbol table.
      StringAttr actualName = symbolTable.insert(outlinedFunc);

      // Replace the original op with a func.call.
      builder.setInsertionPoint(op);
      auto call = func::CallOp::create(builder, loc, actualName, resultTypes,
                                       callOperands);
      op->replaceAllUsesWith(call.getResults());
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createSkeletonPreferencePartitionPass() {
  return std::make_unique<SkeletonPreferencePartitionPass>();
}

} // namespace skeleton
} // namespace mlir
