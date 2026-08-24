//===- SkeletonPatternMerge.cpp - Merge skeleton patterns -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Merges skeleton.map + pure_fn patterns into named skeleton ops.
// For example:
//   skeleton.map {pure_fn = @my_add}  →  skeleton.vector_add
//   where @my_add body is a single arith.addf.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonDialect.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace skeleton {

#define GEN_PASS_DEF_SKELETONPATTERNMERGE
#include "mlir/Dialect/Skeleton/Transforms/Passes.h.inc"

namespace {

/// Check if a func.func body consists of a single arith.addf on the arguments
/// and returns the result.
static bool isAddFn(func::FuncOp fn) {
  if (!fn || fn.getBody().getBlocks().size() != 1)
    return false;
  auto &block = fn.getBody().front();
  // Expect: %r = arith.addf %a, %b : f32
  //         return %r : f32
  // That's 2 ops: addf + return.
  if (std::distance(block.begin(), block.end()) != 2)
    return false;

  auto addf = dyn_cast<arith::AddFOp>(&block.front());
  if (!addf)
    return false;

  // Check that the addf uses the block arguments.
  if (addf.getLhs() != fn.getArgument(0) ||
      addf.getRhs() != fn.getArgument(1))
    return false;

  // Check the return uses the addf result.
  auto ret = dyn_cast<func::ReturnOp>(block.getTerminator());
  if (!ret || ret.getOperands().size() != 1)
    return false;
  if (ret.getOperand(0) != addf.getResult())
    return false;

  return true;
}

/// Convert skeleton.map {pure_fn = @add_like_fn} → skeleton.vector_add.
struct MergeMapAddToVectorAdd : public OpRewritePattern<MapOp> {
  using OpRewritePattern<MapOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MapOp op,
                                PatternRewriter &rewriter) const override {
    // Only handle map with pure_fn and no body, with exactly 2 inputs.
    if (!op.getPureFnAttr() || !op.getBody().empty())
      return failure();
    if (op.getInputs().size() != 2)
      return failure();

    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto fn = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        op, op.getPureFnAttr());
    if (!fn)
      return failure();

    // Check if the pure function is just a simple addf.
    if (!isAddFn(fn))
      return failure();

    // Create skeleton.vector_add.
    auto vectorAdd = VectorAddOp::create(
        rewriter, op.getLoc(), op.getResult().getType(),
        op.getInputs()[0], op.getInputs()[1], op.getOutput(),
        op.getPreferenceAttr());

    rewriter.replaceOp(op, vectorAdd.getResult());
    return success();
  }
};

class SkeletonPatternMergePass
    : public impl::SkeletonPatternMergeBase<SkeletonPatternMergePass> {
public:
  using SkeletonPatternMergeBase::SkeletonPatternMergeBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MergeMapAddToVectorAdd>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createSkeletonPatternMergePass() {
  return std::make_unique<SkeletonPatternMergePass>();
}

} // namespace skeleton
} // namespace mlir
