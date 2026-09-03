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
//   skeleton.map {pure_fn = @my_add}  →  skeleton.vector.add
//   where @my_add computes a+b (arith.addf or arith.addi on its two arguments,
//   either order) and returns the result.
//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonDialect.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonOps.h"
#include "mlir/IR/BuiltinTypes.h"
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

/// Check if a func.func body computes `a + b` and returns it.
///
/// Accepts both `arith.addf` (floating point) and `arith.addi` (integer):
/// vector.add is type-polymorphic, mirroring `linalg.add`, so the element type
/// comes from the map operands.
///
/// Uses root-op matching on the returned value: the returned value must be the
/// direct result of an `arith.addf`/`arith.addi` whose two operands are
/// exactly the function's two arguments, in either order (addition is
/// commutative). Extra unrelated instructions in the body are tolerated — they
/// are still inlined by skeleton-to-linalg, so merging remains semantically
/// sound.
///
/// A single-block body is required: multi-block bodies are silently truncated
/// by the inlining in SkeletonToLinalg (see TODO-5), so merging them here
/// would silently change semantics.
static bool isAddOp(func::FuncOp fn) {
  if (!fn || fn.getBody().getBlocks().size() != 1)
    return false;
  auto &block = fn.getBody().front();
  if (block.getNumArguments() != 2)
    return false;

  // The returned value must be the result of an integer or floating-point
  // addition.
  auto ret = dyn_cast<func::ReturnOp>(block.getTerminator());
  if (!ret || ret.getOperands().size() != 1)
    return false;
  auto *addOp = ret.getOperand(0).getDefiningOp();
  if (!isa<arith::AddFOp, arith::AddIOp>(addOp))
    return false;

  // The add must add exactly the two function arguments, in either order.
  Value arg0 = fn.getArgument(0), arg1 = fn.getArgument(1);
  return (addOp->getOperand(0) == arg0 && addOp->getOperand(1) == arg1) ||
         (addOp->getOperand(0) == arg1 && addOp->getOperand(1) == arg0);
}

/// Convert skeleton.map {pure_fn = @add_like_fn} → skeleton.vector.add.
struct MergeMapAddToVectorAdd : public OpRewritePattern<MapOp> {
  using OpRewritePattern<MapOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MapOp op,
                                PatternRewriter &rewriter) const override {
    // Only handle map with pure_fn and no body, with exactly 2 inputs.
    if (!op.getPureFnAttr() || !op.getBody().empty())
      return failure();
    if (op.getInputs().size() != 2)
      return failure();

    // vector.add is 1D-only and requires a uniform element type (see
    // VectorAddOp::verify). Check before merging: without this, a 2D+
    // map{pure_fn=@add} would be rewritten into a vector.add that fails
    // verification. (Element-type uniformity is already implied by MapOp::verify
    // via the pure_fn signature check, but we keep it explicit here so this
    // rewrite never emits IR the verifier rejects.)
    auto lhsType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
    auto rhsType = dyn_cast<RankedTensorType>(op.getInputs()[1].getType());
    auto initType = dyn_cast<RankedTensorType>(op.getInit().getType());
    if (!lhsType || !rhsType || !initType || lhsType.getRank() != 1 ||
        rhsType.getRank() != 1 || initType.getRank() != 1)
      return failure();
    auto eltType = lhsType.getElementType();
    if (rhsType.getElementType() != eltType ||
        initType.getElementType() != eltType)
      return failure();

    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto fn = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        op, op.getPureFnAttr());
    if (!fn)
      return failure();

    // Check if the pure function is just a simple add.
    if (!isAddOp(fn))
      return failure();

    // Create skeleton.vector.add. The result type is derived from init
    // (destination-passing), so it's not repeated here.
    auto vectorAdd = VectorAddOp::create(
        rewriter, op.getLoc(),
        op.getInputs()[0], op.getInputs()[1], op.getInit(),
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
