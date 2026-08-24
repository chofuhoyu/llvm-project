//===- SkeletonToLinalg.cpp - Skeleton to Linalg conversion -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonDialect.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace skeleton {

#define GEN_PASS_DEF_SKELETONTOLINALG
#include "mlir/Dialect/Skeleton/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// ConvertMapOp (pure_fn, no region) → linalg.generic
//===----------------------------------------------------------------------===//

struct ConvertMapOp : public OpRewritePattern<MapOp> {
  using OpRewritePattern<MapOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MapOp op,
                                PatternRewriter &rewriter) const override {
    // Only handle the case with pure_fn and no body region.
    if (!op.getPureFnAttr() || !op.getBody().empty())
      return failure();

    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto fn = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        op, op.getPureFnAttr());
    if (!fn)
      return failure();

    // Compute indexing maps and iterator types from the output tensor.
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    unsigned rank = resultType.getRank();
    auto *ctx = rewriter.getContext();

    // Identity map for each operand.
    SmallVector<AffineMap> indexingMaps;
    unsigned numOperands = op.getInputs().size() + 1; // inputs + output
    for (unsigned i = 0; i < numOperands; ++i)
      indexingMaps.push_back(
          AffineMap::getMultiDimIdentityMap(rank, ctx));

    // All parallel iterators for element-wise ops.
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    auto generic = linalg::GenericOp::create(
        rewriter, op.getLoc(),
        /*resultTensorTypes=*/resultType,
        /*inputs=*/op.getInputs(),
        /*outputs=*/ValueRange{op.getOutput()},
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        /*doc=*/"",
        /*libraryCall=*/"",
        [&](OpBuilder &bodyBuilder, Location loc, ValueRange blockArgs) {
          // Map blockArgs to pure_fn parameters and clone the body.
          IRMapping mapper;
          for (unsigned i = 0; i < blockArgs.size() - 1; ++i)
            mapper.map(fn.getArgument(i), blockArgs[i]);
          for (auto &bodyOp : fn.getBody().front().without_terminator())
            bodyBuilder.clone(bodyOp, mapper);
          auto retOp = cast<func::ReturnOp>(
              fn.getBody().front().getTerminator());
          Value mappedRet = mapper.lookupOrDefault(retOp.getOperand(0));
          linalg::YieldOp::create(bodyBuilder, loc, mappedRet);
        });

    // Preserve skeleton.preference as a discardable attribute.
    if (auto pref = op.getPreferenceAttr())
      generic->setDiscardableAttr("skeleton.preference", pref);

    rewriter.replaceOp(op, generic->getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertReduceOp (pure_fn, no region) → linalg.reduce
//===----------------------------------------------------------------------===//

struct ConvertReduceOp : public OpRewritePattern<ReduceOp> {
  using OpRewritePattern<ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getPureFnAttr() || !op.getBody().empty())
      return failure();

    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto fn = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        op, op.getPureFnAttr());
    if (!fn)
      return failure();

    auto inputType = cast<RankedTensorType>(op.getInput().getType());

    // Build a linalg.reduce with the pure_fn body cloned into its combiner
    // region.
    auto reduce = linalg::ReduceOp::create(
        rewriter, op.getLoc(),
        /*inputs=*/ValueRange{op.getInput()},
        /*inits=*/ValueRange{op.getOutput()},
        /*dimensions=*/
        llvm::to_vector(llvm::seq<int64_t>(0, inputType.getRank())),
        [&](OpBuilder &bodyBuilder, Location loc, ValueRange blockArgs) {
          // blockArgs: [accumulator, element]
          IRMapping mapper;
          mapper.map(fn.getArgument(0), blockArgs[0]);
          mapper.map(fn.getArgument(1), blockArgs[1]);
          for (auto &bodyOp : fn.getBody().front().without_terminator())
            bodyBuilder.clone(bodyOp, mapper);
          auto retOp = cast<func::ReturnOp>(
              fn.getBody().front().getTerminator());
          Value mappedRet = mapper.lookupOrDefault(retOp.getOperand(0));
          linalg::YieldOp::create(bodyBuilder, loc, mappedRet);
        });

    // Preserve skeleton.preference.
    if (auto pref = op.getPreferenceAttr())
      reduce->setDiscardableAttr("skeleton.preference", pref);

    rewriter.replaceOp(op, reduce->getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertVectorAddOp → linalg.add
//===----------------------------------------------------------------------===//

struct ConvertVectorAddOp : public OpRewritePattern<VectorAddOp> {
  using OpRewritePattern<VectorAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(VectorAddOp op,
                                PatternRewriter &rewriter) const override {
    auto addOp = linalg::AddOp::create(
        rewriter, op.getLoc(), op.getResult().getType(),
        ValueRange{op.getLhs(), op.getRhs()}, op.getOutput());

    if (auto pref = op.getPreferenceAttr())
      addOp->setDiscardableAttr("skeleton.preference", pref);

    rewriter.replaceOp(op, addOp->getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class SkeletonToLinalgPass
    : public impl::SkeletonToLinalgBase<SkeletonToLinalgPass> {
public:
  using impl::SkeletonToLinalgBase<SkeletonToLinalgPass>::SkeletonToLinalgBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertMapOp, ConvertReduceOp, ConvertVectorAddOp>(
        &getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createSkeletonToLinalgPass() {
  return std::make_unique<SkeletonToLinalgPass>();
}

} // namespace skeleton
} // namespace mlir
