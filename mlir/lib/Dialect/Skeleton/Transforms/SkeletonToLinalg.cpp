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

/// Clone a skeleton op's body region into a linalg region and map its block
/// arguments to \p blockArgs. The map caller passes the linalg.generic order,
/// which is already the skeleton order (inputs followed by init); the reduce
/// caller swaps because linalg.reduce uses (element, accumulator). The
/// terminator becomes linalg.yield.
static void cloneSkeletonBody(Region &body, OpBuilder &builder, Location loc,
                              ValueRange blockArgs) {
  Block &entry = body.front();
  IRMapping mapper;
  for (unsigned i = 0; i < blockArgs.size(); ++i)
    mapper.map(entry.getArgument(i), blockArgs[i]);
  for (Operation &op : entry.without_terminator())
    builder.clone(op, mapper);
  auto yield = cast<YieldOp>(entry.getTerminator());
  Value value = mapper.lookupOrDefault(yield.getValue());
  linalg::YieldOp::create(builder, loc, value);
}

// ConvertMapOp (pure_fn or body region) → linalg.generic

struct ConvertMapOp : public OpRewritePattern<MapOp> {
  using OpRewritePattern<MapOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MapOp op,
                                PatternRewriter &rewriter) const override {
    bool hasPureFn = op.getPureFnAttr() != nullptr;
    bool hasBody = !op.getBody().empty();
    if (hasPureFn == hasBody)
      return failure();

    func::FuncOp pureFn;
    if (hasPureFn) {
      auto module = op->getParentOfType<ModuleOp>();
      if (!module)
        return failure();

      pureFn = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          op, op.getPureFnAttr());
      if (!pureFn)
        return failure();

      // linalg regions must be a single straight-line block: reject pure_fn
      // bodies we cannot represent.
      if (pureFn.isDeclaration())
        return op.emitOpError("pure_fn '")
               << pureFn.getName() << "' must be defined to lower to linalg";
      if (!pureFn.getBody().hasOneBlock())
        return op.emitOpError("pure_fn '")
               << pureFn.getName()
               << "' must have a single-block body to lower to linalg";
    }

    // Compute indexing maps and iterator types from the result tensor.
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    unsigned rank = resultType.getRank();
    auto *ctx = rewriter.getContext();

    // An input of the result's rank maps identically onto it; a lower-rank
    // input is broadcast by matching its dimensions to the leading ones (a
    // rank-0 input therefore maps to no dimension at all). The map is the
    // identity on the result's dimensions restricted to the first
    // operandRank of them.
    auto broadcastMap = [&](unsigned operandRank) {
      SmallVector<AffineExpr> dims;
      for (unsigned d = 0; d < operandRank; ++d)
        dims.push_back(rewriter.getAffineDimExpr(d));
      return AffineMap::get(rank, /*symbolCount=*/0, dims, ctx);
    };

    SmallVector<AffineMap> indexingMaps;
    for (Value input : op.getInputs())
      indexingMaps.push_back(
          broadcastMap(cast<RankedTensorType>(input.getType()).getRank()));
    // The destination is the result: identity.
    indexingMaps.push_back(AffineMap::getMultiDimIdentityMap(rank, ctx));

    // All parallel iterators for element-wise ops.
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    auto generic = linalg::GenericOp::create(
        rewriter, op.getLoc(),
        /*resultTensorTypes=*/resultType,
        /*inputs=*/op.getInputs(),
        /*outputs=*/ValueRange{op.getInit()},
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        /*doc=*/"",
        /*libraryCall=*/"",
        [&](OpBuilder &bodyBuilder, Location loc, ValueRange blockArgs) {
          if (hasBody) {
            cloneSkeletonBody(op.getBody(), bodyBuilder, loc, blockArgs);
            return;
          }
          // The pure_fn body is single-block (checked above). Map blockArgs
          // to pure_fn parameters and clone the body.
          IRMapping mapper;
          for (unsigned i = 0; i < blockArgs.size() - 1; ++i)
            mapper.map(pureFn.getArgument(i), blockArgs[i]);
          for (auto &bodyOp : pureFn.getBody().front().without_terminator())
            bodyBuilder.clone(bodyOp, mapper);
          auto retOp =
              cast<func::ReturnOp>(pureFn.getBody().front().getTerminator());
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

// ConvertReduceOp (pure_fn or body region) → linalg.reduce

struct ConvertReduceOp : public OpRewritePattern<ReduceOp> {
  using OpRewritePattern<ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const override {
    bool hasPureFn = op.getPureFnAttr() != nullptr;
    bool hasBody = !op.getBody().empty();
    if (hasPureFn == hasBody)
      return failure();

    func::FuncOp pureFn;
    if (hasPureFn) {
      auto module = op->getParentOfType<ModuleOp>();
      if (!module)
        return failure();

      pureFn = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          op, op.getPureFnAttr());
      if (!pureFn)
        return failure();

      // linalg regions must be a single straight-line block: reject pure_fn
      // bodies we cannot represent.
      if (pureFn.isDeclaration())
        return op.emitOpError("pure_fn '")
               << pureFn.getName() << "' must be defined to lower to linalg";
      if (!pureFn.getBody().hasOneBlock())
        return op.emitOpError("pure_fn '")
               << pureFn.getName()
               << "' must have a single-block body to lower to linalg";
    }

    auto inputType = cast<RankedTensorType>(op.getInput().getType());
    auto initType = cast<RankedTensorType>(op.getInit().getType());

    // The reduction covers the trailing dimensions the init does not have: the
    // init keeps a prefix of the input's dimensions, so an init of rank 0
    // reduces every element.
    SmallVector<int64_t> dimensions;
    for (int64_t d = initType.getRank(); d < inputType.getRank(); ++d)
      dimensions.push_back(d);

    // Build a linalg.reduce with the pure_fn body cloned into its combiner
    // region.
    auto reduce = linalg::ReduceOp::create(
        rewriter, op.getLoc(),
        /*inputs=*/ValueRange{op.getInput()},
        /*inits=*/ValueRange{op.getInit()},
        /*dimensions=*/dimensions,
        [&](OpBuilder &bodyBuilder, Location loc, ValueRange blockArgs) {
          if (hasBody) {
            // skeleton.reduce body arguments are (accumulator, element);
            // linalg.reduce combiner arguments are (element, accumulator).
            ValueRange reduceArgs = {blockArgs[1], blockArgs[0]};
            cloneSkeletonBody(op.getBody(), bodyBuilder, loc, reduceArgs);
            return;
          }
          // The pure_fn body is single-block (checked above). Its first
          // argument is the accumulator and its second is the element, which
          // is the reverse of the linalg.reduce combiner order.
          IRMapping mapper;
          mapper.map(pureFn.getArgument(0), blockArgs[1]);
          mapper.map(pureFn.getArgument(1), blockArgs[0]);
          for (auto &bodyOp : pureFn.getBody().front().without_terminator())
            bodyBuilder.clone(bodyOp, mapper);
          auto retOp =
              cast<func::ReturnOp>(pureFn.getBody().front().getTerminator());
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

// ConvertVectorAddOp → linalg.add

struct ConvertVectorAddOp : public OpRewritePattern<VectorAddOp> {
  using OpRewritePattern<VectorAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(VectorAddOp op,
                                PatternRewriter &rewriter) const override {
    auto addOp = linalg::AddOp::create(
        rewriter, op.getLoc(), op.getResult().getType(),
        ValueRange{op.getLhs(), op.getRhs()}, op.getInit());

    if (auto pref = op.getPreferenceAttr())
      addOp->setDiscardableAttr("skeleton.preference", pref);

    rewriter.replaceOp(op, addOp->getResults());
    return success();
  }
};

// Pass

class SkeletonToLinalgPass
    : public impl::SkeletonToLinalgBase<SkeletonToLinalgPass> {
public:
  using impl::SkeletonToLinalgBase<SkeletonToLinalgPass>::SkeletonToLinalgBase;

  void runOnOperation() override {
    // A body region may contain scalar arithmetic, but a nested skeleton
    // operator is a separate computation graph that cannot be lowered inside
    // an element-wise linalg body. Reject it here, where the diagnostic points
    // at the outer operator, instead of leaving an unlowerable op behind.
    WalkResult check = getOperation()->walk([](Operation *op) {
      if (!isa<MapOp, ReduceOp>(op) || op->getRegion(0).empty())
        return WalkResult::advance();
      WalkResult nested = op->getRegion(0).walk([](Operation *inner) {
        if (isa<MapOp, ReduceOp, VectorAddOp>(inner))
          return WalkResult::interrupt();
        return WalkResult::advance();
      });
      if (nested.wasInterrupted()) {
        op->emitError(
            "nested skeleton operators in a body region are not supported");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (check.wasInterrupted()) {
      signalPassFailure();
      return;
    }

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
