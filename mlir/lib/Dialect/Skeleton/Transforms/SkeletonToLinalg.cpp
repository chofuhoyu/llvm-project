//===- SkeletonToLinalg.cpp - Skeleton to Linalg conversion -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Skeleton/IR/SkeletonOps.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonDialect.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonAttrs.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace skeleton {

#define GEN_PASS_DEF_SKELETONTOLINALG
#include "mlir/Dialect/Skeleton/Transforms/Passes.h.inc"

namespace {

struct ConvertCustomMatmul : public OpRewritePattern<CustomMatmulOp> {
  using OpRewritePattern<CustomMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CustomMatmulOp op,
                                PatternRewriter &rewriter) const override {
    auto matmul = rewriter.create<linalg::MatmulOp>(
        op.getLoc(), op.getResult().getType(),
        ValueRange{op.getLhs(), op.getRhs()}, op.getOutput());

    // Preserve the preference attribute as a discardable attribute
    if (auto pref = op.getPreferenceAttr()) {
      matmul->setAttr("skeleton.preference", pref);
    }

    rewriter.replaceOp(op, matmul->getResults());
    return success();
  }
};

class SkeletonToLinalgPass
    : public impl::SkeletonToLinalgBase<SkeletonToLinalgPass> {
public:
  using impl::SkeletonToLinalgBase<SkeletonToLinalgPass>::SkeletonToLinalgBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertCustomMatmul>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(),
                                     std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createSkeletonToLinalgPass() {
  return std::make_unique<SkeletonToLinalgPass>();
}

} // namespace skeleton
} // namespace mlir
