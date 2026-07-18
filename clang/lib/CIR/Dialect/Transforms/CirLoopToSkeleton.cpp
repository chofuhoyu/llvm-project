//===- CirLoopToSkeleton.cpp - Annotated CIR loops to Skeleton -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts cir.for, cir.while, and cir.do operations that carry
// cir.annotation attributes into skeleton.annotate operations.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonAttrs.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonDialect.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_CIRLOOPTOSKELETON
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

/// Returns the first cir::AnnotationAttr from the op's annotations attribute.
static cir::AnnotationAttr getFirstAnnotation(Operation *op) {
  auto annotations = op->getAttrOfType<ArrayAttr>("annotations");
  if (!annotations || annotations.empty())
    return {};
  return dyn_cast<cir::AnnotationAttr>(annotations[0]);
}

/// Structure to hold data extracted from an annotated loop operation.
struct AnnotatedLoopInfo {
  cir::AnnotationAttr annotation;
  StringRef name;
  StringRef preference;
  Region *bodyRegion;
};

/// Extract annotation info from a loop op. Returns true if the op carries
/// a valid annotation.
static bool extractLoopInfo(Operation *op, AnnotatedLoopInfo &info) {
  auto ann = getFirstAnnotation(op);
  if (!ann)
    return false;

  info.annotation = ann;
  info.name = ann.getName().getValue();

  // Extract preference from the first string argument, if present.
  if (auto args = ann.getArgs()) {
    if (!args.empty()) {
      if (auto firstArg = dyn_cast<StringAttr>(args[0]))
        info.preference = firstArg.getValue();
    }
  }
  // Default preference.
  if (info.preference.empty())
    info.preference = "CPU";

  return true;
}

/// Pattern to convert an annotated cir::ForOp to skeleton.annotate.
struct ConvertAnnotatedForOp : public OpRewritePattern<cir::ForOp> {
  using OpRewritePattern<cir::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cir::ForOp op,
                                PatternRewriter &rewriter) const override {
    AnnotatedLoopInfo info;
    if (!extractLoopInfo(op, info))
      return failure();

    // Create skeleton.annotate as a marker before the loop.
    rewriter.setInsertionPoint(op);
    auto annotateOp = rewriter.create<skeleton::AnnotateOp>(
        op.getLoc(),
        /*preference=*/nullptr,
        /*annotation_name=*/nullptr);

    if (!info.name.empty())
      annotateOp.setAnnotationNameAttr(
          rewriter.getStringAttr(info.name));

    if (!info.preference.empty()) {
      annotateOp.setPreferenceAttr(skeleton::PreferenceAttr::get(
          getContext(), rewriter.getStringAttr(info.preference)));
    }

    return success();
  }
};

/// Pattern to recognize an annotated cir::WhileOp.
struct ConvertAnnotatedWhileOp : public OpRewritePattern<cir::WhileOp> {
  using OpRewritePattern<cir::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cir::WhileOp op,
                                PatternRewriter &rewriter) const override {
    AnnotatedLoopInfo info;
    if (!extractLoopInfo(op, info))
      return failure();

    rewriter.setInsertionPoint(op);
    auto annotateOp = rewriter.create<skeleton::AnnotateOp>(
        op.getLoc(),
        /*preference=*/nullptr,
        /*annotation_name=*/nullptr);
    if (!info.name.empty())
      annotateOp.setAnnotationNameAttr(rewriter.getStringAttr(info.name));
    if (!info.preference.empty())
      annotateOp.setPreferenceAttr(skeleton::PreferenceAttr::get(
          getContext(), rewriter.getStringAttr(info.preference)));
    return success();
  }
};

/// Pattern to recognize an annotated cir::DoWhileOp.
struct ConvertAnnotatedDoWhileOp : public OpRewritePattern<cir::DoWhileOp> {
  using OpRewritePattern<cir::DoWhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cir::DoWhileOp op,
                                PatternRewriter &rewriter) const override {
    AnnotatedLoopInfo info;
    if (!extractLoopInfo(op, info))
      return failure();

    rewriter.setInsertionPoint(op);
    auto annotateOp = rewriter.create<skeleton::AnnotateOp>(
        op.getLoc(),
        /*preference=*/nullptr,
        /*annotation_name=*/nullptr);
    if (!info.name.empty())
      annotateOp.setAnnotationNameAttr(rewriter.getStringAttr(info.name));
    if (!info.preference.empty())
      annotateOp.setPreferenceAttr(skeleton::PreferenceAttr::get(
          getContext(), rewriter.getStringAttr(info.preference)));
    return success();
  }
};

class CIRLoopToSkeletonPass
    : public impl::CIRLoopToSkeletonBase<CIRLoopToSkeletonPass> {
public:
  using CIRLoopToSkeletonBase::CIRLoopToSkeletonBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    getOperation()->walk([&](Operation *op) {
      AnnotatedLoopInfo info;
      if (!extractLoopInfo(op, info))
        return;

      OpBuilder builder(ctx);
      builder.setInsertionPoint(op);
      auto annotateOp = builder.create<skeleton::AnnotateOp>(
          op->getLoc(),
          /*preference=*/nullptr,
          /*annotation_name=*/nullptr);
      if (!info.name.empty())
        annotateOp.setAnnotationNameAttr(
            builder.getStringAttr(info.name));
      if (!info.preference.empty())
        annotateOp.setPreferenceAttr(skeleton::PreferenceAttr::get(
            ctx, builder.getStringAttr(info.preference)));
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createCIRLoopToSkeletonPass() {
  return std::make_unique<CIRLoopToSkeletonPass>();
}
