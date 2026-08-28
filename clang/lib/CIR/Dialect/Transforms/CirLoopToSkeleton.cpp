//===- CirLoopToSkeleton.cpp - Annotated CIR loops to Skeleton -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Semi-automatic path (TODO): Converts annotated CIR loops (cir.for, cir.while,
// cir.do) carrying cir.annotation attributes into Skeleton dialect structured
// computation operations.
//
// The semi-automatic path recognizes annotated for-loops from C/C++ source
// code and maps them to skeleton operators. This pass currently only detects
// annotations and emits a diagnostic for unsupported patterns.
//
// The full manual path (skeleton helper function calls → skeleton ops) is
// implemented in CirCallToSkeleton.cpp.
//
// TODO items for semi-automatic path:
//  - Pattern-match loop bodies to produce skeleton.map / skeleton.reduce
//  - Auto-extract inline computation from loop body into a pure func.func
//  - Support lambda syntax for inline pure functions
//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
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

/// Extract loop annotation info. Returns true if the op carries a valid
/// annotation.
struct AnnotatedLoopInfo {
  cir::AnnotationAttr annotation;
  StringRef name;
  StringRef preference;
};

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

class CIRLoopToSkeletonPass
    : public impl::CIRLoopToSkeletonBase<CIRLoopToSkeletonPass> {
public:
  using CIRLoopToSkeletonBase::CIRLoopToSkeletonBase;

  void runOnOperation() override {
    getOperation()->walk([&](Operation *op) {
      AnnotatedLoopInfo info;
      if (!extractLoopInfo(op, info))
        return;

      // TODO: Semi-automatic path — pattern-match the loop body to produce
      // skeleton.map / skeleton.reduce operations.
      //
      // The semi-automatic path is not yet implemented. For now, annotated
      // loops that use the for-loop + annotation pattern emit a warning.
      // Use the full manual path (skeleton helper function calls) instead.
      //
      // Future work:
      //  - Analyze loop body to recognize computation patterns
      //  - Auto-extract inline computation into a pure func.func
      //  - Produce skeleton.map / skeleton.reduce with pure_fn reference
      //  - Support lambda syntax for inline pure functions
      op->emitWarning("semi-automatic skeleton path not yet implemented; "
                      "use the full manual path (skeleton helper function "
                      "calls) instead");
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createCIRLoopToSkeletonPass() {
  return std::make_unique<CIRLoopToSkeletonPass>();
}
