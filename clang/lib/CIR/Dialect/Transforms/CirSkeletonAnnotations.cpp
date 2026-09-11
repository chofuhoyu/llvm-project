//===- CirSkeletonAnnotations.cpp - parse skeleton annotations -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Transforms/CirSkeletonAnnotations.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "clang/CIR/Dialect/Transforms/CIRAnnotations.h"

using namespace mlir;
using namespace cir;

namespace {
// The annotation names the skeleton paths read off CIR ops. They live only
// here: consumers refer to these annotations through the typed accessors
// below. "skeleton.pure" marks pure functions, "skeleton.op" names the
// operator of a skeleton helper declaration, and "skeleton.region" marks the
// scope handed to the skeleton compiler — a host function (manual path) or an
// annotated loop (semi-automatic path) — carrying the execution preference.
constexpr char kSkeletonPure[] = "skeleton.pure";
constexpr char kSkeletonOp[] = "skeleton.op";
constexpr char kSkeletonRegion[] = "skeleton.region";

/// Parse the preference a "skeleton.region" annotation carries as its first
/// string argument, reporting on \p op with \p annName in the message.
FailureOr<SkeletonPreference> parsePreference(Operation *op,
                                              AnnotationAttr ann,
                                              StringRef annName) {
  auto args = ann.getArgs();
  if (!args || args.empty()) {
    op->emitError() << "invalid \"" << annName
                    << "\" annotation: expected one argument naming the "
                       "preference (\"CPU\" or \"GPU\")";
    return failure();
  }
  auto prefStr = dyn_cast<StringAttr>(args[0]);
  if (!prefStr) {
    op->emitError() << "invalid \"" << annName
                    << "\" annotation: the first argument must be a string "
                       "naming the preference (\"CPU\" or \"GPU\")";
    return failure();
  }
  StringRef pref = prefStr.getValue();
  if (pref == "CPU")
    return SkeletonPreference::CPU;
  if (pref == "GPU")
    return SkeletonPreference::GPU;
  op->emitError() << "invalid \"" << annName
                  << "\" annotation: unsupported preference \"" << pref
                  << R"(", expected "CPU" or "GPU")";
  return failure();
}
} // namespace

namespace cir {

bool hasSkeletonPureAnnotation(Operation *op) {
  return static_cast<bool>(getAnnotationByName(op, kSkeletonPure));
}

StringRef toString(SkeletonPreference preference) {
  switch (preference) {
  case SkeletonPreference::CPU:
    return "CPU";
  case SkeletonPreference::GPU:
    return "GPU";
  }
  llvm_unreachable("unknown skeleton preference");
}

bool hasSkeletonOpAnnotation(Operation *op) {
  return static_cast<bool>(getAnnotationByName(op, kSkeletonOp));
}

FailureOr<SkeletonOpType> getSkeletonOpType(Operation *op) {
  AnnotationAttr ann = getAnnotationByName(op, kSkeletonOp);
  if (!ann) {
    op->emitError() << "expected a \"" << kSkeletonOp
                    << "\" annotation; call hasSkeletonOpAnnotation first";
    return failure();
  }

  auto args = ann.getArgs();
  if (!args || args.empty()) {
    op->emitError() << "invalid \"" << kSkeletonOp
                    << "\" annotation: expected one argument naming the "
                       "operator (\"map\" or \"reduce\")";
    return failure();
  }
  auto typeStr = dyn_cast<StringAttr>(args[0]);
  if (!typeStr) {
    op->emitError() << "invalid \"" << kSkeletonOp
                    << "\" annotation: the first argument must be a string "
                       "naming the operator (\"map\" or \"reduce\")";
    return failure();
  }
  StringRef type = typeStr.getValue();
  if (type == "map")
    return SkeletonOpType::Map;
  if (type == "reduce")
    return SkeletonOpType::Reduce;
  op->emitError() << "invalid \"" << kSkeletonOp
                  << "\" annotation: unsupported op type \"" << type
                  << R"(", expected "map" or "reduce")";
  return failure();
}

FailureOr<SkeletonPreference> getSkeletonPreference(Operation *op) {
  AnnotationAttr ann = getAnnotationByName(op, kSkeletonRegion);
  if (!ann)
    return SkeletonPreference::CPU;
  return parsePreference(op, ann, kSkeletonRegion);
}

bool hasSkeletonRegionAnnotation(Operation *op) {
  return static_cast<bool>(getAnnotationByName(op, kSkeletonRegion));
}

} // namespace cir
