//===- SkeletonDialect.cpp - Skeleton dialect implementation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Skeleton/IR/SkeletonDialect.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonAttrs.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::skeleton;

// PreferenceAttr

LogicalResult
PreferenceAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       StringAttr value) {
  if (value != "CPU" && value != "GPU")
    return emitError() << "expected 'CPU' or 'GPU' for preference attribute, got '"
                       << value.getValue() << "'";
  return success();
}

// Include the full attribute definitions (storage class + implementations)
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Skeleton/IR/SkeletonAttrs.cpp.inc"

// Include the dialect declarations.
#include "mlir/Dialect/Skeleton/IR/SkeletonOpsDialect.cpp.inc"

// SkeletonDialect

void SkeletonDialect::initialize() {
  registerAttributes();
  registerOperations();
}

void SkeletonDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Skeleton/IR/SkeletonAttrs.cpp.inc"
      >();
}

void SkeletonDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Skeleton/IR/SkeletonOps.cpp.inc"
      >();
}
