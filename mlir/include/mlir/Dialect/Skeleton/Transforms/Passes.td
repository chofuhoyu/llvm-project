//===- Passes.td - Skeleton pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SKELETON_PASSES
#define MLIR_DIALECT_SKELETON_PASSES

include "mlir/Pass/PassBase.td"

def SkeletonToLinalg : Pass<"skeleton-to-linalg"> {
  let summary = "Convert Skeleton ops to Linalg ops";
  let description = [{
    Converts skeleton.custom_matmul operations to linalg.matmul operations.
    The skeleton.preference attribute is preserved as a discardable attribute
    on the resulting linalg op.
  }];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::skeleton::SkeletonDialect"
  ];
}

#endif // MLIR_DIALECT_SKELETON_PASSES
