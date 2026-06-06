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

def SkeletonTargetLower : Pass<"skeleton-target-lower"> {
  let summary = "Lower ops based on skeleton.target attribute";
  let description = [{
    Lowers linalg.matmul ops in functions marked with skeleton.target
    attribute. CPU targets use -convert-linalg-to-loops (scf.for),
    while GPU targets use -convert-linalg-to-parallel-loops (scf.parallel).
    Both paths are preceded by one-shot-bufferize.
  }];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::skeleton::SkeletonDialect",
    "mlir::bufferization::BufferizationDialect"
  ];
}

def SkeletonPreferencePartition : Pass<"skeleton-preference-partition"> {
  let summary = "Partition ops by skeleton.preference into separate functions";
  let description = [{
    Partitions linalg.matmul ops that carry a skeleton.preference attribute
    into separate func.func ops. Each outlined function is marked with a
    skeleton.target attribute ("CPU" or "GPU") and the original op is
    replaced with a func.call.
  }];
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::skeleton::SkeletonDialect"
  ];
}

#endif // MLIR_DIALECT_SKELETON_PASSES
