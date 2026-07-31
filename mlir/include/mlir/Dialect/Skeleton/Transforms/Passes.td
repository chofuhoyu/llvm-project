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
    "mlir::skeleton::SkeletonDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect"
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
    "mlir::bufferization::BufferizationDialect",
    "mlir::memref::MemRefDialect",
    "mlir::func::FuncDialect",
    "mlir::gpu::GPUDialect"
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

def SkeletonFinalizeMemRefToGpu : Pass<"skeleton-finalize-memref-to-gpu"> {
  let summary = "Lower GPU memory lifecycle ops to LLVM";
  let description = [{
    Finalizes GPU memory lifecycle operations inserted by skeleton-target-lower
    into LLVM IR. Converts gpu.alloc host_shared to mgpuMemAlloc runtime calls,
    gpu.dealloc to mgpuMemFree, and memref.copy to llvm.intr.memcpy.
    GPU kernel functions (marked with skeleton.target) are left untouched for
    the subsequent GPU lowering pipeline.
  }];
  let dependentDialects = [
    "mlir::skeleton::SkeletonDialect",
    "mlir::memref::MemRefDialect",
    "mlir::gpu::GPUDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

def SkeletonPatternMerge : Pass<"skeleton-pattern-merge"> {
  let summary = "Merge skeleton.map + pure_fn patterns into named ops";
  let description = [{
    Recognizes skeleton.map ops with simple pure functions and merges them
    into named skeleton ops for further optimization.

    Currently supports:
    - skeleton.map {pure_fn = @add_fn} → skeleton.vector_add
      (where @add_fn body is a single arith.addf of its arguments)
  }];
  let dependentDialects = [
    "mlir::skeleton::SkeletonDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect"
  ];
}

#endif // MLIR_DIALECT_SKELETON_PASSES
