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
  let summary = "Lower an operator unit marked with skeleton.target";
  let description = [{
    Lowers the linalg ops of a function marked with skeleton.target. CPU
    targets use -convert-linalg-to-loops (scf.for), GPU targets use
    -convert-linalg-to-parallel-loops (scf.parallel) and get a Unified Memory
    bridge around their call. Both paths are preceded by one-shot-bufferize.

    The runtime driver uses this pass to compile one operator unit for the
    device it placed it on; it no longer makes placement decisions itself.
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

def SkeletonPatternMerge : Pass<"skeleton-pattern-merge"> {
  let summary = "Merge skeleton.map + pure_fn patterns into named ops";
  let description = [{
    Recognizes skeleton.map ops with simple pure functions and merges them
    into named skeleton ops for further optimization.

    Currently supports:
    - skeleton.map {pure_fn = @add_fn} → skeleton.vector.add
      (where @add_fn body is a single arith.addf of its arguments)
  }];
  let dependentDialects = [
    "mlir::skeleton::SkeletonDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect"
  ];
}

def SkeletonOutline : Pass<"skeleton-outline"> {
  let summary = "Outline skeleton operators into op unit functions";
  let description = [{
    Turns every operator unit into its own function so that the runtime can
    decide where it runs and recompile it on demand. An operator unit is a
    skeleton operator (map, reduce, vector.add) or a linalg operator carrying
    a skeleton.preference attribute.

    Each function keeps the operator in its skeleton form and records
    skeleton.op_id, skeleton.preference and skeleton.bytes, so the driver can
    re-enter compilation from skeleton IR for that operator alone.
  }];
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::skeleton::SkeletonDialect"
  ];
}

def SkeletonLowerDispatch : Pass<"skeleton-lower-dispatch"> {
  let summary = "Lower op unit calls to runtime resolves";
  let description = [{
    Runs on a module already lowered to the LLVM dialect. Every direct call to
    an outlined op unit becomes a skeleton_rt_resolve call followed by an
    indirect call with the operator's own signature, so the runtime can hand
    back the implementation chosen for this invocation.
  }];
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::func::FuncDialect"
  ];
}

#endif // MLIR_DIALECT_SKELETON_PASSES
