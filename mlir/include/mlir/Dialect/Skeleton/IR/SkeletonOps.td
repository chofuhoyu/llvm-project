//===- SkeletonOps.td - Skeleton dialect operations ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SKELETON_OPS
#define SKELETON_OPS

include "SkeletonDialect.td"
include "SkeletonAttrs.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Skeleton CustomMatmulOp
//===----------------------------------------------------------------------===//

def Skeleton_CustomMatmulOp : Op<Skeleton_Dialect, "custom_matmul"> {
  let summary = "A skeleton custom matrix multiplication operation";
  let description = [{
    Represents a matrix multiplication pattern annotated by the programmer.
    Carries a skeleton.preference attribute that indicates the desired target.
    This operation is lowered to linalg.matmul by the skeleton-to-linalg pass.
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs,
    AnyRankedTensor:$output,
    OptionalAttr<Skeleton_PreferenceAttr>:$preference
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    (`preference` `=` $preference^)?
    `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    `->` type($result)
    attr-dict
  }];

  let hasVerifier = 1;
}

#endif // SKELETON_OPS
