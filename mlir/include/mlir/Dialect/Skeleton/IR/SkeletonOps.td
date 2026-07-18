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

//===----------------------------------------------------------------------===//
// Skeleton VectorAddOp
//===----------------------------------------------------------------------===//

def Skeleton_VectorAddOp : Op<Skeleton_Dialect, "vector_add"> {
  let summary = "A skeleton vector addition operation";
  let description = [{
    Represents an element-wise vector addition (C[i] = A[i] + B[i])
    annotated by the programmer. Carries a skeleton.preference attribute
    indicating the desired execution target. This operation is lowered to
    linalg.add by the skeleton-to-linalg pass.

    Example:
    ```
    %0 = skeleton.vector_add preference = #skeleton.preference<"CPU">
        ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
        outs(%out : tensor<8xf32>) -> tensor<8xf32>
    ```
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

//===----------------------------------------------------------------------===//
// Skeleton AnnotateOp
//===----------------------------------------------------------------------===//

def Skeleton_AnnotateOp : Op<Skeleton_Dialect, "annotate"> {
  let summary = "A skeleton annotation marker carrying frontend annotation info";
  let description = [{
    Marker operation that carries annotation metadata from C/C++ source
    through the compilation pipeline. Carries a skeleton.preference attribute
    and the original annotation name.

    This is the intermediate representation between CIR frontend annotations
    and structured computation operations like `skeleton.vector_add`. A
    subsequent skeleton pass can recognize specific annotation names and
    produce more specific operations.

    Example:
    ```
    skeleton.annotate preference = #skeleton.preference<"GPU">
      name = "skeleton_vector_add"
    ```
  }];

  let arguments = (ins
    OptionalAttr<Skeleton_PreferenceAttr>:$preference,
    OptionalAttr<StrAttr>:$annotation_name
  );

  let assemblyFormat = [{
    (`preference` `=` $preference^)?
    (`name` `=` $annotation_name^)?
    attr-dict
  }];
}

#endif // SKELETON_OPS
