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
// Skeleton MapOp
//===----------------------------------------------------------------------===//

def Skeleton_MapOp : Op<Skeleton_Dialect, "map"> {
  let summary = "A skeleton map operation applying a pure function element-wise";
  let description = [{
    Applies a pure function (referenced via `pure_fn` attribute) to input
    tensors element-wise, or executes a nested skeleton computation graph
    provided in the optional `body` region.

    Exactly one of `pure_fn` or `body` must be present.

    Example (leaf, pure_fn):
    ```mlir
    %r = skeleton.map {preference = #skeleton.preference<"CPU">, pure_fn = @my_add}
      ins(%A, %B : tensor<?xf32>, tensor<?xf32>)
      outs(%C : tensor<?xf32>) -> tensor<?xf32>
    ```

    Example (nested, region body):
    ```mlir
    %r = skeleton.map {preference = #skeleton.preference<"GPU">}
      ins(%matrix : tensor<?x?xf32>)
      outs(%output : tensor<?xf32>) {
    ^bb0(%row: tensor<?xf32>, %init: tensor<f32>):
      %sub = skeleton.reduce {pure_fn = @my_sum}
        ins(%row : tensor<?xf32>)
        outs(%init : tensor<f32>) -> tensor<f32>
      skeleton.yield %sub : tensor<f32>
    } -> tensor<?xf32>
    ```
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    AnyRankedTensor:$output,
    OptionalAttr<FlatSymbolRefAttr>:$pure_fn,
    OptionalAttr<Skeleton_PreferenceAttr>:$preference
  );

  let results = (outs AnyRankedTensor:$result);

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    (`preference` `=` $preference^)?
    (`pure_fn` `=` $pure_fn^)?
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $output `:` type($output) `)`
    ($body^)?
    `->` type($result)
    attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Skeleton ReduceOp
//===----------------------------------------------------------------------===//

def Skeleton_ReduceOp : Op<Skeleton_Dialect, "reduce"> {
  let summary = "A skeleton reduce operation applying a pure function as reduction";
  let description = [{
    Reduces an input tensor along its elements using a pure function
    (referenced via `pure_fn` attribute), or executes a nested skeleton
    computation graph provided in the optional `body` region.

    Exactly one of `pure_fn` or `body` must be present.

    The pure function must be a binary associative operation: `(T, T) -> T`.

    Example:
    ```mlir
    %r = skeleton.reduce {preference = #skeleton.preference<"CPU">, pure_fn = @my_add}
      ins(%input : tensor<?xf32>)
      outs(%init : tensor<f32>) -> tensor<f32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$output,
    OptionalAttr<FlatSymbolRefAttr>:$pure_fn,
    OptionalAttr<Skeleton_PreferenceAttr>:$preference
  );

  let results = (outs AnyRankedTensor:$result);

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    (`preference` `=` $preference^)?
    (`pure_fn` `=` $pure_fn^)?
    `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    ($body^)?
    `->` type($result)
    attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Skeleton YieldOp
//===----------------------------------------------------------------------===//

def Skeleton_YieldOp : Op<Skeleton_Dialect, "yield", [
  Terminator, HasParent<"MapOp, ReduceOp">
]> {
  let summary = "Yield a value from a skeleton region body";
  let description = [{
    Terminates a skeleton operation's region body and yields the result
    value back to the parent operation.
  }];

  let arguments = (ins AnyType:$value);

  let assemblyFormat = "$value attr-dict `:` type($value)";
}

//===----------------------------------------------------------------------===//
// Skeleton VectorAddOp (named op, product of pattern merge)
//===----------------------------------------------------------------------===//

def Skeleton_VectorAddOp : Op<Skeleton_Dialect, "vector_add"> {
  let summary = "A named skeleton vector addition operation";
  let description = [{
    Represents an element-wise vector addition (C[i] = A[i] + B[i]).
    This is a named operation produced by merging a skeleton.map with
    an arith.addf pure function. It carries a skeleton.preference
    attribute indicating the desired execution target.

    This operation is lowered to linalg.add by the skeleton-to-linalg pass.

    Example:
    ```
    %0 = skeleton.vector_add preference = #skeleton.preference<"CPU">
      ins(%lhs, %rhs : tensor<?xf32>, tensor<?xf32>)
      outs(%out : tensor<?xf32>) -> tensor<?xf32>
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

#endif // SKELETON_OPS
