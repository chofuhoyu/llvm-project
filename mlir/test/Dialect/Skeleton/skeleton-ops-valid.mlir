// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// MapOp — leaf mode (pure_fn only, no body)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @map_leaf
func.func @my_add(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @map_leaf(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK: skeleton.map {pure_fn = @my_add}
  %r = skeleton.map pure_fn = @my_add
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}

// -----

//===----------------------------------------------------------------------===//
// MapOp — with preference attribute
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @map_with_pref
func.func @map_with_pref(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK: skeleton.map {preference = #skeleton.preference<"CPU">, pure_fn = @my_add}
  %r = skeleton.map preference = #skeleton.preference<"CPU"> pure_fn = @my_add
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}

// -----

//===----------------------------------------------------------------------===//
// ReduceOp — leaf mode
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @reduce_leaf
func.func @my_sum(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @reduce_leaf(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: skeleton.reduce {pure_fn = @my_sum}
  %r = skeleton.reduce pure_fn = @my_sum
    ins(%arg0 : tensor<8xf32>)
    outs(%arg1 : tensor<f32>) -> tensor<f32>
  return %r : tensor<f32>
}

// -----

//===----------------------------------------------------------------------===//
// ReduceOp — with preference
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @reduce_with_pref
func.func @reduce_with_pref(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: skeleton.reduce {preference = #skeleton.preference<"GPU">, pure_fn = @my_sum}
  %r = skeleton.reduce preference = #skeleton.preference<"GPU"> pure_fn = @my_sum
    ins(%arg0 : tensor<8xf32>)
    outs(%arg1 : tensor<f32>) -> tensor<f32>
  return %r : tensor<f32>
}

// -----

//===----------------------------------------------------------------------===//
// MapOp — nested with region body containing ReduceOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @map_nested_reduce
func.func @map_nested_reduce(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: skeleton.map
  // CHECK: ^bb0(%{{.*}}: f32, %{{.*}}: f32):
  // CHECK:   arith.addf
  // CHECK:   skeleton.yield
  %r = skeleton.map preference = #skeleton.preference<"GPU">
    ins(%arg0 : tensor<?x?xf32>)
    outs(%arg1 : tensor<?xf32>) {
  ^bb0(%row: f32, %init: f32):
    // Simple nested: just yield the sum of row and init.
    %sum = arith.addf %row, %init : f32
    skeleton.yield %sum : f32
  } -> tensor<?xf32>
  return %r : tensor<?xf32>
}

// -----

//===----------------------------------------------------------------------===//
// VectorAddOp — dynamic shape
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @vector_add_op
func.func @vector_add_op(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: skeleton.vector_add
  %r = skeleton.vector_add preference = #skeleton.preference<"CPU">
    ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
    outs(%arg2 : tensor<?xf32>) -> tensor<?xf32>
  return %r : tensor<?xf32>
}

// -----

//===----------------------------------------------------------------------===//
// VectorAddOp — static shapes
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @vector_add_static
func.func @vector_add_static(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK: skeleton.vector_add
  %r = skeleton.vector_add
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}
