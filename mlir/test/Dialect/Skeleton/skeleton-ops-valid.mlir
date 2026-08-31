// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// MapOp — leaf mode (pure_fn only, no body)

func.func @my_add(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @map_leaf(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-LABEL: func @map_leaf
  // CHECK: skeleton.map pure_fn = @my_add
  %r = skeleton.map pure_fn = @my_add
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}

// -----

// MapOp — with preference attribute

func.func @my_add(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @map_with_pref(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-LABEL: func @map_with_pref
  // CHECK: skeleton.map preference = <"CPU"> pure_fn = @my_add
  %r = skeleton.map preference = #skeleton.preference<"CPU"> pure_fn = @my_add
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}

// -----

// ReduceOp — leaf mode

func.func @my_sum(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @reduce_leaf(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK-LABEL: func @reduce_leaf
  // CHECK: skeleton.reduce pure_fn = @my_sum
  %r = skeleton.reduce pure_fn = @my_sum
    ins(%arg0 : tensor<8xf32>)
    outs(%arg1 : tensor<f32>) -> tensor<f32>
  return %r : tensor<f32>
}

// -----

// ReduceOp — with preference

func.func @my_sum(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @reduce_with_pref(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK-LABEL: func @reduce_with_pref
  // CHECK: skeleton.reduce preference = <"GPU"> pure_fn = @my_sum
  %r = skeleton.reduce preference = #skeleton.preference<"GPU"> pure_fn = @my_sum
    ins(%arg0 : tensor<8xf32>)
    outs(%arg1 : tensor<f32>) -> tensor<f32>
  return %r : tensor<f32>
}

// -----

// MapOp — with region body

func.func @map_with_body(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-LABEL: func @map_with_body
  // CHECK: skeleton.map preference = <"GPU">
  // CHECK: ^bb0(%{{.*}}: f32, %{{.*}}: f32):
  // CHECK:   arith.addf
  // CHECK:   skeleton.yield
  %r = skeleton.map preference = #skeleton.preference<"GPU">
    ins(%arg0 : tensor<?xf32>)
    outs(%arg1 : tensor<?xf32>) {
  ^bb0(%elem: f32, %init: f32):
    %sum = arith.addf %elem, %init : f32
    skeleton.yield %sum : f32
  } -> tensor<?xf32>
  return %r : tensor<?xf32>
}

// -----

// VectorAddOp — dynamic shape

func.func @vector_add_op(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-LABEL: func @vector_add_op
  // CHECK: skeleton.vector_add preference = <"CPU">
  %r = skeleton.vector_add preference = #skeleton.preference<"CPU">
    ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
    outs(%arg2 : tensor<?xf32>) -> tensor<?xf32>
  return %r : tensor<?xf32>
}

// -----

// VectorAddOp — static shapes

func.func @vector_add_static(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-LABEL: func @vector_add_static
  // CHECK: skeleton.vector_add
  %r = skeleton.vector_add
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}
