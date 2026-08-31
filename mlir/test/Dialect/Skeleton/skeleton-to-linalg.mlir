// RUN: mlir-opt %s -skeleton-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// Test: skeleton.map (pure_fn, no body) → linalg.generic

func.func @my_add(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @test_map_leaf(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %0 = skeleton.map pure_fn = @my_add
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>)
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @test_map_leaf
// CHECK: linalg.generic
// CHECK-SAME: ins({{.*}} : tensor<8xf32>, tensor<8xf32>)
// CHECK-SAME: outs({{.*}} : tensor<8xf32>)
// CHECK-NOT: skeleton.map

// -----

// Test: skeleton.map with preference preserved on linalg.generic

func.func @my_add(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @test_map_with_pref(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %0 = skeleton.map preference = #skeleton.preference<"GPU"> pure_fn = @my_add
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>)
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @test_map_with_pref
// CHECK: linalg.generic
// CHECK-SAME: skeleton.preference = #skeleton.preference<"GPU">
// CHECK-NOT: skeleton.map

// -----

// Test: skeleton.reduce (pure_fn, no body) → linalg.reduce

func.func @my_sum(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @test_reduce_leaf(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = skeleton.reduce pure_fn = @my_sum
    ins(%arg0 : tensor<8xf32>)
    outs(%arg1 : tensor<f32>)
  return %0 : tensor<f32>
}

// CHECK-LABEL: func @test_reduce_leaf
// CHECK: linalg.reduce
// CHECK-SAME: ins({{.*}} : tensor<8xf32>)
// CHECK-SAME: outs({{.*}} : tensor<f32>)
// CHECK-NOT: skeleton.reduce

// -----

// Test: skeleton.reduce with preference preserved

func.func @my_sum(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @test_reduce_with_pref(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = skeleton.reduce preference = #skeleton.preference<"CPU"> pure_fn = @my_sum
    ins(%arg0 : tensor<8xf32>)
    outs(%arg1 : tensor<f32>)
  return %0 : tensor<f32>
}

// CHECK-LABEL: func @test_reduce_with_pref
// CHECK: linalg.reduce
// CHECK-SAME: skeleton.preference = #skeleton.preference<"CPU">
// CHECK-NOT: skeleton.reduce

// -----

// Test: skeleton.vector_add → linalg.add

func.func @test_vector_add(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %0 = skeleton.vector_add
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>)
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @test_vector_add
// CHECK: linalg.add
// CHECK-SAME: ins({{.*}} : tensor<8xf32>, tensor<8xf32>)
// CHECK-SAME: outs({{.*}} : tensor<8xf32>)
// CHECK-NOT: skeleton.vector_add

// -----

// Test: skeleton.vector_add with preference → linalg.add with attr

func.func @test_vector_add_with_pref(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %0 = skeleton.vector_add preference = #skeleton.preference<"GPU">
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>)
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @test_vector_add_with_pref
// CHECK: linalg.add
// CHECK-SAME: skeleton.preference = #skeleton.preference<"GPU">
// CHECK-NOT: skeleton.vector_add
