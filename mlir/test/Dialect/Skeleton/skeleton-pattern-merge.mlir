// RUN: mlir-opt %s -skeleton-pattern-merge -split-input-file -verify-diagnostics | FileCheck %s

// Test: skeleton.map {pure_fn = @add_fn} → skeleton.vector_add

func.func @add_fn(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @test_merge_add(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %r = skeleton.map pure_fn = @add_fn
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}

// CHECK-LABEL: func @test_merge_add
// CHECK: skeleton.vector_add
// CHECK-NOT: skeleton.map

// -----

// Test: skeleton.map with mul function is NOT merged (no named op for mul yet)

func.func @mul_fn(%a: f32, %b: f32) -> f32 {
  %r = arith.mulf %a, %b : f32
  return %r : f32
}

func.func @test_no_merge_mul(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %r = skeleton.map pure_fn = @mul_fn
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}

// CHECK-LABEL: func @test_no_merge_mul
// CHECK: skeleton.map
// CHECK-NOT: skeleton.vector_add

// -----

// Test: skeleton.map with preference preserved after merge

func.func @add_fn(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @test_merge_with_pref(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %r = skeleton.map preference = #skeleton.preference<"GPU"> pure_fn = @add_fn
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}

// CHECK-LABEL: func @test_merge_with_pref
// CHECK: skeleton.vector_add preference = <"GPU">
// CHECK-NOT: skeleton.map
