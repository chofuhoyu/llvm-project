// RUN: mlir-opt %s -skeleton-to-linalg -split-input-file | FileCheck %s

// Test: skeleton.custom_matmul without preference → linalg.matmul
func.func @test_custom_matmul_no_pref(%arg0: tensor<16x8xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = skeleton.custom_matmul ins(%arg0, %arg1 : tensor<16x8xf32>, tensor<8x32xf32>) outs(%arg2 : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func @test_custom_matmul_no_pref
// CHECK: linalg.matmul
// CHECK-SAME: ins({{.*}} : tensor<16x8xf32>, tensor<8x32xf32>)
// CHECK-SAME: outs({{.*}} : tensor<16x32xf32>)
// CHECK-NOT: skeleton.custom_matmul

// -----

// Test: skeleton.custom_matmul with preference → linalg.matmul with preserved attr
func.func @test_custom_matmul_with_pref(%arg0: tensor<16x8xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = skeleton.custom_matmul preference = #skeleton.preference<"GPU"> ins(%arg0, %arg1 : tensor<16x8xf32>, tensor<8x32xf32>) outs(%arg2 : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func @test_custom_matmul_with_pref
// CHECK: linalg.matmul
// CHECK-SAME: skeleton.preference = #skeleton.preference<GPU>
// CHECK-NOT: skeleton.custom_matmul

// -----

// Test: multiple skeleton ops in same function all convert
func.func @test_multiple_matmuls(%arg0: tensor<16x8xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<16x32xf32>, %arg3: tensor<16x8xf32>, %arg4: tensor<8x32xf32>, %arg5: tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xf32>) {
  %0 = skeleton.custom_matmul ins(%arg0, %arg1 : tensor<16x8xf32>, tensor<8x32xf32>) outs(%arg2 : tensor<16x32xf32>) -> tensor<16x32xf32>
  %1 = skeleton.custom_matmul preference = #skeleton.preference<"CPU"> ins(%arg3, %arg4 : tensor<16x8xf32>, tensor<8x32xf32>) outs(%arg5 : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0, %1 : tensor<16x32xf32>, tensor<16x32xf32>
}

// CHECK-LABEL: func @test_multiple_matmuls
// CHECK: linalg.matmul
// CHECK: linalg.matmul
// CHECK-SAME: skeleton.preference = #skeleton.preference<CPU>
// CHECK-NOT: skeleton.custom_matmul
