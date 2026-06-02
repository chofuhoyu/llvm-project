// RUN: mlir-opt %s -skeleton-preference-partition -split-input-file | FileCheck %s

// Test 1: single GPU-preference matmul is outlined into a new function.
func.func @test_single_gpu(%arg0: tensor<16x8xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.matmul {skeleton.preference = #skeleton.preference<"GPU">} ins(%arg0, %arg1 : tensor<16x8xf32>, tensor<8x32xf32>) outs(%arg2 : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func @test_single_gpu
// CHECK: call @GPU_matmul_0
// CHECK-SAME: -> tensor<16x32xf32>

// CHECK-LABEL: func private @GPU_matmul_0
// CHECK-SAME: attributes {skeleton.target = "GPU"}
// CHECK: linalg.matmul
// CHECK-SAME: skeleton.preference = #skeleton.preference<GPU>
// CHECK: return

// -----

// Test 2: single CPU-preference matmul is outlined.
func.func @test_single_cpu(%arg0: tensor<16x8xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.matmul {skeleton.preference = #skeleton.preference<"CPU">} ins(%arg0, %arg1 : tensor<16x8xf32>, tensor<8x32xf32>) outs(%arg2 : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func @test_single_cpu
// CHECK: call @CPU_matmul_0
// CHECK-SAME: -> tensor<16x32xf32>

// CHECK-LABEL: func private @CPU_matmul_0
// CHECK-SAME: attributes {skeleton.target = "CPU"}
// CHECK: linalg.matmul
// CHECK-SAME: skeleton.preference = #skeleton.preference<CPU>
// CHECK: return

// -----

// Test 3: mixed CPU + GPU + no-preference matmuls.  No-preference matmul stays in place.
func.func @test_mixed(%arg0: tensor<16x8xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xf32>, tensor<16x32xf32>) {
  %0 = linalg.matmul {skeleton.preference = #skeleton.preference<"GPU">} ins(%arg0, %arg1 : tensor<16x8xf32>, tensor<8x32xf32>) outs(%arg2 : tensor<16x32xf32>) -> tensor<16x32xf32>
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<16x8xf32>, tensor<8x32xf32>) outs(%arg2 : tensor<16x32xf32>) -> tensor<16x32xf32>
  %2 = linalg.matmul {skeleton.preference = #skeleton.preference<"CPU">} ins(%arg0, %arg1 : tensor<16x8xf32>, tensor<8x32xf32>) outs(%arg2 : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0, %1, %2 : tensor<16x32xf32>, tensor<16x32xf32>, tensor<16x32xf32>
}

// CHECK-LABEL: func @test_mixed
// GPU matmul → call
// CHECK: call @GPU_matmul_0
// CHECK-SAME: -> tensor<16x32xf32>
// no-preference matmul stays
// CHECK: linalg.matmul
// CHECK-NOT: skeleton.preference
// CPU matmul → call
// CHECK: call @CPU_matmul_1
// CHECK-SAME: -> tensor<16x32xf32>

// CHECK-LABEL: func private @GPU_matmul_0
// CHECK-SAME: attributes {skeleton.target = "GPU"}
// CHECK: linalg.matmul
// CHECK: return

// CHECK-LABEL: func private @CPU_matmul_1
// CHECK-SAME: attributes {skeleton.target = "CPU"}
// CHECK: linalg.matmul
// CHECK: return
