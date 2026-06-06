// RUN: mlir-opt %s -skeleton-target-lower -split-input-file | FileCheck %s

// Test 1: CPU target function → linalg.matmul lowered to scf.for
func.func private @cpu_func(%arg0: tensor<16x8xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> attributes {skeleton.target = "CPU"} {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<16x8xf32>, tensor<8x32xf32>) outs(%arg2 : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func private @cpu_func
// CPU path should produce scf.for loops (not scf.parallel)
// CHECK: scf.for
// CHECK-NOT: scf.parallel

// -----

// Test 2: GPU target function → linalg.matmul lowered to scf.parallel
func.func private @gpu_func(%arg0: tensor<16x8xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> attributes {skeleton.target = "GPU"} {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<16x8xf32>, tensor<8x32xf32>) outs(%arg2 : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func private @gpu_func
// GPU path should produce scf.parallel (not plain scf.for)
// CHECK: scf.parallel
