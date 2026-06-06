// RUN: mlir-opt %s -skeleton-target-lower -split-input-file | FileCheck %s --check-prefixes=CHECK,CORE
// RUN: mlir-opt %s -skeleton-target-lower -gpu-map-parallel-loops -convert-parallel-loops-to-gpu -gpu-kernel-outlining -split-input-file | FileCheck %s --check-prefixes=CHECK,GPU

// Test 1: CPU target → linalg.matmul lowered to scf.for.
// Both runs produce the same output (GPU passes skip CPU functions).
func.func private @cpu_func(%arg0: tensor<16x8xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> attributes {skeleton.target = "CPU"} {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<16x8xf32>, tensor<8x32xf32>) outs(%arg2 : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func private @cpu_func
// CORE: scf.for
// CORE-NOT: scf.parallel
// GPU: scf.for
// GPU-NOT: gpu.launch_func

// -----

// Test 2: GPU target.
// CORE run: stops at scf.parallel.
// GPU run: continues to gpu.launch_func + gpu.module + gpu.func.
func.func private @gpu_func(%arg0: tensor<16x8xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> attributes {skeleton.target = "GPU"} {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<16x8xf32>, tensor<8x32xf32>) outs(%arg2 : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func private @gpu_func
// CORE: scf.parallel
// CORE-NOT: gpu.launch_func
// GPU: gpu.launch_func
// GPU: gpu.module
// GPU: gpu.func
