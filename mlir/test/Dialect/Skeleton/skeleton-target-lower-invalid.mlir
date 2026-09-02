// RUN: mlir-opt %s -skeleton-target-lower -split-input-file -verify-diagnostics

// Test 1: a dynamically-shaped memref operand to a GPU-target call fails the
// pass explicitly instead of silently passing host memory to the kernel.

func.func private @gpu_func_dynamic(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {skeleton.target = #skeleton.target<"GPU">} {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

func.func @call_dynamic(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error@+1 {{cannot bridge dynamically-shaped memref operand to GPU (static shapes required for GPU lowering)}}
  %0 = call @gpu_func_dynamic(%arg0, %arg1, %arg2) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
