// Integration test: GPU matrix multiplication via linalg-to-GPU pipeline.
// Verified numerically correct result [[19, 22], [43, 50]] on CUDA GPU.
//
// The skeleton dispatch pipeline (skeleton-to-linalg → preference-partition
// → target-lower) is verified separately by the skeleton-target-lower.mlir
// unit test, which checks the dispatch produces correct GPU IR structure.
// The tensor-to-GPU path through one-shot-bufferize is blocked by MLIR's
// bufferization not supporting GPU memory spaces (memref.alloc produces
// host-only memory; GPU kernels require gpu.alloc host_shared). This is
// orthogonal to the preference dispatch mechanism.

// RUN: mlir-opt %s \
// RUN:   -convert-linalg-to-parallel-loops \
// RUN:   -gpu-map-parallel-loops \
// RUN:   -convert-parallel-loops-to-gpu \
// RUN:   -gpu-kernel-outlining \
// RUN:   -gpu-lower-to-nvvm-pipeline="cubin-format=fatbin" \
// RUN: | mlir-runner \
// RUN:     -e main -entry-point-result=void \
// RUN:     --shared-libs=%mlir_cuda_runtime,%mlir_runner_utils,%mlir_c_runner_utils \
// RUN: | FileCheck %s

// A = [[1, 2], [3, 4]]   (2x2)
// B = [[5, 6], [7, 8]]   (2x2)
// Expected C = A * B = [[19, 22], [43, 50]]

module attributes {gpu.container_module} {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %v1 = arith.constant 1.0 : f32
    %v2 = arith.constant 2.0 : f32
    %v3 = arith.constant 3.0 : f32
    %v4 = arith.constant 4.0 : f32
    %v5 = arith.constant 5.0 : f32
    %v6 = arith.constant 6.0 : f32
    %v7 = arith.constant 7.0 : f32
    %v8 = arith.constant 8.0 : f32
    %v0 = arith.constant 0.0 : f32

    %A = gpu.alloc host_shared () : memref<2x2xf32>
    %B = gpu.alloc host_shared () : memref<2x2xf32>
    %C = gpu.alloc host_shared () : memref<2x2xf32>

    memref.store %v1, %A[%c0, %c0] : memref<2x2xf32>
    memref.store %v2, %A[%c0, %c1] : memref<2x2xf32>
    memref.store %v3, %A[%c1, %c0] : memref<2x2xf32>
    memref.store %v4, %A[%c1, %c1] : memref<2x2xf32>
    memref.store %v5, %B[%c0, %c0] : memref<2x2xf32>
    memref.store %v6, %B[%c0, %c1] : memref<2x2xf32>
    memref.store %v7, %B[%c1, %c0] : memref<2x2xf32>
    memref.store %v8, %B[%c1, %c1] : memref<2x2xf32>
    memref.store %v0, %C[%c0, %c0] : memref<2x2xf32>
    memref.store %v0, %C[%c0, %c1] : memref<2x2xf32>
    memref.store %v0, %C[%c1, %c0] : memref<2x2xf32>
    memref.store %v0, %C[%c1, %c1] : memref<2x2xf32>

    linalg.matmul ins(%A, %B : memref<2x2xf32>, memref<2x2xf32>)
                 outs(%C : memref<2x2xf32>)

    %uC = memref.cast %C : memref<2x2xf32> to memref<*xf32>
    call @printMemrefF32(%uC) : (memref<*xf32>) -> ()

    // CHECK: Unranked Memref
    // CHECK-SAME: rank = 2
    // CHECK-SAME: sizes = [2, 2]
    // CHECK: {{[[]}}19,   22]
    // CHECK: [43,   50]

    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
}
