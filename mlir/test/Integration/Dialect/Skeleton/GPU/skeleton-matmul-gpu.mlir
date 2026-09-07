// RUN: mlir-opt %s \
// RUN:   -skeleton-preference-partition \
// RUN:   -skeleton-target-lower \
// RUN:   -skeleton-finalize-memref-to-gpu \
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

func.func @main() {
  %A = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %B = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
  %C = arith.constant dense<0.0> : tensor<2x2xf32>

  %result = linalg.matmul {skeleton.preference = #skeleton.preference<"GPU">}
      ins(%A, %B : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%C : tensor<2x2xf32>) -> tensor<2x2xf32>

  %unranked = tensor.cast %result : tensor<2x2xf32> to tensor<*xf32>
  call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()

  // CHECK: Unranked Memref
  // CHECK-SAME: rank = 2
  // CHECK-SAME: sizes = [2, 2]
  // CHECK: {{[[]}}19,   22]
  // CHECK: [43,   50]

  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
