// REQUIRES: mlir-skeleton-driver
// RUN: mlir-skeleton-driver %s -entry main \
// RUN:   --shared-libs=%mlir_cuda_runtime,%mlir_runner_utils,%mlir_c_runner_utils | FileCheck %s

// Two GPU-preference operators in one program. Each becomes its own op unit
// with its own JIT engine and cubin; both must compile and run in the same
// process without colliding.

func.func private @printMemrefF32(tensor<*xf32>)

func.func @main() {
  %A = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %B = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
  %I = arith.constant dense<[[1.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf32>
  %C = arith.constant dense<0.0> : tensor<2x2xf32>

  %first = linalg.matmul {skeleton.preference = #skeleton.preference<"GPU">}
      ins(%A, %B : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%C : tensor<2x2xf32>) -> tensor<2x2xf32>
  %second = linalg.matmul {skeleton.preference = #skeleton.preference<"GPU">}
      ins(%first, %I : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%C : tensor<2x2xf32>) -> tensor<2x2xf32>

  %unranked = tensor.cast %second : tensor<2x2xf32> to tensor<*xf32>
  call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()

  // CHECK: Unranked Memref
  // CHECK-SAME: rank = 2
  // CHECK-SAME: sizes = [2, 2]
  // CHECK: {{[[]}}19,   22]
  // CHECK: [43,   50]

  return
}
