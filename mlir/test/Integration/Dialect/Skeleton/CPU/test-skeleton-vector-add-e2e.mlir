// RUN: mlir-opt %s \
// RUN:   -skeleton-to-linalg \
// RUN:   -skeleton-preference-partition \
// RUN:   -skeleton-target-lower \
// RUN:   -convert-scf-to-cf \
// RUN:   -expand-strided-metadata \
// RUN:   -convert-arith-to-llvm \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-cf-to-llvm \
// RUN:   -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils \
// RUN: | FileCheck %s

// This is an end-to-end correctness test for the skeleton vector_add path:
//   skeleton.vector_add → skeleton-to-linalg → linalg.add
//   → skeleton-preference-partition → outlined function
//   → skeleton-target-lower → scf loops
//   → full LLVM lowering → mlir-runner

// A = [1.0, 2.0, 3.0, 4.0]  (4-element vector)
// B = [5.0, 6.0, 7.0, 8.0]  (4-element vector)
// Expected C = A + B = [6.0, 8.0, 10.0, 12.0]

func.func @main() {
  %A = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %B = arith.constant dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf32>
  %C = arith.constant dense<0.0> : tensor<4xf32>

  %result = skeleton.vector_add preference = #skeleton.preference<"CPU">
      ins(%A, %B : tensor<4xf32>, tensor<4xf32>)
      outs(%C : tensor<4xf32>)

  %unranked = tensor.cast %result : tensor<4xf32> to tensor<*xf32>
  call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()

  // CHECK: Unranked Memref
  // CHECK-SAME: rank = 1
  // CHECK-SAME: sizes = [4]
  // CHECK: [6,    8,    10,   12]

  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
