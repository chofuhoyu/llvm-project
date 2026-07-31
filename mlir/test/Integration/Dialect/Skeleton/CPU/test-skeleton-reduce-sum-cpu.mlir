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

// End-to-end correctness test for skeleton.reduce on CPU:
//   skeleton.reduce {pure_fn = @my_sum} → linalg.reduce → scf loops → LLVM

// Input = [1.0, 2.0, 3.0, 4.0]
// Expected sum = 1+2+3+4 = 10.0

func.func @my_sum(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @main() {
  %input = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %init = arith.constant 0.0 : f32
  %init_t = tensor.from_elements %init : tensor<f32>

  %result = skeleton.reduce preference = #skeleton.preference<"CPU"> pure_fn = @my_sum
    ins(%input : tensor<4xf32>)
    outs(%init_t : tensor<f32>) -> tensor<f32>

  %unranked = tensor.cast %result : tensor<f32> to tensor<*xf32>
  call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()

  // CHECK: Unranked Memref
  // CHECK-SAME: rank = 0
  // CHECK: [10]

  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
