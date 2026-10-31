// RUN: mlir-skeleton-driver %s -entry main \
// RUN:   --shared-libs=%mlir_c_runner_utils,%mlir_runner_utils | FileCheck %s

// Operators written with a body region instead of a pure_fn must survive the
// runtime path: outline, per-operator recompilation, bufferization, and CPU
// lowering. The map adds the input to a zero init; the reduce sums it.

func.func private @printMemrefF32(tensor<*xf32>)

func.func @main() {
  %input = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %zeros = arith.constant dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf32>
  %zero = arith.constant dense<0.0> : tensor<f32>

  %mapped = skeleton.map ins(%input : tensor<4xf32>) outs(%zeros : tensor<4xf32>)
      (%elem: f32, %init: f32) {
    %sum = arith.addf %elem, %init : f32
    skeleton.yield %sum : f32
  }

  %reduced = skeleton.reduce ins(%mapped : tensor<4xf32>) outs(%zero : tensor<f32>)
      (%acc: f32, %elem: f32) {
    %sum = arith.addf %acc, %elem : f32
    skeleton.yield %sum : f32
  }

  %unranked = tensor.cast %reduced : tensor<f32> to tensor<*xf32>
  call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()

  // CHECK: Unranked Memref
  // CHECK-SAME: rank = 0
  // CHECK: [10]

  return
}
