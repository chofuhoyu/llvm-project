// REQUIRES: mlir-skeleton-driver
// RUN: not mlir-skeleton-driver %s -entry main \
// RUN:   --shared-libs=%mlir_c_runner_utils,%mlir_runner_utils 2>&1 | \
// RUN: FileCheck %s --check-prefix=ERROR

// An operator that cannot be compiled must terminate with a diagnostic, not
// call a null function pointer. A pure function that only has a declaration
// passes the skeleton verifier but cannot be lowered, so the unit fails to
// compile and the runtime has no CPU fallback.

// ERROR: [skeleton-rt] error: no implementation for op=
// ERROR-NOT: PLEASE submit a bug report

func.func private @my_add(%a: f32, %b: f32) -> f32

func.func @main() {
  %input = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %zeros = arith.constant dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf32>

  %mapped = skeleton.map pure_fn = @my_add
      ins(%input, %zeros : tensor<4xf32>, tensor<4xf32>)
      outs(%zeros : tensor<4xf32>)

  %unranked = tensor.cast %mapped : tensor<4xf32> to tensor<*xf32>
  call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
  return
}

func.func private @printMemrefF32(tensor<*xf32>)
