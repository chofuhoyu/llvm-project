// REQUIRES: mlir-skeleton-driver
// RUN: not mlir-skeleton-driver %s -entry main \
// RUN:   --shared-libs=%mlir_c_runner_utils,%mlir_runner_utils 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERROR

// A skeleton operator nested in another operator's body region is not
// supported. skeleton-outline keeps the graph in one unit, so the unit fails
// to lower with the nested-operator diagnostic rather than referencing a
// function whose body was moved away.

// ERROR: nested skeleton operators in a body region are not supported
// ERROR: [skeleton-driver] op=0: cannot lower the unit for the CPU
// ERROR: [skeleton-rt] error: no implementation for op=0

func.func @main() {
  %a = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %b = arith.constant dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf32>
  %0 = skeleton.map ins(%a : tensor<4xf32>) outs(%b : tensor<4xf32>)
      (%e: f32, %i: f32) {
    %inner = skeleton.map ins(%a : tensor<4xf32>) outs(%b : tensor<4xf32>)
        (%e2: f32, %i2: f32) {
      skeleton.yield %e2 : f32
    }
    skeleton.yield %i : f32
  }
  %u = tensor.cast %0 : tensor<4xf32> to tensor<*xf32>
  call @printMemrefF32(%u) : (tensor<*xf32>) -> ()
  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
