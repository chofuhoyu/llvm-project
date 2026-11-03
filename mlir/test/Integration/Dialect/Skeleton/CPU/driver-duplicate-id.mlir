// REQUIRES: mlir-skeleton-driver
// RUN: not mlir-skeleton-driver %s -entry main \
// RUN:   --shared-libs=%mlir_c_runner_utils,%mlir_runner_utils 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERROR

// Operator ids address the runtime table, so two units may not share one.
// skeleton-outline assigns unique ids; this catches hand-written IR.

// ERROR: operator unit '__skeleton_op_9' has a duplicate skeleton.op_id (0)

func.func private @__skeleton_op_0(%a: tensor<4xf32>) -> tensor<4xf32>
    attributes {skeleton.op_id = 0 : i64} {
  return %a : tensor<4xf32>
}

func.func private @__skeleton_op_9(%a: tensor<4xf32>) -> tensor<4xf32>
    attributes {skeleton.op_id = 0 : i64} {
  return %a : tensor<4xf32>
}

func.func @main() {
  %a = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %0 = func.call @__skeleton_op_0(%a) : (tensor<4xf32>) -> tensor<4xf32>
  %u = tensor.cast %0 : tensor<4xf32> to tensor<*xf32>
  call @printMemrefF32(%u) : (tensor<*xf32>) -> ()
  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
