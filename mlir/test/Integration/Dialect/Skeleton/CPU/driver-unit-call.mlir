// REQUIRES: mlir-skeleton-driver
// RUN: not mlir-skeleton-driver %s -entry main \
// RUN:   --shared-libs=%mlir_c_runner_utils,%mlir_runner_utils 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERROR

// An operator unit may not call another unit. The callee's body is moved into
// that unit's own module, so cloning the reference here would copy an empty
// declaration and the unit would fail to link at JIT time.

// ERROR: operator unit '__skeleton_op_0' calls operator unit '__skeleton_op_1'

func.func private @__skeleton_op_1(%a: tensor<4xf32>) -> tensor<4xf32>
    attributes {skeleton.op_id = 1 : i64} {
  return %a : tensor<4xf32>
}

func.func private @__skeleton_op_0(%a: tensor<4xf32>) -> tensor<4xf32>
    attributes {skeleton.op_id = 0 : i64} {
  %0 = func.call @__skeleton_op_1(%a) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @main() {
  %a = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %0 = func.call @__skeleton_op_0(%a) : (tensor<4xf32>) -> tensor<4xf32>
  %u = tensor.cast %0 : tensor<4xf32> to tensor<*xf32>
  call @printMemrefF32(%u) : (tensor<*xf32>) -> ()
  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
