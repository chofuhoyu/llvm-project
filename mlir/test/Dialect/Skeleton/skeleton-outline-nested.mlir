// RUN: mlir-opt %s -skeleton-outline | FileCheck %s

// An operator nested inside another unit's body region stays in that unit. It
// is not outlined separately: that would leave a unit nothing calls, while the
// outer unit would reference a function whose body was moved away.
// skeleton-to-linalg rejects the nested graph with its own diagnostic.

// CHECK-LABEL: func.func @host
// CHECK: call @__skeleton_op_0
// CHECK-LABEL: func.func private @__skeleton_op_0(
// CHECK: skeleton.map
// CHECK: skeleton.map
// CHECK: return
// CHECK-NOT: __skeleton_op_1

func.func @host(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %0 = skeleton.map ins(%a : tensor<4xf32>) outs(%b : tensor<4xf32>)
      (%e: f32, %i: f32) {
    %inner = skeleton.map ins(%a : tensor<4xf32>) outs(%b : tensor<4xf32>)
        (%e2: f32, %i2: f32) {
      skeleton.yield %e2 : f32
    }
    skeleton.yield %i : f32
  }
  return %0 : tensor<4xf32>
}
