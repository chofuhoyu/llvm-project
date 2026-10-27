// RUN: mlir-opt %s -skeleton-outline | FileCheck %s

// Both operator units become private functions that keep the operator in its
// skeleton form, carry the placement metadata, and are called from the host.
// Ids follow module walk order: the matmul is 0, the map is 1.

// CHECK-LABEL: func.func private @__skeleton_op_0(
// CHECK-SAME: -> tensor<2x2xf32> attributes {
// CHECK-SAME: skeleton.bytes = {{[0-9]+}} : i64
// CHECK-SAME: skeleton.op_id = 0 : i64
// CHECK-SAME: skeleton.preference = #skeleton.preference<"GPU">
// CHECK: linalg.matmul
// CHECK: return
// CHECK-LABEL: func.func private @__skeleton_op_1(
// CHECK-SAME: -> tensor<2x2xf32> attributes {
// CHECK-SAME: skeleton.op_id = 1 : i64
// CHECK-SAME: skeleton.preference = #skeleton.preference<"CPU">
// CHECK: skeleton.map
// CHECK: return
// CHECK-LABEL: func.func @host
// CHECK: func.call @__skeleton_op_0
// CHECK: func.call @__skeleton_op_1

func.func @my_add(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @host(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: tensor<2x2xf32>,
                %d: tensor<2x2xf32>, %e: tensor<2x2xf32>, %f: tensor<2x2xf32>)
    -> tensor<2x2xf32> {
  %0 = linalg.matmul {skeleton.preference = #skeleton.preference<"GPU">}
      ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%c : tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = skeleton.map preference = #skeleton.preference<"CPU"> pure_fn = @my_add
      ins(%d, %e : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%f : tensor<2x2xf32>)
  return %1 : tensor<2x2xf32>
}
