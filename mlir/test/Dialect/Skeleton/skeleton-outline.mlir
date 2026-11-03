// RUN: mlir-opt %s -skeleton-outline | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// Both operator units become private functions that keep the operator in its
// skeleton form, carry the placement metadata, and are called from the host.
// Ids follow module walk order: the matmul is 0, the map is 1, the two-result
// generic is 2.

// CHECK-LABEL: func.func @host
// CHECK: call @__skeleton_op_0
// CHECK: call @__skeleton_op_1
// CHECK-LABEL: func.func @host_multi
// CHECK: call @__skeleton_op_2
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
// CHECK-LABEL: func.func private @__skeleton_op_2(
// CHECK-SAME: -> (tensor<2x2xf32>, tensor<2x2xf32>) attributes {
// CHECK-SAME: skeleton.op_id = 2 : i64
// CHECK-SAME: skeleton.preference = #skeleton.preference<"CPU">
// CHECK: linalg.generic
// CHECK: return

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

// A unit that prefers the CPU may have several results: it is never compiled
// for the GPU, whose bridge gives an operator one output memref.
func.func @host_multi(%a: tensor<2x2xf32>, %c: tensor<2x2xf32>,
                      %d: tensor<2x2xf32>)
    -> (tensor<2x2xf32>, tensor<2x2xf32>) {
  %r:2 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"],
      skeleton.preference = #skeleton.preference<"CPU">}
    ins(%a : tensor<2x2xf32>)
    outs(%c, %d : tensor<2x2xf32>, tensor<2x2xf32>) {
    ^bb0(%x: f32, %o1: f32, %o2: f32):
      %y = arith.addf %x, %x : f32
      linalg.yield %y, %x : f32, f32
  } -> (tensor<2x2xf32>, tensor<2x2xf32>)
  return %r#0, %r#1 : tensor<2x2xf32>, tensor<2x2xf32>
}
