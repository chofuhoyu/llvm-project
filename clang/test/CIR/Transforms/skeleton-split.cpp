// cir-skeleton-split takes the operators out of a C++ program and leaves the
// host behind as an ordinary C++ function.
//
// For each host function marked `skeleton.region` the pass creates two
// functions next to it: the operator unit (a func.func on memrefs, with the
// skeleton operator inside) and the ABI thunk (a declaration that
// skeleton-lower-host-abi later fills in). The host keeps its symbol and its C++
// signature, and its body becomes a call to the thunk.
//
// The shapes come from the host's parameter types: an array reference carries a
// compile-time extent, a span reference carries a run-time length, and a `T &`
// destination is a single element.
//
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -I %S/../../../../mlir/include %s -o %t.cir
// RUN: cir-opt %t.cir -cir-skeleton-split -o - | FileCheck %s

#include "mlir/Dialect/Skeleton/SkeletonOps.h"

__attribute__((annotate("skeleton.pure")))
float my_add(float a, float b) {
  return a + b;
}

__attribute__((annotate("skeleton.pure")))
float my_max(float a, float b) {
  return a > b ? a : b;
}

extern "C" __attribute__((annotate("skeleton.region", "CPU")))
void vec_add(const skeleton_detail::span<const float> &a,
             const skeleton_detail::span<const float> &b,
             const skeleton_detail::span<float> &out) {
  skeleton_map(my_add, out, a, b);
}

extern "C" __attribute__((annotate("skeleton.region", "CPU")))
void vec_add_static(const float (&a)[4], const float (&b)[4],
                    float (&out)[4]) {
  skeleton_map(my_add, out, a, b);
}

extern "C" __attribute__((annotate("skeleton.region", "CPU")))
void rowmax(const float (&x)[3][4], float (&out)[3]) {
  skeleton_reduce(my_max, out, x, -3.4e38f);
}

int main() {
  float a[4] = {1, 2, 3, 4};
  float b[4] = {5, 6, 7, 8};
  float c[4] = {0, 0, 0, 0};
  vec_add_static(a, b, c);
  return c[0] > 0 ? 0 : 1;
}

// The pure functions become func.func with an arith body; the max one carries a
// comparison and a select, which is what makes `a > b ? a : b` translatable.
//
// CHECK-LABEL: func.func @_Z6my_addff({{.*}}: f32, {{.*}}: f32) -> f32
// CHECK: arith.addf
// CHECK-LABEL: func.func @_Z6my_maxff({{.*}}: f32, {{.*}}: f32) -> f32
// CHECK: arith.cmpf ogt
// CHECK: arith.select

// The map unit: the destination is the first data argument, so it is the last
// memref in the signature (data arguments first, then scalars). There is no
// return value -- the caller owns the buffer.
//
// CHECK-LABEL: func.func private @__skeleton_op_0({{.*}}: memref<?xf32>, {{.*}}: memref<?xf32>, {{.*}}: memref<?xf32>)
// CHECK: bufferization.to_tensor {{.*}} restrict : memref<?xf32>
// CHECK: skeleton.map
// CHECK-SAME: pure_fn = @_Z6my_addff
// CHECK-SAME: outs({{.*}} : tensor<?xf32>)
// CHECK: bufferization.materialize_in_destination {{.*}} in restrict writable {{.*}} : (tensor<?xf32>, memref<?xf32>) -> ()
// CHECK: return

// Its thunk takes one pointer per data argument and one length per dynamic
// dimension, and describes the unit it belongs to.
//
// CHECK: cir.func private @__skeleton_op_0_shim({{.*}}: !cir.ptr<!cir.void>, {{.*}}: !u64i, {{.*}}: !cir.ptr<!cir.void>, {{.*}}: !u64i, {{.*}}: !cir.ptr<!cir.void>, {{.*}}: !u64i) attributes {skeleton.host_abi = {op_id = 0 : i64, operands = [memref<?xf32>, memref<?xf32>, memref<?xf32>], unit = "__skeleton_op_0"}}

// The host keeps its symbol and its C++ signature; its body is the thunk call.
//
// CHECK-LABEL: cir.func no_inline dso_local @vec_add({{.*}}: !cir.ptr<!cir.struct<"skeleton_detail::span<const float>" {{.*}}>>, {{.*}}: !cir.ptr<!cir.struct<"skeleton_detail::span<const float>" {{.*}}>>, {{.*}}: !cir.ptr<!cir.struct<"skeleton_detail::span<float>" {{.*}}>>) {
// CHECK: cir.get_member {{.*}}[0]
// CHECK: cir.load
// CHECK: cir.get_member {{.*}}[1]
// CHECK: cir.call @__skeleton_op_0_shim
// CHECK: cir.return

// An array-reference host has compile-time extents, so its thunk takes no
// lengths at all.
//
// CHECK: cir.func private @__skeleton_op_1_shim({{.*}}: !cir.ptr<!cir.void>, {{.*}}: !cir.ptr<!cir.void>, {{.*}}: !cir.ptr<!cir.void>) attributes {skeleton.host_abi = {op_id = 1 : i64, operands = [memref<4xf32>, memref<4xf32>, memref<4xf32>], unit = "__skeleton_op_1"}}

// The reduce unit takes the input and the single-element destination, and the
// identity as a scalar. The identity is filled into the destination, which the
// reduction then accumulates in place.
//
// CHECK-LABEL: func.func private @__skeleton_op_2(
// CHECK-SAME: memref<3x4xf32>,
// CHECK-SAME: memref<3xf32>,
// CHECK-SAME: [[INIT:.*]]: f32)
// CHECK: bufferization.to_tensor {{.*}} restrict writable : memref<3xf32>
// CHECK: linalg.fill ins([[INIT]] : f32)
// CHECK: skeleton.reduce
// CHECK-SAME: pure_fn = @_Z6my_maxff
// CHECK-SAME: outs({{.*}} : tensor<3xf32>)
// CHECK: bufferization.materialize_in_destination {{.*}} in writable

// `main` is untouched: it is ordinary C++ that the host lowering handles.
//
// CHECK-LABEL: cir.func no_inline dso_local @main(
