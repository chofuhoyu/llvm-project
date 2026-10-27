// Real-C++ variant of cir-call-to-skeleton.cir: the skeleton helper functions
// come from the installed manual-path header (SkeletonOps.h) rather than from
// hand-written cir.func declarations. Pure functions here carry a body, which
// the pass must translate to arith (via CirFuncToArith).
//
// The host's parameter types are where an array's shape comes from: an array
// reference carries a compile-time extent, skeleton_detail::span carries a
// runtime one, and a bare pointer writes down no extent at all.
//
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -I %S/../../../../mlir/include %s -o %t.cir
// RUN: cir-opt %t.cir -cir-call-to-skeleton -o - | FileCheck %s
//
// The two pure functions (real definitions, so no `private`; bodies become
// arith) ...
//
// CHECK-LABEL: func.func @_Z6my_addff({{.*}}: f32, {{.*}}: f32) -> f32
// CHECK: arith.addf
// CHECK-LABEL: func.func @_Z6my_sumff({{.*}}: f32, {{.*}}: f32) -> f32
// CHECK: arith.addf
//
// ... the map host wrapper (extern "C" keeps the symbol unmangled) ...
//
// CHECK-LABEL: func.func @vec_add
// CHECK-SAME: ([[A:.*]]: memref<?xf32>, [[B:.*]]: memref<?xf32>) -> tensor<?xf32>
// CHECK: bufferization.to_tensor
// CHECK: tensor.dim
// CHECK: tensor.empty
// CHECK: {{%.*}} = skeleton.map
// CHECK-SAME: pure_fn = @_Z6my_addff
//
// ... and the reduce host wrapper.
//
// CHECK-LABEL: func.func @sum_all
// CHECK-SAME: ([[A:.*]]: memref<?xf32>) -> tensor<f32>
// CHECK: bufferization.to_tensor
// The accumulator is zero-initialized, not an uninitialized tensor.empty.
// CHECK: %[[INIT:.*]] = arith.constant dense<0.000000e+00> : tensor<f32>
// CHECK: skeleton.reduce
// CHECK-SAME: pure_fn = @_Z6my_sumff
// CHECK-SAME: outs(%[[INIT]] : tensor<f32>)
//
// The same reduction over an array reference, whose extent is in the type.
//
// CHECK-LABEL: func.func @sum_all_static
// CHECK-SAME: ([[A:.*]]: memref<4xf32>) -> tensor<f32>
// CHECK: skeleton.reduce
// CHECK-SAME: pure_fn = @_Z6my_sumff
//
// The three ways a host can spell a data parameter. An array reference gives a
// static memref and an output that inherits its shape, with no extent to look
// up at run time ...
//
// CHECK-LABEL: func.func @vec_add_static
// CHECK-SAME: ([[A:.*]]: memref<4xf32>, [[B:.*]]: memref<4xf32>) -> tensor<4xf32>
// CHECK: bufferization.to_tensor
// CHECK-NOT: tensor.dim
// CHECK-NOT: cf.assert
// CHECK: tensor.empty() : tensor<4xf32>
// CHECK: {{%.*}} = skeleton.map
// CHECK-SAME: pure_fn = @_Z6my_addff
//
// ... nesting contributes the outer extents, so a second dimension arrives as
// written ...
//
// CHECK-LABEL: func.func @vec_add_2d
// CHECK-SAME: ([[A:.*]]: memref<2x4xf32>, [[B:.*]]: memref<2x4xf32>) -> tensor<2x4xf32>
// CHECK: tensor.empty() : tensor<2x4xf32>
// CHECK: {{%.*}} = skeleton.map
//
// ... and skeleton_detail::span says the extent is a runtime value, which
// leaves the memref dynamic and its output sized from the input. Two runtime
// lengths are only known to agree at run time, so they are checked against each
// other rather than against a type ...
//
// CHECK-LABEL: func.func @vec_add_span
// CHECK-SAME: ([[A:.*]]: memref<?xf32>, [[B:.*]]: memref<?xf32>) -> tensor<?xf32>
// CHECK: tensor.dim
// CHECK: cf.assert
// CHECK: tensor.empty
// CHECK: {{%.*}} = skeleton.map
// CHECK-SAME: pure_fn = @_Z6my_addff
//
// ... and the span's data member names the same data array as the span itself.
//
// CHECK-LABEL: func.func @vec_add_span_data
// CHECK-SAME: ([[A:.*]]: memref<?xf32>, [[B:.*]]: memref<?xf32>) -> tensor<?xf32>
// CHECK: cf.assert
// CHECK: {{%.*}} = skeleton.map
// CHECK-SAME: pure_fn = @_Z6my_addff
//
// Return-value style: no destination-passing.
// CHECK-NOT: bufferization.materialize_in_destination

#include "mlir/Dialect/Skeleton/SkeletonOps.h"

__attribute__((annotate("skeleton.pure")))
float my_add(float a, float b) {
  return a + b;
}

__attribute__((annotate("skeleton.pure")))
float my_sum(float a, float b) {
  return a + b;
}

extern "C" __attribute__((annotate("skeleton.region", "CPU")))
float *vec_add(float *a, float *b) {
  return skeleton_map(my_add, a, b);
}

extern "C" __attribute__((annotate("skeleton.region", "CPU")))
float sum_all(float *a) {
  return skeleton_reduce(my_sum, a);
}

extern "C" __attribute__((annotate("skeleton.region", "CPU")))
float sum_all_static(float (&a)[4]) {
  return skeleton_reduce(my_sum, a);
}

// An array reference carries a compile-time extent: the pass reads N out of the
// parameter type and the map's output inherits that shape.
extern "C" __attribute__((annotate("skeleton.region", "CPU")))
float *vec_add_static(float (&a)[4], float (&b)[4]) {
  return skeleton_map(my_add, a, b);
}

// Nesting supplies the outer extents, so a second dimension arrives as it is
// written: rank-2 in, rank-2 out.
extern "C" __attribute__((annotate("skeleton.region", "CPU")))
float *vec_add_2d(float (&a)[2][4], float (&b)[2][4]) {
  return skeleton_map(my_add, a, b);
}

// skeleton_detail::span says the extent is a runtime value. The pass needs only
// the type to know that; the length itself travels in the memref, which the
// caller sizes.
extern "C" __attribute__((annotate("skeleton.region", "CPU")))
float *vec_add_span(const skeleton_detail::span<float> &a,
                    const skeleton_detail::span<float> &b) {
  return skeleton_map(my_add, a, b);
}

// The span's data member reached directly, rather than through the span: the
// helper's const reference binds to it, so it arrives as a member access.
extern "C" __attribute__((annotate("skeleton.region", "CPU")))
float *vec_add_span_data(const skeleton_detail::span<float> &a,
                         const skeleton_detail::span<float> &b) {
  return skeleton_map(my_add, a.data, b.data);
}
