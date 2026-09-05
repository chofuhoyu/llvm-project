// Real-C++ variant of cir-call-to-skeleton.cir: the skeleton helper functions
// come from the installed manual-path header (SkeletonOps.h) rather than from
// hand-written cir.func declarations. Pure functions here carry a body, which
// the pass must translate to arith (via CirFuncToArith).
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
// CHECK: skeleton.map
// CHECK-SAME: pure_fn = @_Z6my_addff
//
// ... and the reduce host wrapper.
//
// CHECK-LABEL: func.func @sum_all
// CHECK-SAME: ([[A:.*]]: memref<?xf32>) -> tensor<f32>
// CHECK: bufferization.to_tensor
// CHECK: skeleton.reduce
// CHECK-SAME: pure_fn = @_Z6my_sumff
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
