// Real-C++ variant of the manual path's failure modes: a host whose data
// parameters do not all denote the same shape. The skeleton.map verifier only
// rejects a static dimension disagreeing with another static one, so on its own
// it would let a compile-time extent through against a dynamic one and the map
// would read out of bounds at run time. The pass checks the shapes while the
// module is still CIR, where the parameter types that carry them are still
// visible.
//
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -I %S/../../../../mlir/include %s -o %t.cir
// RUN: not cir-opt %t.cir -cir-call-to-skeleton 2>&1 | FileCheck %s

// Two compile-time extents that disagree.
//
// CHECK: error: skeleton call data operand 1 denotes 'memref<8xf32>' but data operand 0 denotes 'memref<4xf32>'; all data operands must have the same element type and shape
//
// A compile-time extent against an unknown one ...
//
// CHECK: error: skeleton call data operand 1 denotes 'memref<?xf32>' but data operand 0 denotes 'memref<4xf32>'; all data operands must have the same element type and shape
//
// ... and the other way round.
//
// CHECK: error: skeleton call data operand 1 denotes 'memref<4xf32>' but data operand 0 denotes 'memref<?xf32>'; all data operands must have the same element type and shape
//
// A data operand that is not an array at all.
//
// CHECK: error: skeleton call data operand 1 is host parameter 1 of type '!cir.int<s, 32>', which is not a data array
//
// The span's size member is a length, not a data array.
//
// CHECK: error: skeleton call data operand 1 is member 1 of host parameter 1 of type {{.*}}, which is not a data array

#include "mlir/Dialect/Skeleton/SkeletonOps.h"

__attribute__((annotate("skeleton.pure")))
float my_add(float a, float b) {
  return a + b;
}

extern "C" __attribute__((annotate("skeleton.region", "CPU")))
float *bad_static(float (&a)[4], float (&b)[8]) {
  return skeleton_map(my_add, a, b);
}

extern "C" __attribute__((annotate("skeleton.region", "CPU")))
float *bad_mixed(float (&a)[4], float *b) {
  return skeleton_map(my_add, a, b);
}

extern "C" __attribute__((annotate("skeleton.region", "CPU")))
float *bad_mixed_rev(float *a, float (&b)[4]) {
  return skeleton_map(my_add, a, b);
}

extern "C" __attribute__((annotate("skeleton.region", "CPU")))
float *bad_scalar(float (&a)[4], int n) {
  return skeleton_map(my_add, a, n);
}

extern "C" __attribute__((annotate("skeleton.region", "CPU")))
float *bad_span_size(const skeleton_detail::span<float> &a,
                     const skeleton_detail::span<float> &b) {
  return skeleton_map(my_add, a.data, b.size);
}
