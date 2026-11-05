// The failure modes of cir-skeleton-split. Each of these would otherwise be
// silently mis-compiled -- a dropped statement, an operator reading a buffer
// with the wrong extent, a caller left dangling -- so the pass reports them
// while the CIR still has the source types that explain what went wrong.
//
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -I %S/../../../../mlir/include %s -o %t.cir
// RUN: not cir-opt %t.cir -cir-skeleton-split 2>&1 | FileCheck %s

#include "mlir/Dialect/Skeleton/SkeletonOps.h"

#include <cstdio>

__attribute__((annotate("skeleton.pure")))
float my_add(float a, float b) {
  return a + b;
}

float plain_add(float a, float b) {
  return a + b;
}

// A helper that was not marked as a pure function cannot be applied element-wise
// by an operator unit: it would have to be compiled into the unit's body.
// CHECK: error: the first argument of a skeleton operator call must be a pure function reference
__attribute__((annotate("skeleton.region", "CPU")))
void bad_pure_fn(float (&out)[4], float (&a)[4]) {
  skeleton_map(plain_add, out, a, a);
}

// A bare pointer says nothing about how many elements follow it. The operator
// would read whatever the caller happened to have; a span is the form that
// carries a length.
// CHECK: error: skeleton call argument 1 is host parameter 0 of type '!cir.ptr<!cir.float>', which is not a data argument
__attribute__((annotate("skeleton.region", "CPU")))
void bad_pointer(const float *a, float (&out)[4]) {
  skeleton_map(my_add, out, a, a);
}

// A data argument whose element type disagrees with the destination's would be
// reinterpreted silently.
// CHECK: error: skeleton call argument 2 denotes 'memref<4xi32>' but the destination denotes 'memref<4xf32>'; all data arguments must have the same element type
__attribute__((annotate("skeleton.region", "CPU")))
void bad_types(const float (&a)[4], const int (&b)[4], float (&out)[4]) {
  skeleton_map(my_add, out, a, b);
}

// Two operators in one host: each wants its own unit and its own placement, and
// rewriting the body for both would need intermediate buffers the pass does not
// allocate.
// CHECK: error: a skeleton region host must contain exactly one skeleton operator call, found 2
__attribute__((annotate("skeleton.region", "CPU")))
void two_calls(float (&a)[4], float (&out)[4], float (&out2)[4]) {
  skeleton_map(my_add, out, a, a);
  skeleton_map(my_add, out2, a, a);
}

// Anything besides the operator call would be dropped when the body is
// replaced, so a host that does more than forward its parameters is rejected.
// CHECK: error: a skeleton region host must be a thin wrapper around its one skeleton operator call
__attribute__((annotate("skeleton.region", "CPU")))
void not_thin(float (&a)[4], float (&out)[4]) {
  printf("not a skeleton statement\n");
  skeleton_map(my_add, out, a, a);
}

// The boundary is destination-passing: the caller owns the result buffer, so
// there is no value to return.
// CHECK: error: a skeleton region host must return void
__attribute__((annotate("skeleton.region", "CPU")))
float *bad_return(float (&a)[4], float (&out)[4]) {
  skeleton_map(my_add, out, a, a);
  return a;
}

// An operator call outside an annotated host would leave the host compiled as
// ordinary C++ while its body expects an operator unit.
// CHECK: error: calls a skeleton operator but is not marked as a skeleton region
void not_annotated(float (&a)[4], float (&out)[4]) {
  skeleton_map(my_add, out, a, a);
}

// A pure function only exists inside the operator units that reference it, so
// a host call to it would be left dangling after the split.
// CHECK: error: host code may not call the pure function '_Z6my_addff'
void calls_pure(float &out, float a) {
  out = my_add(a, a);
}
