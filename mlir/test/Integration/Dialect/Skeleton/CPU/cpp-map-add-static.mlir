// REQUIRES: skeletonxx
// REQUIRES: clang-cir
//
// End-to-end run of the static-extent operator: array-reference parameters give
// the unit a memref<4xf32> signature. Same program, both execution paths.
//
// RUN: %skeletonxx %S/cpp-map-add-static.cpp -o %t.prog
// RUN: %t.prog | FileCheck %s
// RUN: %skeletonxx %S/cpp-map-add-static.cpp --run | FileCheck %s

// CHECK: 6
// CHECK-NEXT: 8
// CHECK-NEXT: 10
// CHECK-NEXT: 12
