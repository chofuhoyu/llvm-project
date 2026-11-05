// REQUIRES: skeletonxx
// REQUIRES: clang-cir
//
// End-to-end run of a reduction written entirely in C++: the destination is a
// single element (`float &`), which is what makes the unit's result a rank-0
// tensor and reduces over every input dimension.
//
// RUN: %skeletonxx %S/cpp-reduce-sum.cpp -o %t.prog
// RUN: %t.prog | FileCheck %s
// RUN: %skeletonxx %S/cpp-reduce-sum.cpp --run | FileCheck %s

// CHECK: 21
