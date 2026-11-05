// REQUIRES: skeletonxx
// REQUIRES: clang-cir
//
// End-to-end run of a program that is entirely C++: the operator (pure function
// my_add + host wrapper vec_add) and the execution driver (`main`, which
// allocates the buffers and prints the result) live in cpp-map-add.cpp.
//
// The artifact path runs the emitted executable on its own; the driver path
// re-reads the module and runs it in-process, compiling the operator on demand.
// Expected c = a + b = [6, 8, 10, 12].
//
// RUN: %skeletonxx %S/cpp-map-add.cpp -o %t.prog
// RUN: %t.prog | FileCheck %s
// RUN: %skeletonxx %S/cpp-map-add.cpp --run | FileCheck %s

// CHECK: 6
// CHECK-NEXT: 8
// CHECK-NEXT: 10
// CHECK-NEXT: 12
