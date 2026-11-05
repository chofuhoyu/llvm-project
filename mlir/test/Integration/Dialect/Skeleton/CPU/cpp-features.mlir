// REQUIRES: skeletonxx
// REQUIRES: clang-cir
//
// The C++ feature matrix the front end has to keep working: `main`, control
// flow, `std::vector`, `std::string`, `printf`, a global constant, a template,
// recursion and a function pointer, around one skeleton operator.
//
// RUN: %skeletonxx %S/cpp-features.cpp -o %t.prog
// RUN: %t.prog | FileCheck %s
// RUN: %skeletonxx %S/cpp-features.cpp --run | FileCheck %s

// CHECK: c[0] = 11.000000
// CHECK-NEXT: c[1] = 22.000000
// CHECK-NEXT: c[2] = 33.000000
// CHECK-NEXT: c[3] = 44.000000
// CHECK-NEXT: string = features
// CHECK-NEXT: scale = 3
// CHECK-NEXT: twice(scale) = 6
// CHECK-NEXT: fib(10) = 55
// CHECK-NEXT: dynamic = 4.000000
// CHECK-NEXT: twice add = 84.000000
// CHECK-NEXT: branches = 5
