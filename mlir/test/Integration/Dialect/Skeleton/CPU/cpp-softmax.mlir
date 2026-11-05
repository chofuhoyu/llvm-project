// REQUIRES: skeletonxx
// REQUIRES: clang-cir
//
// The first real operator, written as a flat DAG of four skeleton operators
// (rowmax, shift_exp, rowsum, normalize) in one C++ file with `main` and the
// printing. The numerical expectations are the reference softmax of each row:
//
//   [1,2,3,4]     -> 0.0321 0.0871 0.2369 0.6439 (max 4, sum 1.553002)
//   [0,0,0,0]     -> 0.2500 0.2500 0.2500 0.2500 (max 0, sum 4.000000)
//   [10,0,10,0]   -> 0.5000 0.0000 0.5000 0.0000 (max 10, sum 2.000091)
//
// Both execution paths run it: the emitted artifact and the driver process.
//
// RUN: %skeletonxx %S/cpp-softmax.cpp -o %t.prog
// RUN: %t.prog | FileCheck %s
// RUN: %skeletonxx %S/cpp-softmax.cpp --run | FileCheck %s

// CHECK: row 0: max = 4.000000 sum = 1.553002
// CHECK-NEXT: 0.0321
// CHECK-NEXT: 0.0871
// CHECK-NEXT: 0.2369
// CHECK-NEXT: 0.6439
// CHECK: row 1: max = 0.000000 sum = 4.000000
// CHECK-NEXT: 0.2500
// CHECK-NEXT: 0.2500
// CHECK-NEXT: 0.2500
// CHECK-NEXT: 0.2500
// CHECK: row 2: max = 10.000000 sum = 2.000091
// CHECK-NEXT: 0.5000
// CHECK-NEXT: 0.0000
// CHECK-NEXT: 0.5000
// CHECK-NEXT: 0.0000
