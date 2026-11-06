// REQUIRES: skeletonxx
// REQUIRES: clang-cir
//
// The first real operator on the device: the same flat DAG of four operators
// (rowmax, shift_exp, rowsum, normalize) as CPU/cpp-softmax, with every host
// asking for the GPU. Each unit is compiled for the device on its own, so a
// reduction that cannot run there shows up as this test failing rather than as
// a silently slower program.
//
// The expected values are the reference softmax of each row:
//
//   [1,2,3,4]     -> 0.0321 0.0871 0.2369 0.6439 (max 4, sum 1.553002)
//   [0,0,0,0]     -> 0.2500 0.2500 0.2500 0.2500 (max 0, sum 4.000000)
//   [10,0,10,0]   -> 0.5000 0.0000 0.5000 0.0000 (max 10, sum 2.000091)
//
// RUN: %skeletonxx %S/cpp-softmax.cpp --run \
// RUN:   --shared-lib=%mlir_cuda_runtime --shared-lib=%mlir_runner_utils \
// RUN:   --shared-lib=%mlir_c_runner_utils | FileCheck %s
// All four units have to be compiled for the device and placed there. Without
// this line the test would also pass if every unit had quietly fallen back to
// the CPU, which is exactly the failure mode this file exists to rule out.
// RUN: %skeletonxx %S/cpp-softmax.cpp --run --driver-arg=--trace \
// RUN:   --shared-lib=%mlir_cuda_runtime --shared-lib=%mlir_runner_utils \
// RUN:   --shared-lib=%mlir_c_runner_utils 2>&1 | FileCheck %s --check-prefix=TRACE

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

// TRACE: [skeleton-rt] op=0 compile device=GPU
// TRACE: [skeleton-rt] op=0 pref=GPU probe=GPU free={{.*}}choose=GPU
// TRACE: [skeleton-rt] op=1 compile device=GPU
// TRACE: [skeleton-rt] op=2 compile device=GPU
// TRACE: [skeleton-rt] op=3 compile device=GPU
