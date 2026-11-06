// REQUIRES: skeletonxx
// REQUIRES: clang-cir
//
// The managed path on the device: a C++ operator whose data arguments are
// runtime buffers (skeleton_detail::buffer<float>), so the unit is marked
// managed and lowering does not put an allocation and a copy around the call.
// The kernel works on the runtime's memory directly and the host reads the
// result from the same memory afterwards.
//
// The length is a run-time value, so this also covers a one-dimensional
// dynamic shape on the device.
//
// RUN: %skeletonxx %S/cpp-managed-add.cpp --run \
// RUN:   --shared-lib=%mlir_cuda_runtime --shared-lib=%mlir_runner_utils \
// RUN:   --shared-lib=%mlir_c_runner_utils | FileCheck %s
// The managed unit has to be compiled for the device and placed there; a CPU
// fallback would produce the same numbers, so the trace is what distinguishes
// them.
// RUN: %skeletonxx %S/cpp-managed-add.cpp --run --driver-arg=--trace \
// RUN:   --shared-lib=%mlir_cuda_runtime --shared-lib=%mlir_runner_utils \
// RUN:   --shared-lib=%mlir_c_runner_utils 2>&1 | FileCheck %s --check-prefix=TRACE

// CHECK: managed add: ok (262144 elements)
// a[0] + b[0] = 0 + 0
// CHECK-NEXT: 0.0
// a[1] + b[1] = 1 + 1
// CHECK-NEXT: 2.0
// a[2] + b[2] = 2 + 2
// CHECK-NEXT: 4.0
// i = 262143: 262143 % 97 = 49, 262143 % 89 = 38
// CHECK-NEXT: 87.0

// TRACE: [skeleton-rt] op=0 compile device=GPU
// TRACE: [skeleton-rt] op=0 pref=GPU probe=GPU free={{.*}}choose=GPU
