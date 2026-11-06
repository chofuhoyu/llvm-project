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

// CHECK: 0.0
// CHECK-NEXT: 11.0
// CHECK-NEXT: 22.0
// CHECK-NEXT: 33.0
// CHECK-NEXT: 44.0
// CHECK-NEXT: 55.0
