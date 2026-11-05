// REQUIRES: skeletonxx
// REQUIRES: clang-cir
//
// The GPU twin of CPU/cpp-map-add-static: the host asks for the GPU, so the
// driver compiles the operator unit for the device it placed it on and runs it
// there. The artifact path is CPU-only by design (an artifact carries no GPU
// code), so this test drives the program through the driver.
//
// RUN: %skeletonxx %S/cpp-map-add-static.cpp --run \
// RUN:   --shared-lib=%mlir_cuda_runtime --shared-lib=%mlir_runner_utils \
// RUN:   --shared-lib=%mlir_c_runner_utils | FileCheck %s

// CHECK: 6
// CHECK-NEXT: 8
// CHECK-NEXT: 10
// CHECK-NEXT: 12
