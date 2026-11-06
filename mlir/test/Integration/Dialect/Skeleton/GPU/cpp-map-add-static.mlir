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
// The numbers alone do not say where they came from: a unit that fails to
// compile for the device falls back to the CPU and still prints the right
// answer. The trace does say it.
// RUN: %skeletonxx %S/cpp-map-add-static.cpp --run --driver-arg=--trace \
// RUN:   --shared-lib=%mlir_cuda_runtime --shared-lib=%mlir_runner_utils \
// RUN:   --shared-lib=%mlir_c_runner_utils 2>&1 | FileCheck %s --check-prefix=TRACE

// CHECK: 6
// CHECK-NEXT: 8
// CHECK-NEXT: 10
// CHECK-NEXT: 12

// TRACE: [skeleton-rt] op=0 compile device=GPU
// TRACE: [skeleton-rt] op=0 pref=GPU probe=GPU free={{.*}}choose=GPU
