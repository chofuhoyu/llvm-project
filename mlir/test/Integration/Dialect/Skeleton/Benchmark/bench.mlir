// REQUIRES: skeletonxx
// REQUIRES: clang-cir
//
// The benchmark through both devices and both execution paths. It is a lit test
// because the numbers are only meaningful if the results are right: every
// measured run compares its output against a plain C++ reference first, and a
// mismatch fails the test instead of printing a timing for garbage.
//
//   * driver, GPU available (the placement decides; on a machine with a device
//     the operator units are compiled for it)
//   * driver, GPU forced unavailable (the same binaries, placed on the CPU)
//   * artifact (CPU only by design: an artifact carries no device code)
//
// The timings themselves are recorded in caelum/doc/benchmark.md; only their
// shape is checked here, so the test does not depend on the machine.

// RUN: %skeletonxx %S/bench.cpp --run \
// RUN:   --shared-lib=%mlir_cuda_runtime --shared-lib=%mlir_runner_utils \
// RUN:   --shared-lib=%mlir_c_runner_utils | FileCheck %s --check-prefix=GPU
// RUN: env SKELETON_BENCH_DEVICE=cpu %skeletonxx %S/bench.cpp --run \
// RUN:   --shared-lib=%mlir_cuda_runtime --shared-lib=%mlir_runner_utils \
// RUN:   --shared-lib=%mlir_c_runner_utils | FileCheck %s --check-prefix=CPU
// RUN: %skeletonxx %S/bench.cpp -o %t.prog && %t.prog | FileCheck %s --check-prefix=ART
// The numbers say nothing about where they were computed: a unit that cannot be
// compiled for the device falls back to the CPU and still matches the reference.
// This run asks the trace where the operators went.
// RUN: %skeletonxx %S/bench.cpp --run --driver-arg=--trace \
// RUN:   --shared-lib=%mlir_cuda_runtime --shared-lib=%mlir_runner_utils \
// RUN:   --shared-lib=%mlir_c_runner_utils 2>&1 | FileCheck %s --check-prefix=TRACE

// GPU: softmax: ok
// GPU: softmax: inputs unchanged
// GPU: layernorm: ok
// GPU: layernorm: inputs unchanged
// GPU: matmul: ok
// GPU: map-add: ok
// GPU: timing softmax {{[0-9.]+}} ms/iteration
// GPU: timing layernorm {{[0-9.]+}} ms/iteration
// GPU: timing matmul {{[0-9.]+}} ms/iteration
// GPU: timing map-add {{[0-9.]+}} ms/iteration

// CPU: softmax: ok
// CPU: softmax: inputs unchanged
// CPU: layernorm: ok
// CPU: layernorm: inputs unchanged
// CPU: matmul: ok
// CPU: map-add: ok
// CPU: timing softmax {{[0-9.]+}} ms/iteration
// CPU: timing layernorm {{[0-9.]+}} ms/iteration
// CPU: timing matmul {{[0-9.]+}} ms/iteration
// CPU: timing map-add {{[0-9.]+}} ms/iteration

// ART: softmax: ok
// ART: softmax: inputs unchanged
// ART: layernorm: ok
// ART: layernorm: inputs unchanged
// ART: matmul: ok
// ART: map-add: ok
// ART: timing softmax {{[0-9.]+}} ms/iteration
// ART: timing layernorm {{[0-9.]+}} ms/iteration
// ART: timing matmul {{[0-9.]+}} ms/iteration
// ART: timing map-add {{[0-9.]+}} ms/iteration

// TRACE: [skeleton-rt] op=0 compile device=GPU
// TRACE: [skeleton-rt] op=0 pref=GPU probe=GPU free={{.*}}choose=GPU exec=0
// TRACE: [skeleton-rt] op=11 compile device=GPU
