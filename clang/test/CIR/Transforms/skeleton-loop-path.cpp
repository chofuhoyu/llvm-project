// Real-C++ variant of cir-loop-to-skeleton.cir: an annotated element-wise for
// loop compiles through clang -emit-cir and is rewritten by cir-loop-to-
// skeleton into a pure func.func (the extracted body, in arith) and a host
// func.func carrying a skeleton.map.
//
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: cir-opt %t.cir -cir-loop-to-skeleton -o - | FileCheck %s

// The extracted element-wise add becomes a private pure function.
// CHECK-LABEL: func.func private @__skeleton_pure_0
// CHECK: arith.addf

// The host wrapper: input arrays become memrefs, the result tensor replaces
// the output array parameter (the "N" bound and the C output pointer drop).
// CHECK-LABEL: func.func @annotated_for
// CHECK-SAME: ([[A:.*]]: memref<?xf32>, [[B:.*]]: memref<?xf32>) -> tensor<?xf32>
// CHECK: bufferization.to_tensor
// CHECK: tensor.dim
// CHECK: tensor.empty
// CHECK: skeleton.map
// CHECK-SAME: preference = <"CPU"> pure_fn = @__skeleton_pure_0

// Return-value style: no destination-passing.
// CHECK-NOT: bufferization.materialize_in_destination

extern "C" void annotated_for(int N, float *A, float *B, float *C) {
  __attribute__((annotate("skeleton.region", "CPU")))
  for (int i = 0; i < N; ++i)
    C[i] = A[i] + B[i];
}
