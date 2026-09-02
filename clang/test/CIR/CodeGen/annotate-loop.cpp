// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Test: __attribute__((annotate(...))) on for/while/do-while loops
// produces cir.annotation attributes on CIR loop operations.

// Test: annotated for loop with args.
void annotated_for(int N, float *A, float *B, float *C) {
  __attribute__((annotate("skeleton.vector.add", "CPU")))
  for (int i = 0; i < N; ++i)
    C[i] = A[i] + B[i];
}

// CHECK-DAG: [#cir.annotation<"skeleton.vector.add", ["CPU"]>]

// Test: annotated while loop.
void annotated_while(int *x) {
  __attribute__((annotate("test_while")))
  while (*x < 10) {
    ++(*x);
  }
}

// CHECK-DAG: [#cir.annotation<"test_while">]

// Test: annotated do-while loop.
void annotated_do_while(int *x) {
  __attribute__((annotate("test_do")))
  do {
    ++(*x);
  } while (*x < 10);
}

// CHECK-DAG: [#cir.annotation<"test_do">]

// Test: unannotated for loop does NOT carry annotations.
void unannotated_for() {
  for (int i = 0; i < 10; ++i) {}
}

// CHECK: cir.for
// CHECK-NOT: [#cir.annotation
