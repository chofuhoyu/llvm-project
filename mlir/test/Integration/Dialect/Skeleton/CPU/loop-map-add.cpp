// Sidecar source for loop-map-add.mlir: a single annotated element-wise loop
// (no skeleton header or helper calls) that the semi-automatic path rewrites
// into a skeleton.map. The execution driver (@main) is hand-written MLIR in
// loop-map-add.driver because a C++ main cannot yet ride the whole
// CIR -> Skeleton -> LLVM chain.

extern "C" void annotated_for(int N, float *A, float *B, float *C) {
  __attribute__((annotate("skeleton.region", "CPU")))
  for (int i = 0; i < N; ++i)
    C[i] = A[i] + B[i];
}
