// Sidecar source for cpp-vector-add-e2e.mlir: defines the operators only.
// The execution driver (@main) is hand-written MLIR in cpp-vector-add-e2e.driver
// because a C++ main cannot yet ride the whole CIR -> Skeleton -> LLVM chain.

#include "mlir/Dialect/Skeleton/SkeletonOps.h"

__attribute__((annotate("skeleton.pure")))
float my_add(float a, float b) {
  return a + b;
}

extern "C" __attribute__((annotate("skeleton.region", "CPU")))
float *vec_add(float *a, float *b) {
  return skeleton_map(my_add, a, b);
}
