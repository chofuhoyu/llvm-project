// Sidecar source for cpp-map-add-static.mlir: the static-shape counterpart of
// cpp-map-add.cpp. The host's parameters are array references, so each extent
// is part of the type the pass reads and the rewritten function takes
// memref<4xf32> instead of memref<?xf32>. That is also what lets this host
// reach the GPU: the bridge refuses a dynamically-shaped operand.
//
// The execution driver (@main) is hand-written MLIR in
// cpp-map-add-static.driver because a C++ main cannot yet ride the whole
// CIR -> Skeleton -> LLVM chain.

#include "mlir/Dialect/Skeleton/SkeletonOps.h"

__attribute__((annotate("skeleton.pure")))
float my_add(float a, float b) {
  return a + b;
}

extern "C" __attribute__((annotate("skeleton.region", "CPU")))
float *vec_add(float (&a)[4], float (&b)[4]) {
  return skeleton_map(my_add, a, b);
}
