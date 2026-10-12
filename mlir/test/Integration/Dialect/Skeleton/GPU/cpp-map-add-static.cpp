// Sidecar source for cpp-map-add-static-gpu.mlir: the GPU-tagged counterpart
// of the CPU test's source, differing only in the preference the host asks
// for. It is the first host that reaches the GPU from C++: an array reference
// gives the pass a compile-time extent, so the rewritten function takes
// memref<4xf32>, which is what the device bridge requires of the operands it
// copies across.
//
// The execution driver (@main) is the CPU test's (../CPU/cpp-map-add-static
// .driver): the host's signature is the same on both targets, so only the
// lowering tail differs.

#include "mlir/Dialect/Skeleton/SkeletonOps.h"

__attribute__((annotate("skeleton.pure")))
float my_add(float a, float b) {
  return a + b;
}

extern "C" __attribute__((annotate("skeleton.region", "GPU")))
float *vec_add(float (&a)[4], float (&b)[4]) {
  return skeleton_map(my_add, a, b);
}
