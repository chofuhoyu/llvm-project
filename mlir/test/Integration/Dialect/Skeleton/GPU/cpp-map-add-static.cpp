// The CPU test's static-extent program with the operator asking for the GPU.
//
// The buffers here are host arrays, so the device bridge gives every operand a
// Unified Memory allocation and copies the data in and out around the call.
// That is the path for a host that owns its buffers; cpp-managed-add is the
// same program with runtime buffers, which needs neither the allocation nor
// the copies.
//
// The driver path is the one that exercises the GPU: an emitted artifact
// carries CPU implementations only, and recompiles the operator for the device
// the runtime picks when it runs the artifact.
//
// Expected c = a + b = [6, 8, 10, 12].

#include "mlir/Dialect/Skeleton/SkeletonOps.h"

#include <cstdio>

__attribute__((annotate("skeleton.pure")))
float my_add(float a, float b) {
  return a + b;
}

extern "C" __attribute__((annotate("skeleton.region", "GPU")))
void vec_add(const float (&a)[4], const float (&b)[4], float (&out)[4]) {
  skeleton_map(my_add, out, a, b);
}

int main() {
  float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
  float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  vec_add(a, b, c);

  for (int i = 0; i < 4; ++i)
    printf("%f\n", c[i]);
  return 0;
}
