// The static-extent counterpart of cpp-map-add.cpp: the host's data arguments
// are array references, so every extent is part of the type the front end reads
// and vec_add's unit takes memref<4xf32> instead of memref<?xf32>.
//
// That is also what lets the same operator reach the GPU: the device bridge
// refuses a dynamically-shaped operand (GPU/cpp-map-add-static.cpp is the same
// program with the GPU preference).
//
// Expected c = a + b = [6, 8, 10, 12].

#include "mlir/Dialect/Skeleton/SkeletonOps.h"

#include <cstdio>

__attribute__((annotate("skeleton.pure")))
float my_add(float a, float b) {
  return a + b;
}

extern "C" __attribute__((annotate("skeleton.region", "CPU")))
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
