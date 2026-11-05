// The whole program in C++: the operator, its host wrapper and `main`.
//
// `skeleton++` lowers it in one command -- clang -fclangir -emit-cir, then
// cir-opt --cir-skeleton-split --cir-to-llvm --cir-strip-cir-attrs, then the
// skeleton driver -- and both execution paths run the same program: the emitted
// artifact (its own process, CPU implementations linked in) and the driver
// (JIT, operators compiled on demand).
//
// The host takes its data through skeleton_detail::span, so the operator's
// extent is a run-time value: vec_add's unit takes memref<?xf32>. Expected
// c = a + b = [6, 8, 10, 12].

#include "mlir/Dialect/Skeleton/SkeletonOps.h"

#include <cstdio>

__attribute__((annotate("skeleton.pure")))
float my_add(float a, float b) {
  return a + b;
}

extern "C" __attribute__((annotate("skeleton.region", "CPU")))
void vec_add(const skeleton_detail::span<const float> &a,
             const skeleton_detail::span<const float> &b,
             const skeleton_detail::span<float> &out) {
  skeleton_map(my_add, out, a, b);
}

int main() {
  float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
  float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  skeleton_detail::span<const float> sa{a, 4};
  skeleton_detail::span<const float> sb{b, 4};
  skeleton_detail::span<float> sc{c, 4};
  vec_add(sa, sb, sc);

  for (int i = 0; i < 4; ++i)
    printf("%f\n", c[i]);
  return 0;
}
