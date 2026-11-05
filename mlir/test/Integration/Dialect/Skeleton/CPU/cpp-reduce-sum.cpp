// A reduction, whole program in C++.
//
// sum_all reduces a run-time-length span to a single float. The destination is
// `float &out`: one element, which is how the boundary says the operator's
// result is a scalar rather than an array. The identity (0 here) is passed
// explicitly, so a combiner whose identity is not zero -- max, min, product --
// can use the same helper.
//
// Expected sum(1..6) = 21.

#include "mlir/Dialect/Skeleton/SkeletonOps.h"

#include <cstdio>

__attribute__((annotate("skeleton.pure")))
float my_sum(float a, float b) {
  return a + b;
}

extern "C" __attribute__((annotate("skeleton.region", "CPU")))
void sum_all(const skeleton_detail::span<const float> &x, float &out) {
  skeleton_reduce(my_sum, out, x, 0.0f);
}

int main() {
  float x[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float total = 0.0f;

  skeleton_detail::span<const float> sx{x, 6};
  sum_all(sx, total);

  printf("%f\n", total);
  return 0;
}
