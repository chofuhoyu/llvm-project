// Sidecar source for cpp-reduce-sum-e2e.mlir: defines the operators only.
// The execution driver (@main) is hand-written MLIR in cpp-reduce-sum-e2e.driver.

#include "mlir/Dialect/Skeleton/SkeletonOps.h"

__attribute__((annotate("skeleton.pure")))
float my_sum(float a, float b) {
  return a + b;
}

extern "C" __attribute__((annotate("skeleton.region", "CPU")))
float sum_all(float *a) {
  return skeleton_reduce(my_sum, a);
}
