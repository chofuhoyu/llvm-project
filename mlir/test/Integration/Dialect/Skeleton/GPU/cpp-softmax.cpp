// The first operator written as a real one: row-wise softmax, asking for the
// GPU.
//
// It is not a single fused operator but a flat DAG of four skeleton operators,
// each an independent unit with its own placement decision (operators.md §2:
// "a big operator is a big graph, not a big unit"):
//
//   rowmax    reduce max over the last dimension      [3][4] -> [3]
//   shift_exp map  exp(x - rowmax)                    [3][4] -> [3][4]
//   rowsum    reduce sum over the last dimension      [3][4] -> [3]
//   normalize map  y / rowsum                         [3][4] -> [3][4]
//
// Two shape rules make the graph expressible with two primitives:
//
// - a reduce's destination keeps a prefix of the input's dimensions, so
//   `float (&m)[3]` from `float (&x)[3][4]` means "reduce the trailing
//   dimension";
// - a map broadcasts an input of a lower rank over the leading dimensions of
//   its destination, so a reduction result feeds the next element-wise step.
//
// The subtraction of the row maximum is what keeps exp in range; that is why
// the reduction's identity (-FLT_MAX) is passed explicitly instead of relying
// on a zero the operator could not know about.
//
// Everything -- the operators, the buffers, the printing -- is C++.

#include "mlir/Dialect/Skeleton/SkeletonOps.h"

#include <cfloat>
#include <cmath>
#include <cstdio>

__attribute__((annotate("skeleton.pure")))
float max_fn(float a, float b) {
  return a > b ? a : b;
}

__attribute__((annotate("skeleton.pure")))
float add_fn(float a, float b) {
  return a + b;
}

__attribute__((annotate("skeleton.pure")))
float exp_shift(float x, float m) {
  return expf(x - m);
}

__attribute__((annotate("skeleton.pure")))
float div_fn(float x, float s) {
  return x / s;
}

extern "C" __attribute__((annotate("skeleton.region", "GPU")))
void softmax_rowmax(const float (&x)[3][4], float (&m)[3]) {
  skeleton_reduce(max_fn, m, x, -FLT_MAX);
}

extern "C" __attribute__((annotate("skeleton.region", "GPU")))
void softmax_shift_exp(const float (&x)[3][4], const float (&m)[3],
                       float (&y)[3][4]) {
  skeleton_map(exp_shift, y, x, m);
}

extern "C" __attribute__((annotate("skeleton.region", "GPU")))
void softmax_rowsum(const float (&y)[3][4], float (&s)[3]) {
  skeleton_reduce(add_fn, s, y, 0.0f);
}

extern "C" __attribute__((annotate("skeleton.region", "GPU")))
void softmax_normalize(const float (&y)[3][4], const float (&s)[3],
                       float (&out)[3][4]) {
  skeleton_map(div_fn, out, y, s);
}

int main() {
  // Three rows with very different spreads: a plain ramp, a uniform row (whose
  // softmax is 1/4 everywhere) and one where two elements dominate.
  float x[3][4] = {{1.0f, 2.0f, 3.0f, 4.0f},
                   {0.0f, 0.0f, 0.0f, 0.0f},
                   {10.0f, 0.0f, 10.0f, 0.0f}};
  float m[3] = {0.0f, 0.0f, 0.0f};
  float y[3][4] = {};
  float s[3] = {0.0f, 0.0f, 0.0f};
  float out[3][4] = {};

  softmax_rowmax(x, m);
  softmax_shift_exp(x, m, y);
  softmax_rowsum(y, s);
  softmax_normalize(y, s, out);

  for (int r = 0; r < 3; ++r) {
    printf("row %d: max = %f sum = %f\n", r, m[r], s[r]);
    for (int c = 0; c < 4; ++c)
      printf("  %.4f\n", out[r][c]);
  }
  return 0;
}
