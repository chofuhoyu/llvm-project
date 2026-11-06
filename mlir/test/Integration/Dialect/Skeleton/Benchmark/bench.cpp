// The benchmark: whole programs written in C++, measured on both devices.
//
// operators.md §5 asks for a real load with fixed inputs, a reference
// implementation to compare against, and timings. This program is that load.
// It runs three shapes of work through the same operator boundary:
//
//   softmax    four operators, row-wise, on static arrays
//   layernorm  six operators, row-wise, on static arrays
//   matmul     one operator, the contraction itself
//   map-add    one operator over runtime buffers, lengths known only at run
//              time (so the buffers are the ones the runtime owns)
//
// The first two hand the operator host buffers, so device lowering gives each
// operand a Unified Memory allocation and copies the data in and out around the
// call. The third allocates its buffers from the runtime, so the device works
// on them where they are. Both paths are what a program can express today, and
// the timing table separates them.
//
// Every result is checked against a plain C++ reference before it is timed, so
// a timing is only printed for a result that is known good. `SKELETON_BENCH_DEVICE=cpu`
// forces the placement onto the CPU, which is how the two rows of the table are
// produced from one binary.
//
// The operator, the reference, the printing and the timing are all C++.

#include "mlir/Dialect/Skeleton/SkeletonOps.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace {

constexpr int kRows = 32;
constexpr int kCols = 128;
constexpr int kIterations = 32;
constexpr unsigned long kAddElements = 1u << 18;

double nowMs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) * 1.0e3 +
         static_cast<double>(ts.tv_nsec) / 1.0e6;
}

// The placement is a run-time decision, so the benchmark selects a device the
// same way a test does: by telling the runtime the GPU is unavailable.
void selectDeviceFromEnvironment() {
  const char *device = getenv("SKELETON_BENCH_DEVICE");
  if (device && strcmp(device, "cpu") == 0)
    skeleton_rt_set_device_available(SKELETON_DEVICE_GPU, 0);
}

bool closeEnough(const char *name, const float *got, const float *want,
                 int count, float tolerance) {
  float worst = 0.0f;
  for (int i = 0; i < count; ++i) {
    float diff = got[i] - want[i];
    if (diff < 0.0f)
      diff = -diff;
    if (diff > worst)
      worst = diff;
  }
  if (worst <= tolerance) {
    printf("%s: ok (max |diff| %.6g)\n", name, static_cast<double>(worst));
    return true;
  }
  printf("%s: MISMATCH (max |diff| %.6g, tolerance %.6g)\n", name,
         static_cast<double>(worst), static_cast<double>(tolerance));
  return false;
}

// Pure functions: the scalar body of an operator.

__attribute__((annotate("skeleton.pure"))) float max_fn(float a, float b) {
  return a > b ? a : b;
}

__attribute__((annotate("skeleton.pure"))) float add_fn(float a, float b) {
  return a + b;
}

__attribute__((annotate("skeleton.pure"))) float exp_shift(float x, float m) {
  return expf(x - m);
}

__attribute__((annotate("skeleton.pure"))) float div_fn(float x, float s) {
  return x / s;
}

__attribute__((annotate("skeleton.pure"))) float center_fn(float x, float sum,
                                                           float inv_n) {
  return x - sum * inv_n;
}

__attribute__((annotate("skeleton.pure"))) float square_fn(float x) {
  return x * x;
}

__attribute__((annotate("skeleton.pure"))) float stats_fn(float sumsq,
                                                          float inv_n,
                                                          float eps) {
  return 1.0f / sqrtf(sumsq * inv_n + eps);
}

__attribute__((annotate("skeleton.pure"))) float scale_fn(float x, float k) {
  return x * k;
}

// Softmax as a flat DAG of four operators: rowmax, shift_exp, rowsum,
// normalize. Each is one host function, one unit, one placement decision.

extern "C" __attribute__((annotate("skeleton.region", "GPU")))
void softmax_rowmax(const float (&x)[kRows][kCols], float (&m)[kRows]) {
  skeleton_reduce(max_fn, m, x, -3.4e38f);
}

extern "C" __attribute__((annotate("skeleton.region", "GPU")))
void softmax_shift_exp(const float (&x)[kRows][kCols],
                       const float (&m)[kRows],
                       float (&y)[kRows][kCols]) {
  skeleton_map(exp_shift, y, x, m);
}

extern "C" __attribute__((annotate("skeleton.region", "GPU")))
void softmax_rowsum(const float (&y)[kRows][kCols], float (&s)[kRows]) {
  skeleton_reduce(add_fn, s, y, 0.0f);
}

extern "C" __attribute__((annotate("skeleton.region", "GPU")))
void softmax_normalize(const float (&y)[kRows][kCols],
                       const float (&s)[kRows],
                       float (&out)[kRows][kCols]) {
  skeleton_map(div_fn, out, y, s);
}

// Layernorm as a flat DAG of six operators: rowsum, center, square, rowsum,
// stats, scale. The two scalars a row-wise reduction needs (1/n and epsilon)
// are passed as scalars and broadcast over the destination.

extern "C" __attribute__((annotate("skeleton.region", "GPU")))
void norm_rowsum(const float (&x)[kRows][kCols], float (&s)[kRows]) {
  skeleton_reduce(add_fn, s, x, 0.0f);
}

extern "C" __attribute__((annotate("skeleton.region", "GPU")))
void norm_center(const float (&x)[kRows][kCols], const float (&s)[kRows],
                 float (&c)[kRows][kCols], float inv_n) {
  skeleton_map(center_fn, c, x, s, inv_n);
}

extern "C" __attribute__((annotate("skeleton.region", "GPU")))
void norm_square(const float (&c)[kRows][kCols],
                 float (&sq)[kRows][kCols]) {
  skeleton_map(square_fn, sq, c);
}

extern "C" __attribute__((annotate("skeleton.region", "GPU")))
void norm_sumsq(const float (&sq)[kRows][kCols], float (&ss)[kRows]) {
  skeleton_reduce(add_fn, ss, sq, 0.0f);
}

extern "C" __attribute__((annotate("skeleton.region", "GPU")))
void norm_stats(const float (&ss)[kRows], float (&inv)[kRows], float inv_n,
                float eps) {
  skeleton_map(stats_fn, inv, ss, inv_n, eps);
}

extern "C" __attribute__((annotate("skeleton.region", "GPU")))
void norm_scale(const float (&c)[kRows][kCols], const float (&inv)[kRows],
                float (&out)[kRows][kCols]) {
  skeleton_map(scale_fn, out, c, inv);
}

// The contraction: the one basic operator that cannot be a pure function, so
// it names the operator instead of a scalar body.

extern "C" __attribute__((annotate("skeleton.region", "GPU")))
void matmul(const float (&a)[kRows][kCols], const float (&b)[kCols][kRows],
            float (&out)[kRows][kRows]) {
  skeleton_matmul(out, a, b);
}

// The runtime-buffer operator: the same element-wise add, but over buffers the
// runtime allocated and with lengths that only exist at run time.

extern "C" __attribute__((annotate("skeleton.region", "GPU")))
void add_buffers(const skeleton_detail::buffer<float> &a,
                 const skeleton_detail::buffer<float> &b,
                 const skeleton_detail::buffer<float> &out) {
  skeleton_map(add_fn, out, a, b);
}

void fillSoftmaxInput(float (&x)[kRows][kCols]) {
  for (int r = 0; r < kRows; ++r)
    for (int c = 0; c < kCols; ++c)
      x[r][c] = std::sin(static_cast<float>(r) * 0.7f +
                         static_cast<float>(c) * 0.11f) *
                3.0f;
}

void referenceSoftmax(const float (&x)[kRows][kCols],
                      float (&out)[kRows][kCols]) {
  for (int r = 0; r < kRows; ++r) {
    float m = x[r][0];
    for (int c = 1; c < kCols; ++c)
      if (x[r][c] > m)
        m = x[r][c];
    float s = 0.0f;
    for (int c = 0; c < kCols; ++c)
      s += expf(x[r][c] - m);
    for (int c = 0; c < kCols; ++c)
      out[r][c] = expf(x[r][c] - m) / s;
  }
}

void referenceLayernorm(const float (&x)[kRows][kCols], float (&out)[kRows][kCols],
                        float eps) {
  for (int r = 0; r < kRows; ++r) {
    float sum = 0.0f;
    for (int c = 0; c < kCols; ++c)
      sum += x[r][c];
    float mean = sum / static_cast<float>(kCols);
    float sq = 0.0f;
    for (int c = 0; c < kCols; ++c)
      sq += (x[r][c] - mean) * (x[r][c] - mean);
    float inv = 1.0f / sqrtf(sq / static_cast<float>(kCols) + eps);
    for (int c = 0; c < kCols; ++c)
      out[r][c] = (x[r][c] - mean) * inv;
  }
}

double timeSoftmax(const float (&x)[kRows][kCols],
                   float (&out)[kRows][kCols]) {
  double begin = nowMs();
  for (int i = 0; i < kIterations; ++i) {
    float m[kRows];
    float y[kRows][kCols];
    float s[kRows];
    softmax_rowmax(x, m);
    softmax_shift_exp(x, m, y);
    softmax_rowsum(y, s);
    softmax_normalize(y, s, out);
  }
  return (nowMs() - begin) / kIterations;
}

double timeLayernorm(const float (&x)[kRows][kCols],
                     float (&out)[kRows][kCols], float eps) {
  const float inv_n = 1.0f / static_cast<float>(kCols);
  double begin = nowMs();
  for (int i = 0; i < kIterations; ++i) {
    float s[kRows];
    float c[kRows][kCols];
    float sq[kRows][kCols];
    float ss[kRows];
    float inv[kRows];
    norm_rowsum(x, s);
    norm_center(x, s, c, inv_n);
    norm_square(c, sq);
    norm_sumsq(sq, ss);
    norm_stats(ss, inv, inv_n, eps);
    norm_scale(c, inv, out);
  }
  return (nowMs() - begin) / kIterations;
}

void referenceMatmul(const float (&a)[kRows][kCols],
                     const float (&b)[kCols][kRows],
                     float (&out)[kRows][kRows]) {
  for (int i = 0; i < kRows; ++i)
    for (int j = 0; j < kRows; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < kCols; ++k)
        sum += a[i][k] * b[k][j];
      out[i][j] = sum;
    }
}

double timeMatmul(const float (&a)[kRows][kCols],
                  const float (&b)[kCols][kRows],
                  float (&out)[kRows][kRows]) {
  double begin = nowMs();
  for (int i = 0; i < kIterations; ++i)
    matmul(a, b, out);
  return (nowMs() - begin) / kIterations;
}

double timeAddBuffers(skeleton_detail::buffer<float> &a,
                      skeleton_detail::buffer<float> &b,
                      skeleton_detail::buffer<float> &out) {
  double begin = nowMs();
  for (int i = 0; i < kIterations; ++i)
    add_buffers(a, b, out);
  return (nowMs() - begin) / kIterations;
}

} // namespace

int main() {
  selectDeviceFromEnvironment();

  float x[kRows][kCols], got[kRows][kCols], want[kRows][kCols];
  fillSoftmaxInput(x);

  referenceSoftmax(x, want);
  double softmaxMs = timeSoftmax(x, got);
  bool softmaxOk = closeEnough("softmax", &got[0][0], &want[0][0],
                               kRows * kCols, 1.0e-4f);

  const float eps = 1.0e-5f;
  referenceLayernorm(x, want, eps);
  double normMs = timeLayernorm(x, got, eps);
  bool normOk = closeEnough("layernorm", &got[0][0], &want[0][0],
                            kRows * kCols, 1.0e-4f);

  // A small, dense contraction: the shapes come from the parameter types, so
  // the device sees static extents here.
  float lhs[kRows][kCols], rhs[kCols][kRows], product[kRows][kRows];
  float productWant[kRows][kRows];
  for (int r = 0; r < kRows; ++r)
    for (int c = 0; c < kCols; ++c)
      lhs[r][c] = static_cast<float>((r * 31 + c * 17) % 23) * 0.125f - 1.0f;
  for (int r = 0; r < kCols; ++r)
    for (int c = 0; c < kRows; ++c)
      rhs[r][c] = static_cast<float>((r * 13 + c * 7) % 19) * 0.0625f - 0.5f;
  referenceMatmul(lhs, rhs, productWant);
  double matmulMs = timeMatmul(lhs, rhs, product);
  bool matmulOk = closeEnough("matmul", &product[0][0], &productWant[0][0],
                              kRows * kRows, 1.0e-2f);

  const unsigned long n = kAddElements;
  skeleton_detail::buffer<float> a(n), b(n), out(n);
  for (unsigned long i = 0; i < n; ++i) {
    a.data[i] = static_cast<float>(i % 97);
    b.data[i] = static_cast<float>(i % 89);
    out.data[i] = 0.0f;
  }
  double addMs = timeAddBuffers(a, b, out);
  bool addOk = true;
  float worst = 0.0f;
  for (unsigned long i = 0; i < n; ++i) {
    float diff = out.data[i] - (a.data[i] + b.data[i]);
    if (diff < 0.0f)
      diff = -diff;
    if (diff > worst)
      worst = diff;
  }
  if (worst > 1.0e-3f) {
    addOk = false;
    printf("map-add: MISMATCH (max |diff| %.6g)\n",
           static_cast<double>(worst));
  } else {
    printf("map-add: ok (max |diff| %.6g)\n", static_cast<double>(worst));
  }

  printf("timing softmax %.3f ms/iteration (%d x %d, %d iterations)\n",
         softmaxMs, kRows, kCols, kIterations);
  printf("timing layernorm %.3f ms/iteration (%d x %d, %d iterations)\n",
         normMs, kRows, kCols, kIterations);
  printf("timing matmul %.3f ms/iteration (%d x %d times %d x %d, %d "
         "iterations)\n",
         matmulMs, kRows, kCols, kCols, kRows, kIterations);
  printf("timing map-add %.3f ms/iteration (%lu elements, %d iterations)\n",
         addMs, n, kIterations);

  return (softmaxOk && normOk && matmulOk && addOk) ? 0 : 1;
}
