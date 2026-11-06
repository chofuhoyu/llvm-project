// The first program whose device operator works on memory the runtime
// allocated, instead of on host memory copied into a device buffer.
//
// `skeleton_detail::buffer<T>` asks the runtime for its storage: Unified
// Memory when the machine has a GPU, aligned host memory otherwise. The unit
// built for such a host is marked managed, so lowering leaves the operands
// alone -- no gpu.alloc, no copy in, no copy back. The kernel writes the
// result where the host will read it, which is why the host's own buffers are
// runtime buffers rather than stack arrays.
//
// The length is a run-time value (a buffer carries it in its size member), so
// this is also the one-dimensional dynamic-shape case the device path used to
// refuse.
//
// The destination starts filled with a sentinel and every element is checked
// afterwards. Nothing orders the host's read after the kernel except the
// synchronization the lowering emits at the operator boundary, so a buffer this
// size read too early comes back with sentinels in it rather than with the
// right sums.
//
// Expected c[i] = a[i] + b[i] with a[i] = i % 97 and b[i] = i % 89.

#include "mlir/Dialect/Skeleton/SkeletonOps.h"

#include <cstdio>

namespace {
constexpr unsigned long kElements = 1u << 18;
}

__attribute__((annotate("skeleton.pure")))
float my_add(float a, float b) {
  return a + b;
}

extern "C" __attribute__((annotate("skeleton.region", "GPU")))
void vec_add(const skeleton_detail::buffer<float> &a,
             const skeleton_detail::buffer<float> &b,
             const skeleton_detail::buffer<float> &out) {
  skeleton_map(my_add, out, a, b);
}

int main() {
  const unsigned long n = kElements;
  skeleton_detail::buffer<float> a(n);
  skeleton_detail::buffer<float> b(n);
  skeleton_detail::buffer<float> c(n);
  for (unsigned long i = 0; i < n; ++i) {
    a.data[i] = static_cast<float>(i % 97);
    b.data[i] = static_cast<float>(i % 89);
    c.data[i] = -1.0f;
  }

  vec_add(a, b, c);

  unsigned long wrong = 0;
  unsigned long firstWrong = 0;
  for (unsigned long i = 0; i < n; ++i) {
    float want = a.data[i] + b.data[i];
    if (c.data[i] != want) {
      if (wrong == 0)
        firstWrong = i;
      ++wrong;
    }
  }
  if (wrong != 0) {
    printf("managed add: MISMATCH %lu of %lu elements, first at %lu "
           "(got %.1f want %.1f)\n",
           wrong, n, firstWrong, static_cast<double>(c.data[firstWrong]),
           static_cast<double>(a.data[firstWrong] + b.data[firstWrong]));
    return 1;
  }

  printf("managed add: ok (%lu elements)\n", n);
  printf("%.1f\n", static_cast<double>(c.data[0]));
  printf("%.1f\n", static_cast<double>(c.data[1]));
  printf("%.1f\n", static_cast<double>(c.data[2]));
  printf("%.1f\n", static_cast<double>(c.data[n - 1]));
  return 0;
}
