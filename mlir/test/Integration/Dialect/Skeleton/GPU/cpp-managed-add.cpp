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
// Expected c = a + b = [0, 11, 22, 33, 44, 55].

#include "mlir/Dialect/Skeleton/SkeletonOps.h"

#include <cstdio>

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
  const unsigned long n = 6;
  skeleton_detail::buffer<float> a(n);
  skeleton_detail::buffer<float> b(n);
  skeleton_detail::buffer<float> c(n);
  for (unsigned long i = 0; i < n; ++i) {
    a.data[i] = static_cast<float>(i);
    b.data[i] = static_cast<float>(10 * i);
    c.data[i] = 0.0f;
  }

  vec_add(a, b, c);

  for (unsigned long i = 0; i < n; ++i)
    printf("%.1f\n", c.data[i]);
  return 0;
}
