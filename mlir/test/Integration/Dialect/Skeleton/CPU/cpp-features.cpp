// Every day C++ around an operator: the host side of the program is ordinary
// C++ and has to stay ordinary C++.
//
// The program mixes an operator call with the constructs a real host uses --
// local objects with destructors (`std::vector`, `std::string`), a global
// constant, a template instantiation, recursion, a function pointer, branching
// and `printf` -- and prints a value derived from each so that a wrong
// lowering shows up as a wrong number rather than as a crash.
//
// A pure function is not part of the host program: it exists only inside the
// operator units that reference it, so the host calls an ordinary function
// when it needs the same computation (`host_add` below).

#include "mlir/Dialect/Skeleton/SkeletonOps.h"

#include <cstdio>
#include <string>
#include <vector>

static const int kScale = 3;

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

template <typename T> T twice(T value) {
  return value + value;
}

int fib(int n) {
  if (n < 2)
    return n;
  return fib(n - 1) + fib(n - 2);
}

float host_add(float a, float b) {
  return a + b;
}

float apply(float (*fn)(float, float), float a, float b) {
  return fn(a, b);
}

int main() {
  std::vector<float> a{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> b{10.0f, 20.0f, 30.0f, 40.0f};
  std::vector<float> c(a.size(), 0.0f);

  skeleton_detail::span<const float> sa{a.data(), a.size()};
  skeleton_detail::span<const float> sb{b.data(), b.size()};
  skeleton_detail::span<float> sc{c.data(), c.size()};
  vec_add(sa, sb, sc);

  for (size_t i = 0; i < c.size(); ++i)
    printf("c[%zu] = %f\n", i, c[i]);

  std::string name = "features";
  printf("string = %s\n", name.c_str());
  printf("scale = %d\n", kScale);
  printf("twice(scale) = %d\n", twice(kScale));
  printf("fib(10) = %d\n", fib(10));
  printf("dynamic = %f\n", apply(host_add, 1.5f, 2.5f));
  printf("twice add = %f\n", twice(host_add(20.0f, 22.0f)));

  int branches = 0;
  for (int i = 0; i < 10; ++i)
    if (i % 2 == 0)
      ++branches;
  printf("branches = %d\n", branches);
  return 0;
}
