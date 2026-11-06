// C++ entry point of the Skeleton front end.
//
// A host function marked with
//
//   __attribute__((annotate("skeleton.region", "CPU")))
//
// is an operator boundary. Its body is one call to a helper below, and that
// call becomes the operator unit; the function itself stays an ordinary C++
// function so that the rest of the program calls it like any other. Everything
// outside the marked functions is plain C++: `main`, control flow, standard
// library containers and `printf` all ride the normal CIR -> LLVM path.
//
// The boundary is destination-passing. The helper does not return a value; it
// writes into a buffer the caller owns. The first data argument after the
// function is always the destination, and its shape is the operator's result
// shape. A program composes operators by calling several such functions from
// `main`, not by nesting them:
//
//   __attribute__((annotate("skeleton.pure")))
//   float my_add(float a, float b) { return a + b; }
//
//   extern "C" __attribute__((annotate("skeleton.region", "CPU")))
//   void vec_add(const skeleton_detail::span<const float> &a,
//                const skeleton_detail::span<const float> &b,
//                const skeleton_detail::span<float> &out) {
//     skeleton_map(my_add, out, a, b);
//   }
//
// Data arguments come in three forms, and the form is where the shape comes
// from:
//
//   - `T (&a)[N]` (and `T (&a)[N][M]`, ...) carries a compile-time extent per
//     dimension and becomes memref<NxT> (memref<NxMxT>, ...);
//   - `const skeleton_detail::span<T> &` carries its length in the `size`
//     member and becomes memref<?xT>, with that length as the run-time extent;
//   - `skeleton_detail::buffer<T> &` is the same shape, but its storage comes
//     from the runtime and the unit is marked managed, so a device kernel
//     works on that storage directly instead of on a copy of it;
//   - `T &` is a single element and becomes memref<T>; it is how a reduction
//     writes its scalar result.
//
// A bare pointer (`T *`) says nothing about its length, so it is not a data
// argument: write the length as an array reference's extent, or pass the
// pointer and the length together as a span.
//
// Scalar arguments (`float init`, `int n`, ...) reach the operator unit as
// scalars: skeleton_reduce takes its identity by value, and every other helper
// takes its inputs by reference, so a scalar written at the call site is a
// pointer to it that the split loads before the call. A scalar input to `skeleton_map` is broadcast
// over the destination, which is how a reduction result feeds the next
// element-wise step.
//
// A pure function is marked with "__attribute__((annotate("skeleton.pure")))".
// It is not part of the host program: only the operator unit references it, so
// it may only compute with scalars. Its body is a single block of arithmetic,
// comparisons, selects and the math builtins CIR models as `cir.*` ops (for
// example `expf`); anything else is rejected with a diagnostic rather than
// silently compiled differently.
//
// Host functions that reach an operator must use `extern "C"` to give it a
// stable symbol.

#ifndef MLIR_DIALECT_SKELETON_SKELETONOPS_H
#define MLIR_DIALECT_SKELETON_SKELETONOPS_H

#include "mlir/Dialect/Skeleton/Runtime/SkeletonRuntime.h"

namespace skeleton_detail {

// A non-owning view of an array whose length is only known at run time. Passing
// one by reference is how a host says its extent is not a compile-time one: the
// pass takes the element type from the pointee of `data`, the size member as
// the length, and the operator unit then takes a memref whose dynamic extent is
// that length.
template <typename T> struct span {
  T *data;
  unsigned long size;
};

// A buffer the runtime allocates and owns, so that the memory an operator reads
// and writes can be the memory a device kernel sees. The allocation is the
// runtime's choice: Unified Memory when the machine has a GPU, aligned host
// memory otherwise, which keeps a program that never reaches a device working
// and keeps an artifact free of a CUDA dependency.
//
// Passing one by reference is how a host says its operator's buffers belong to
// the runtime: the operator unit is then marked managed, and lowering leaves
// the buffers alone instead of giving each operand a device allocation and
// copying it in and out. The type has the same two members as span, so a read
// of `data` and `size` is all the boundary needs.
template <typename T> struct buffer {
  // skeleton_rt_alloc reports and exits when it cannot allocate, so the
  // pointer is usable as soon as the constructor returns.
  explicit buffer(unsigned long length)
      : data(static_cast<T *>(
            skeleton_rt_alloc(allocationDevice(), length * sizeof(T)))),
        size(length) {}

  ~buffer() { skeleton_rt_free(allocationDevice(), data); }

  buffer(const buffer &) = delete;
  buffer &operator=(const buffer &) = delete;

  T *data;
  unsigned long size;

private:
  // The device whose memory is the most widely usable. Growing a program from
  // CPU-only to device-capable must not change its allocations, so this asks
  // the runtime rather than the annotation, and the answer only depends on
  // whether the machine has a GPU at all.
  static int allocationDevice() {
    return skeleton_rt_probe(SKELETON_DEVICE_GPU) ? SKELETON_DEVICE_GPU
                                                  : SKELETON_DEVICE_CPU;
  }
};

} // namespace skeleton_detail

// skeleton.map: N-ary element-wise operator. `out` is the destination and every
// input must be broadcastable to it: an input with a lower rank matches the
// leading dimensions of the result, and a scalar input is broadcast everywhere.
// The pure function is applied once per destination element, taking one element
// per input.
template <typename Fn, typename Out, typename... Ins>
__attribute__((annotate("skeleton.op", "map")))
void skeleton_map(Fn fn, const Out &out, const Ins &...ins);

// skeleton.reduce: reduces `in` over the trailing dimensions that `out` does
// not have, so a destination of the same rank reduces nothing and a scalar
// destination (`T &`) reduces every element. `init` is the reduction's
// identity: it is written to every destination element before the first
// element is combined in, so min/max/product reductions pass their own
// identity instead of relying on a zero the operator cannot know about.
template <typename Fn, typename Out, typename In, typename Init>
__attribute__((annotate("skeleton.op", "reduce")))
void skeleton_reduce(Fn fn, const Out &out, const In &in, Init init);

// skeleton.matmul: C = A * B, with `out` the destination, shaped [M][N] for
// A shaped [M][K] and B shaped [K][N]. It takes no pure function: the
// contraction is the operator, and only the linalg matrix multiply expresses
// it. The destination is the first data argument, like every other operator,
// and it is overwritten rather than accumulated into.
template <typename Out, typename A, typename B>
__attribute__((annotate("skeleton.op", "matmul")))
void skeleton_matmul(const Out &out, const A &a, const B &b);

#endif // MLIR_DIALECT_SKELETON_SKELETONOPS_H
