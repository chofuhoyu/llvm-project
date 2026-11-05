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
//   - `T &` is a single element and becomes memref<T>; it is how a reduction
//     writes its scalar result.
//
// A bare pointer (`T *`) says nothing about its length, so it is not a data
// argument: write the length as an array reference's extent, or pass the
// pointer and the length together as a span.
//
// Scalar arguments (`float init`, `int n`, ...) are passed by value and reach
// the operator unit as scalars. A scalar input to `skeleton_map` is broadcast
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

#endif // MLIR_DIALECT_SKELETON_SKELETONOPS_H
