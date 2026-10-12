// C++ entry point of the manual Skeleton path.
//
// Each helper below maps one skeleton operator to a plain C++ function call the
// front-end can recognize:
//
//   __attribute__((annotate("skeleton.op", "map")))
//
// turns the declaration into a cir.func carrying that cir.annotation; the
// cir-call-to-skeleton pass then rewrites a call to it (and the surrounding
// host function) into a func.func that returns a Skeleton dialect op.
//
// The helpers are declarations only and are never defined. The C++ return type
// is a placeholder: the rewritten host function is given its real return type
// (the skeleton op's result tensor) by the pass, independently of what is
// written here. Data arguments are passed by reference so that the host's
// parameter type reaches the call unchanged; that type is where the array's
// shape comes from. A parameter declared as an array reference carries a
// compile-time extent and becomes a static memref (`float (&a)[4]` ->
// memref<4xf32>); one declared as skeleton_detail::span carries a runtime
// extent and becomes a memref with a dynamic extent (`span<float>` ->
// memref<?xf32>); a bare pointer carries no extent and becomes dynamic as
// well. A call's data arguments must denote one and the same element type and
// shape: compile-time extents that disagree are reported by the pass, and
// runtime ones are compared when the rewritten function runs. The pure function
// is passed by name as the first argument and must carry the "skeleton.pure"
// annotation.
//
// To reference the rewritten host function from the execution driver, declare
// it with extern "C" so its symbol stays unmangled.

#ifndef MLIR_DIALECT_SKELETON_SKELETONOPS_H
#define MLIR_DIALECT_SKELETON_SKELETONOPS_H

namespace skeleton_detail {

// The element type of a data argument, so that a helper's placeholder return
// type can name it. Every pointer, array and span layer is peeled: an argument
// of type float*, float[4] or float[2][4] each leaves float.
template <typename T> struct DataElement { using type = T; };
template <typename T> struct DataElement<T *> : DataElement<T> {};
template <typename T, unsigned long N> struct DataElement<T[N]>
    : DataElement<T> {};

// A non-owning view of an array whose length is only known at run time. Naming
// it as a host parameter is how a host says its extent is not a compile-time
// one: the pass takes the element type from the pointee of `data` and the size
// member as the length, and the rewritten function then takes a memref whose
// dynamic extent is that length.
template <typename T> struct span {
  T *data;
  unsigned long size;
};
template <typename T> struct DataElement<span<T>> { using type = T; };

} // namespace skeleton_detail

// skeleton.map: N-ary element-wise operator. fn is applied once per element,
// taking one element from each data array (their element types must agree).
// The number of data arrays is variadic, matching skeleton.map's variadic
// `ins` operand. All data arrays must have the same shape.
template <typename Fn, typename T, typename... Ts>
__attribute__((annotate("skeleton.op", "map")))
typename skeleton_detail::DataElement<T>::type *
skeleton_map(Fn fn, const T &first, const Ts &...rest);

// skeleton.reduce: reduces the elements of a single data array with the binary
// function fn.
template <typename Fn, typename T>
__attribute__((annotate("skeleton.op", "reduce")))
typename skeleton_detail::DataElement<T>::type
skeleton_reduce(Fn fn, const T &input);

#endif // MLIR_DIALECT_SKELETON_SKELETONOPS_H
