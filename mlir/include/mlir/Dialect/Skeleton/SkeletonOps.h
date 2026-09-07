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
// written here. Data arguments are array pointers (e.g. float*); the pass maps
// each to a memref<?xf32>. The pure function is passed by name as the first
// argument and must carry the "skeleton.pure" annotation.
//
// To reference the rewritten host function from the execution driver, declare
// it with extern "C" so its symbol stays unmangled.

#ifndef MLIR_DIALECT_SKELETON_SKELETONOPS_H
#define MLIR_DIALECT_SKELETON_SKELETONOPS_H

namespace skeleton_detail {
template <typename T> struct DataElement;
template <typename T> struct DataElement<T *> { using type = T; };
} // namespace skeleton_detail

// skeleton.map: N-ary element-wise operator. fn is applied once per element,
// taking one element from each data array (their element types must agree).
// The number of data arrays is variadic, matching skeleton.map's variadic
// `ins` operand.
template <typename Fn, typename T, typename... Ts>
__attribute__((annotate("skeleton.op", "map")))
typename skeleton_detail::DataElement<T>::type *
skeleton_map(Fn fn, T first, Ts... rest);

// skeleton.reduce: reduces the elements of a single data array with the binary
// function fn.
template <typename Fn, typename T>
__attribute__((annotate("skeleton.op", "reduce")))
typename skeleton_detail::DataElement<T>::type
skeleton_reduce(Fn fn, T input);

#endif // MLIR_DIALECT_SKELETON_SKELETONOPS_H
