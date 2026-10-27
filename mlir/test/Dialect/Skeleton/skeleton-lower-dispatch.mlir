// RUN: mlir-opt %s -skeleton-lower-dispatch | FileCheck %s

// A direct call to an outlined op unit becomes a resolve followed by an
// indirect call with the operator's own signature. The declaration is erased
// once nothing refers to it.

// CHECK-LABEL: llvm.func @skeleton_rt_resolve(i64) -> !llvm.ptr
// CHECK-LABEL: llvm.func @host
// CHECK: %[[ID:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: %[[IMPL:.*]] = llvm.call @skeleton_rt_resolve(%[[ID]]) : (i64) -> !llvm.ptr
// CHECK: %[[R:.*]] = llvm.call %[[IMPL]](%{{.*}}, %{{.*}}) : !llvm.ptr, (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK: llvm.return %[[R]]
// CHECK-NOT: __skeleton_op_0

llvm.func private @__skeleton_op_0(!llvm.ptr, !llvm.ptr) -> !llvm.ptr

llvm.func @host(%a: !llvm.ptr, %b: !llvm.ptr) -> !llvm.ptr {
  %0 = llvm.call @__skeleton_op_0(%a, %b) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  llvm.return %0 : !llvm.ptr
}
