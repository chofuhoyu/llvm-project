// RUN: mlir-opt %s -skeleton-finalize-memref-to-gpu | FileCheck %s

// Verify the pass converts host-side memref lifecycle ops (here the
// memref.copy that fills a host-shared UVM buffer) to LLVM, while leaving
// gpu.alloc host_shared and gpu.dealloc for the subsequent GPU pipeline,
// which lowers them to mgpuMemAlloc/mgpuMemFree.

// CHECK-LABEL: func.func @basic
// CHECK: gpu.alloc {{.*}}host_shared
// CHECK: llvm.intr.memcpy
// CHECK: gpu.dealloc
func.func @basic(%host : memref<2x2xf32>) {
  %dev = gpu.alloc host_shared () : memref<2x2xf32>
  memref.copy %host, %dev : memref<2x2xf32> to memref<2x2xf32>
  gpu.dealloc %dev : memref<2x2xf32>
  return
}

// Dummy GPU target function to activate the pass.
func.func @GPU_target() attributes {skeleton.target = #skeleton.target<"GPU">} {
  return
}
