// RUN: mlir-opt %s -skeleton-finalize-memref-to-gpu | FileCheck %s

// Verify the pass converts memref ops, gpu.dealloc and gpu.memcpy to LLVM,
// while leaving gpu.alloc for the subsequent GPU pipeline.

// CHECK-LABEL: func.func @basic
// CHECK: gpu.wait
// CHECK: gpu.alloc async
// CHECK: llvm.call @mgpuMemcpy
// CHECK: llvm.call @mgpuMemFree
// CHECK-NOT: gpu.dealloc
// CHECK-NOT: gpu.memcpy
func.func @basic(%host : memref<2x2xf32>) {
  %t0 = gpu.wait async
  %dev, %t1 = gpu.alloc async [%t0] () : memref<2x2xf32>
  %t2 = gpu.memcpy async [%t1] %dev, %host : memref<2x2xf32>, memref<2x2xf32>
  %t3 = gpu.dealloc async [%t2] %dev : memref<2x2xf32>
  return
}

// Dummy GPU target function to activate the pass.
func.func @GPU_target() attributes {skeleton.target = "GPU"} {
  return
}
