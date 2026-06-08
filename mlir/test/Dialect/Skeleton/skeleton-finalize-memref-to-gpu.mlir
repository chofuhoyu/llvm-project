// RUN: mlir-opt %s -skeleton-finalize-memref-to-gpu | FileCheck %s

// Verify the pass converts memref ops and gpu.memcpy to LLVM, while
// leaving gpu.alloc and gpu.dealloc for the subsequent GPU pipeline
// (sync versions are now supported by gpu-to-llvm, #191661).

// CHECK-LABEL: func.func @basic
// CHECK: gpu.alloc
// CHECK: gpu.wait
// CHECK: llvm.call @mgpuMemcpy
// CHECK: gpu.dealloc
func.func @basic(%host : memref<2x2xf32>) {
  %dev = gpu.alloc () : memref<2x2xf32>
  %t0 = gpu.wait async
  %t1 = gpu.memcpy async [%t0] %dev, %host : memref<2x2xf32>, memref<2x2xf32>
  gpu.dealloc %dev : memref<2x2xf32>
  return
}

// Dummy GPU target function to activate the pass.
func.func @GPU_target() attributes {skeleton.target = "GPU"} {
  return
}
