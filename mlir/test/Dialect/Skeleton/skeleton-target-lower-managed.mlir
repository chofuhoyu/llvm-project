// RUN: mlir-opt %s -skeleton-target-lower | FileCheck %s

// A managed GPU target takes its operands from memory the runtime allocated.
// Those buffers are already visible to the device, so the bridge has nothing
// to do: no gpu.alloc, no host copy in, no copy back, no dealloc. The device
// still has to be synchronized before the host reads the result.

func.func private @gpu_add_managed(%a: memref<?xf32>, %b: memref<?xf32>, %c: memref<?xf32>) attributes {skeleton.target = #skeleton.target<"GPU">, skeleton.managed} {
  linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
    ins(%a, %b : memref<?xf32>, memref<?xf32>)
    outs(%c : memref<?xf32>) {
  ^bb0(%x: f32, %y: f32, %z: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  }
  return
}

func.func @main(%a: memref<?xf32>, %b: memref<?xf32>, %c: memref<?xf32>) {
  call @gpu_add_managed(%a, %b, %c) : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
  return
}

// CHECK: func.func private @skeleton_rt_sync(i32)
// CHECK-LABEL: func.func @main
// CHECK-NOT: gpu.alloc
// CHECK: call @gpu_add_managed
// CHECK-NEXT: call @skeleton_rt_sync
// CHECK-NOT: memref.copy
// CHECK-NOT: gpu.dealloc
// CHECK: return
