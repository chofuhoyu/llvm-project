// RUN: mlir-opt %s -skeleton-target-lower | FileCheck %s

// A GPU-target function may return void. The bridge must still allocate,
// copy and deallocate the UVM buffers of its memref operands; it must not
// assume the call has a memref result to redirect. This used to trip an
// assertion in Operation::getResult(0).

func.func private @gpu_add_void(%a: memref<4xf32>, %b: memref<4xf32>, %c: memref<4xf32>) attributes {skeleton.target = #skeleton.target<"GPU">} {
  linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
    ins(%a, %b : memref<4xf32>, memref<4xf32>)
    outs(%c : memref<4xf32>) {
  ^bb0(%x: f32, %y: f32, %z: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  }
  return
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %one = arith.constant 1.0 : f32
  %a = memref.alloc() : memref<4xf32>
  %b = memref.alloc() : memref<4xf32>
  %c = memref.alloc() : memref<4xf32>
  memref.store %one, %a[%c0] : memref<4xf32>
  memref.store %one, %b[%c0] : memref<4xf32>
  memref.store %one, %c[%c0] : memref<4xf32>
  call @gpu_add_void(%a, %b, %c) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
  return
}

// CHECK-LABEL: func.func @main
// CHECK: gpu.alloc {{.*}}host_shared
// CHECK-NEXT: memref.copy
// CHECK: gpu.alloc {{.*}}host_shared
// CHECK-NEXT: memref.copy
// CHECK: gpu.alloc {{.*}}host_shared
// CHECK-NEXT: memref.copy
// CHECK: call @gpu_add_void
// Each writable operand is copied back and deallocated; there is no call
// result to replace.
// CHECK-NEXT: memref.copy
// CHECK-NEXT: gpu.dealloc
// CHECK-NEXT: memref.copy
// CHECK-NEXT: gpu.dealloc
// CHECK-NEXT: memref.copy
// CHECK-NEXT: gpu.dealloc
// CHECK-NOT: gpu.memcpy
// CHECK-NOT: gpu.wait
