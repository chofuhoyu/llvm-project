// RUN: mlir-opt %s -skeleton-target-lower | FileCheck %s

// The GPU bridge gives every memref operand of a GPU-target call a
// host-shared (Unified Memory) allocation and moves the data with synchronous
// host copies. It must not emit explicit device transfers or async tokens:
// those were both racy (a copy stream was never ordered against the kernel
// stream) and unnecessary once host and kernel share the same buffer.

func.func private @gpu_matmul(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {skeleton.target = #skeleton.target<"GPU">} {
  %0 = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%c : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

func.func @main() {
  %a = arith.constant dense<1.0> : tensor<2x2xf32>
  %b = arith.constant dense<2.0> : tensor<2x2xf32>
  %c = arith.constant dense<0.0> : tensor<2x2xf32>
  %r = call @gpu_matmul(%a, %b, %c) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return
}

// A and B are read-only globals, so only C (a writable host buffer) is copied
// back. All three UVM allocations are deallocated: the read-only ones right
// after the call, C after its copy-back.
// The runtime entry point the generated wait goes through is declared once, at
// module level.
// CHECK: func.func private @skeleton_rt_sync(i32)

// CHECK-LABEL: func.func @main
// CHECK: gpu.alloc {{.*}}host_shared
// CHECK-NEXT: memref.copy
// CHECK: gpu.alloc {{.*}}host_shared
// CHECK-NEXT: memref.copy
// CHECK: gpu.alloc {{.*}}host_shared
// CHECK-NEXT: memref.copy
// CHECK: call @gpu_matmul
// The launch is asynchronous and the copy-backs below read what the kernel
// wrote, so the device is synchronized first.
// CHECK: call @skeleton_rt_sync
// CHECK-COUNT-2: gpu.dealloc
// CHECK: memref.copy
// CHECK: gpu.dealloc
// CHECK-NOT: gpu.memcpy
// CHECK-NOT: gpu.wait
