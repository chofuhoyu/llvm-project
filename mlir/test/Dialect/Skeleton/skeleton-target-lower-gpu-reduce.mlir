// RUN: mlir-opt %s -skeleton-target-lower | FileCheck %s

// A reduction on a GPU target. The reduction dimension has to become a
// sequential loop inside the parallel loop over the output, so that each output
// element is computed by one thread: the element is accumulated in the
// destination buffer, which the kernel can reach.
//
// The failure this pins down is the opposite shape -- the whole reduction
// mapped across threads with an `scf.reduce` combiner running on each of them --
// which is what an unsupported lowering looks like.

func.func private @gpu_red(%a: tensor<4x8xf32>, %m: tensor<4xf32>) -> tensor<4xf32> attributes {skeleton.target = #skeleton.target<"GPU">} {
  %0 = linalg.reduce ins(%a : tensor<4x8xf32>) outs(%m : tensor<4xf32>) dimensions = [1] (%e: f32, %acc: f32) {
    %r = arith.maximumf %e, %acc : f32
    linalg.yield %r : f32
  }
  return %0 : tensor<4xf32>
}

func.func @main() {
  %a = arith.constant dense<1.0> : tensor<4x8xf32>
  %m = arith.constant dense<0.0> : tensor<4xf32>
  %r = call @gpu_red(%a, %m) : (tensor<4x8xf32>, tensor<4xf32>) -> tensor<4xf32>
  return
}

// CHECK-LABEL: func.func private @gpu_red
// The parallel loop walks the output; the reduction dimension is a sequential
// loop inside it that loads and stores the accumulator in the destination.
// CHECK: scf.parallel
// CHECK-NEXT: scf.for
// CHECK: memref.load
// CHECK: arith.maximumf
// CHECK: memref.store
// A combiner mapped across threads would leave an scf.reduce region behind.
// CHECK-NOT: scf.reduce.return
// CHECK: scf.reduce
// CHECK-NOT: scf.reduce.return
// CHECK: return
