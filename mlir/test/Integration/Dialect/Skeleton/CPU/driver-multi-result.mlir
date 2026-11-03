// REQUIRES: mlir-skeleton-driver
// RUN: mlir-skeleton-driver %s -entry main --shared-libs=%mlir_c_runner_utils,%mlir_runner_utils | FileCheck %s

// An operator unit that prefers the CPU may have several results. resolve
// only tries the preferred device and then the CPU, so this unit is never
// compiled for the GPU, whose bridge gives an operator one output memref.

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @main() {
  %a = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %b = arith.constant dense<[[2.0, 3.0], [4.0, 5.0]]> : tensor<2x2xf32>
  %c = arith.constant dense<0.0> : tensor<2x2xf32>
  %d = arith.constant dense<0.0> : tensor<2x2xf32>

  %r:2 = linalg.generic {
      indexing_maps = [#map, #map, #map, #map],
      iterator_types = ["parallel", "parallel"],
      skeleton.preference = #skeleton.preference<"CPU">}
    ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>)
    outs(%c, %d : tensor<2x2xf32>, tensor<2x2xf32>) {
    ^bb0(%x: f32, %y: f32, %o1: f32, %o2: f32):
      %sum = arith.addf %x, %y : f32
      %product = arith.mulf %x, %y : f32
      linalg.yield %sum, %product : f32, f32
  } -> (tensor<2x2xf32>, tensor<2x2xf32>)

  %r0 = tensor.cast %r#0 : tensor<2x2xf32> to tensor<*xf32>
  call @printMemrefF32(%r0) : (tensor<*xf32>) -> ()
  %r1 = tensor.cast %r#1 : tensor<2x2xf32> to tensor<*xf32>
  call @printMemrefF32(%r1) : (tensor<*xf32>) -> ()
  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)

// sum = [[3, 5], [7, 9]], product = [[2, 6], [12, 20]].
// CHECK: Unranked Memref
// CHECK: {{\[\[}}3,   5],
// CHECK: 7,   9]]
// CHECK: Unranked Memref
// CHECK: {{\[\[}}2,   6],
// CHECK: 12,   20]]
