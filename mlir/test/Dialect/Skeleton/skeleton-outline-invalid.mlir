// RUN: mlir-opt %s -skeleton-outline -verify-diagnostics

#map = affine_map<(d0) -> (d0)>

// The host dispatch and the GPU bridge handle one result per operator unit.

func.func @multi_result(%a: tensor<4xf32>, %b: tensor<4xf32>,
                        %c: tensor<4xf32>, %d: tensor<4xf32>)
    -> (tensor<4xf32>, tensor<4xf32>) {
  // expected-error@+1 {{operator units with more than one result are not supported}}
  %r:2 = linalg.generic {
      indexing_maps = [#map, #map, #map, #map],
      iterator_types = ["parallel"],
      skeleton.preference = #skeleton.preference<"CPU">}
    ins(%a, %b : tensor<4xf32>, tensor<4xf32>)
    outs(%c, %d : tensor<4xf32>, tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32, %o1: f32, %o2: f32):
      linalg.yield %x, %y : f32, f32
  } -> (tensor<4xf32>, tensor<4xf32>)
  return %r#0, %r#1
}
