// RUN: mlir-opt %s -skeleton-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// Test: skeleton.map (pure_fn, no body) → linalg.generic

func.func @my_add(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @test_map_leaf(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %0 = skeleton.map pure_fn = @my_add
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>)
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @test_map_leaf
// CHECK: linalg.generic
// CHECK-SAME: ins({{.*}} : tensor<8xf32>, tensor<8xf32>)
// CHECK-SAME: outs({{.*}} : tensor<8xf32>)
// CHECK-NOT: skeleton.map

// -----

// Test: skeleton.map with preference preserved on linalg.generic

func.func @my_add(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @test_map_with_pref(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %0 = skeleton.map preference = #skeleton.preference<"GPU"> pure_fn = @my_add
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>)
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @test_map_with_pref
// CHECK: linalg.generic
// CHECK-SAME: skeleton.preference = #skeleton.preference<"GPU">
// CHECK-NOT: skeleton.map

// -----

// Test: skeleton.reduce (pure_fn, no body) → linalg.reduce

func.func @my_sum(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @test_reduce_leaf(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = skeleton.reduce pure_fn = @my_sum
    ins(%arg0 : tensor<8xf32>)
    outs(%arg1 : tensor<f32>)
  return %0 : tensor<f32>
}

// CHECK-LABEL: func @test_reduce_leaf
// CHECK: linalg.reduce
// CHECK-SAME: ins({{.*}} : tensor<8xf32>)
// CHECK-SAME: outs({{.*}} : tensor<f32>)
// CHECK-NOT: skeleton.reduce

// -----

// Test: skeleton.reduce with preference preserved

func.func @my_sum(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @test_reduce_with_pref(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = skeleton.reduce preference = #skeleton.preference<"CPU"> pure_fn = @my_sum
    ins(%arg0 : tensor<8xf32>)
    outs(%arg1 : tensor<f32>)
  return %0 : tensor<f32>
}

// CHECK-LABEL: func @test_reduce_with_pref
// CHECK: linalg.reduce
// CHECK-SAME: skeleton.preference = #skeleton.preference<"CPU">
// CHECK-NOT: skeleton.reduce

// -----

// Test: skeleton.vector.add → linalg.add

func.func @test_vector_add(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %0 = skeleton.vector.add
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>)
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @test_vector_add
// CHECK: linalg.add
// CHECK-SAME: ins({{.*}} : tensor<8xf32>, tensor<8xf32>)
// CHECK-SAME: outs({{.*}} : tensor<8xf32>)
// CHECK-NOT: skeleton.vector.add

// -----

// Test: skeleton.vector.add with preference → linalg.add with attr

func.func @test_vector_add_with_pref(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %0 = skeleton.vector.add preference = #skeleton.preference<"GPU">
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>)
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @test_vector_add_with_pref
// CHECK: linalg.add
// CHECK-SAME: skeleton.preference = #skeleton.preference<"GPU">
// CHECK-NOT: skeleton.vector.add

// -----

// Test: skeleton.vector.add with integer element type → linalg.add.
// vector.add is type-polymorphic, and linalg.add accepts the same element
// types (T1).

func.func @test_vector_add_i32(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>, %arg2: tensor<8xi32>) -> tensor<8xi32> {
  %0 = skeleton.vector.add
    ins(%arg0, %arg1 : tensor<8xi32>, tensor<8xi32>)
    outs(%arg2 : tensor<8xi32>)
  return %0 : tensor<8xi32>
}

// CHECK-LABEL: func @test_vector_add_i32
// CHECK: linalg.add
// CHECK-SAME: ins({{.*}} : tensor<8xi32>, tensor<8xi32>)
// CHECK-SAME: outs({{.*}} : tensor<8xi32>)
// CHECK-NOT: skeleton.vector.add

// -----

// Test: skeleton.map with a multi-block pure_fn is rejected. linalg regions
// must be a single straight-line block, so such bodies cannot be inlined.

func.func @my_conditional(%a: f32, %b: f32) -> f32 {
  %c = arith.cmpf olt, %a, %b : f32
  cf.cond_br %c, ^bb1, ^bb2
^bb1:
  %0 = arith.addf %a, %b : f32
  cf.br ^bb3(%0 : f32)
^bb2:
  %1 = arith.mulf %a, %b : f32
  cf.br ^bb3(%1 : f32)
^bb3(%r: f32):
  return %r : f32
}

func.func @test_map_multi_block(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error@+1 {{pure_fn 'my_conditional' must have a single-block body to lower to linalg}}
  %0 = skeleton.map pure_fn = @my_conditional
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>)
  return %0 : tensor<8xf32>
}

// -----

// Test: skeleton.reduce with a multi-block pure_fn is rejected.

func.func @my_conditional(%a: f32, %b: f32) -> f32 {
  %c = arith.cmpf olt, %a, %b : f32
  cf.cond_br %c, ^bb1, ^bb2
^bb1:
  %0 = arith.addf %a, %b : f32
  cf.br ^bb3(%0 : f32)
^bb2:
  %1 = arith.mulf %a, %b : f32
  cf.br ^bb3(%1 : f32)
^bb3(%r: f32):
  return %r : f32
}

func.func @test_reduce_multi_block(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{pure_fn 'my_conditional' must have a single-block body to lower to linalg}}
  %0 = skeleton.reduce pure_fn = @my_conditional
    ins(%arg0 : tensor<8xf32>)
    outs(%arg1 : tensor<f32>)
  return %0 : tensor<f32>
}

// -----

// Test: skeleton.map with a pure_fn that is only declared (no body) is
// rejected.

func.func private @my_declared(f32, f32) -> f32

func.func @test_map_declared(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error@+1 {{pure_fn 'my_declared' must be defined to lower to linalg}}
  %0 = skeleton.map pure_fn = @my_declared
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>)
  return %0 : tensor<8xf32>
}

// -----

// Test: skeleton.map with a body region → linalg.generic

func.func @test_map_body(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = skeleton.map ins(%arg0 : tensor<8xf32>) outs(%arg1 : tensor<8xf32>)
      (%e: f32, %i: f32) {
    %x = arith.addf %e, %i : f32
    skeleton.yield %x : f32
  }
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @test_map_body
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: linalg.yield
// CHECK-NOT: skeleton.map
// CHECK-NOT: skeleton.yield

// -----

// Test: skeleton.reduce with a body region → linalg.reduce

func.func @test_reduce_body(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = skeleton.reduce ins(%arg0 : tensor<8xf32>) outs(%arg1 : tensor<f32>)
      (%acc: f32, %e: f32) {
    %x = arith.addf %acc, %e : f32
    skeleton.yield %x : f32
  }
  return %0 : tensor<f32>
}

// CHECK-LABEL: func @test_reduce_body
// CHECK: linalg.reduce
// CHECK: arith.addf
// CHECK: linalg.yield
// CHECK-NOT: skeleton.reduce
// CHECK-NOT: skeleton.yield

// -----

// Test: nested skeleton operators in a body region are rejected.

func.func @test_nested_body(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error@+1 {{nested skeleton operators in a body region are not supported}}
  %0 = skeleton.map ins(%arg0 : tensor<8xf32>) outs(%arg1 : tensor<8xf32>)
      (%e: f32, %i: f32) {
    %inner = skeleton.map ins(%arg0 : tensor<8xf32>) outs(%arg1 : tensor<8xf32>)
        (%e2: f32, %i2: f32) {
      skeleton.yield %i2 : f32
    }
    skeleton.yield %i : f32
  }
  return %0 : tensor<8xf32>
}
