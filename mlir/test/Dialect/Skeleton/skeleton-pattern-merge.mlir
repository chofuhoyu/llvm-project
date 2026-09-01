// RUN: mlir-opt %s -skeleton-pattern-merge -split-input-file -verify-diagnostics | FileCheck %s

// Test: skeleton.map {pure_fn = @add_fn} → skeleton.vector_add

func.func @add_fn(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @test_merge_add(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %r = skeleton.map pure_fn = @add_fn
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>)
  return %r : tensor<8xf32>
}

// CHECK-LABEL: func @test_merge_add
// CHECK: skeleton.vector_add
// CHECK-NOT: skeleton.map

// -----

// Test: skeleton.map with mul function is NOT merged (no named op for mul yet)

func.func @mul_fn(%a: f32, %b: f32) -> f32 {
  %r = arith.mulf %a, %b : f32
  return %r : f32
}

func.func @test_no_merge_mul(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %r = skeleton.map pure_fn = @mul_fn
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>)
  return %r : tensor<8xf32>
}

// CHECK-LABEL: func @test_no_merge_mul
// CHECK: skeleton.map
// CHECK-NOT: skeleton.vector_add

// -----

// Test: skeleton.map with preference preserved after merge

func.func @add_fn(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @test_merge_with_pref(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %r = skeleton.map preference = #skeleton.preference<"GPU"> pure_fn = @add_fn
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>)
  return %r : tensor<8xf32>
}

// CHECK-LABEL: func @test_merge_with_pref
// CHECK: skeleton.vector_add preference = <"GPU">
// CHECK-NOT: skeleton.map

// -----

// Test: 2D map{pure_fn=@add} does NOT merge — vector_add is 1D-only.

func.func @add_fn(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @test_no_merge_2d(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>, %arg2: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %r = skeleton.map pure_fn = @add_fn
    ins(%arg0, %arg1 : tensor<2x4xf32>, tensor<2x4xf32>)
    outs(%arg2 : tensor<2x4xf32>)
  return %r : tensor<2x4xf32>
}

// CHECK-LABEL: func @test_no_merge_2d
// CHECK: skeleton.map
// CHECK-NOT: skeleton.vector_add

// -----

// Test: swapped argument order (addf is commutative) still merges.

func.func @add_swapped(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %b, %a : f32
  return %r : f32
}

func.func @test_merge_swapped(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %r = skeleton.map pure_fn = @add_swapped
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>)
  return %r : tensor<8xf32>
}

// CHECK-LABEL: func @test_merge_swapped
// CHECK: skeleton.vector_add
// CHECK-NOT: skeleton.map

// -----

// Test: body with extra unrelated instructions still merges (root-op match).

func.func @add_extra(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  %unused = arith.mulf %a, %b : f32
  return %r : f32
}

func.func @test_merge_extra(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %r = skeleton.map pure_fn = @add_extra
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>)
  return %r : tensor<8xf32>
}

// CHECK-LABEL: func @test_merge_extra
// CHECK: skeleton.vector_add
// CHECK-NOT: skeleton.map

// -----

// Test: f64 addf merges (vector_add accepts any float element type).

func.func @add_f64(%a: f64, %b: f64) -> f64 {
  %r = arith.addf %a, %b : f64
  return %r : f64
}

func.func @test_merge_f64(%arg0: tensor<8xf64>, %arg1: tensor<8xf64>, %arg2: tensor<8xf64>) -> tensor<8xf64> {
  %r = skeleton.map pure_fn = @add_f64
    ins(%arg0, %arg1 : tensor<8xf64>, tensor<8xf64>)
    outs(%arg2 : tensor<8xf64>)
  return %r : tensor<8xf64>
}

// CHECK-LABEL: func @test_merge_f64
// CHECK: skeleton.vector_add
// CHECK-NOT: skeleton.map

// -----

// Test: integer add (arith.addi) merges — vector_add is type-polymorphic,
// mirroring linalg.add.

func.func @add_int(%a: i32, %b: i32) -> i32 {
  %r = arith.addi %a, %b : i32
  return %r : i32
}

func.func @test_merge_addi(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>, %arg2: tensor<8xi32>) -> tensor<8xi32> {
  %r = skeleton.map pure_fn = @add_int
    ins(%arg0, %arg1 : tensor<8xi32>, tensor<8xi32>)
    outs(%arg2 : tensor<8xi32>)
  return %r : tensor<8xi32>
}

// CHECK-LABEL: func @test_merge_addi
// CHECK: skeleton.vector_add
// CHECK-NOT: skeleton.map

// -----

// Test: multi-block body does NOT merge — the inlining in skeleton-to-linalg
// only copies the entry block, so merging here would silently drop blocks.

func.func @add_if(%a: f32, %b: f32) -> f32 {
  %cond = arith.cmpf oeq, %a, %b : f32
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %r1 = arith.addf %a, %b : f32
  cf.br ^bb3(%r1 : f32)
^bb2:
  %r2 = arith.subf %a, %b : f32
  cf.br ^bb3(%r2 : f32)
^bb3(%r: f32):
  return %r : f32
}

func.func @test_no_merge_if(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %r = skeleton.map pure_fn = @add_if
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>)
  return %r : tensor<8xf32>
}

// CHECK-LABEL: func @test_no_merge_if
// CHECK: skeleton.map
// CHECK-NOT: skeleton.vector_add
