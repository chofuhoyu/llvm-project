// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// MapOp: neither pure_fn nor body → error

func.func @missing_both(%A: memref<?xf32>, %B: memref<?xf32>, %C: memref<?xf32>) {
  %A_t = bufferization.to_tensor %A : memref<?xf32> to tensor<?xf32>
  %B_t = bufferization.to_tensor %B : memref<?xf32> to tensor<?xf32>
  %C_t = bufferization.to_tensor %C : memref<?xf32> to tensor<?xf32>
  // expected-error@+1 {{requires exactly one of 'pure_fn' or a body region}}
  %r = skeleton.map
    ins(%A_t, %B_t : tensor<?xf32>, tensor<?xf32>)
    outs(%C_t : tensor<?xf32>)
  return
}

// -----

// MapOp: both pure_fn and body → error

func.func @map_hybrid(%A: memref<?xf32>, %B: memref<?xf32>, %C: memref<?xf32>) {
  %A_t = bufferization.to_tensor %A : memref<?xf32> to tensor<?xf32>
  %B_t = bufferization.to_tensor %B : memref<?xf32> to tensor<?xf32>
  %C_t = bufferization.to_tensor %C : memref<?xf32> to tensor<?xf32>
  // expected-error@+1 {{requires exactly one of 'pure_fn' or a body region}}
  %r = skeleton.map pure_fn = @my_add
    ins(%A_t, %B_t : tensor<?xf32>, tensor<?xf32>)
    outs(%C_t : tensor<?xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %s = arith.addf %a, %b : f32
    skeleton.yield %s : f32
  }
  return
}

// -----

// MapOp: pure_fn not found in symbol table

func.func @pure_fn_not_found(%A: memref<?xf32>, %B: memref<?xf32>, %C: memref<?xf32>) {
  %A_t = bufferization.to_tensor %A : memref<?xf32> to tensor<?xf32>
  %B_t = bufferization.to_tensor %B : memref<?xf32> to tensor<?xf32>
  %C_t = bufferization.to_tensor %C : memref<?xf32> to tensor<?xf32>
  // expected-error@+1 {{pure_fn @nonexistent not found in symbol table}}
  %r = skeleton.map pure_fn = @nonexistent
    ins(%A_t, %B_t : tensor<?xf32>, tensor<?xf32>)
    outs(%C_t : tensor<?xf32>)
  return
}

// -----

// MapOp: pure_fn signature mismatch (wrong number of params)

func.func @bad_sig(%a: f32, %b: f32, %c: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @pure_fn_bad_param_count(%A: memref<?xf32>, %B: memref<?xf32>, %C: memref<?xf32>) {
  %A_t = bufferization.to_tensor %A : memref<?xf32> to tensor<?xf32>
  %B_t = bufferization.to_tensor %B : memref<?xf32> to tensor<?xf32>
  %C_t = bufferization.to_tensor %C : memref<?xf32> to tensor<?xf32>
  // expected-error@+1 {{expects 3 parameters but skeleton op provides 2 inputs}}
  %r = skeleton.map pure_fn = @bad_sig
    ins(%A_t, %B_t : tensor<?xf32>, tensor<?xf32>)
    outs(%C_t : tensor<?xf32>)
  return
}

// -----

// MapOp: pure_fn result type mismatch

func.func @bad_ret(%a: f32, %b: f32) -> i32 {
  %r = arith.constant 0 : i32
  return %r : i32
}

func.func @pure_fn_bad_ret(%A: memref<?xf32>, %B: memref<?xf32>, %C: memref<?xf32>) {
  %A_t = bufferization.to_tensor %A : memref<?xf32> to tensor<?xf32>
  %B_t = bufferization.to_tensor %B : memref<?xf32> to tensor<?xf32>
  %C_t = bufferization.to_tensor %C : memref<?xf32> to tensor<?xf32>
  // expected-error@+1 {{pure_fn result type mismatch}}
  %r = skeleton.map pure_fn = @bad_ret
    ins(%A_t, %B_t : tensor<?xf32>, tensor<?xf32>)
    outs(%C_t : tensor<?xf32>)
  return
}

// -----

// ReduceOp: neither pure_fn nor body → error

func.func @reduce_missing_both(%input: memref<?xf32>, %init: memref<f32>) {
  %in_t = bufferization.to_tensor %input : memref<?xf32> to tensor<?xf32>
  %init_t = bufferization.to_tensor %init : memref<f32> to tensor<f32>
  // expected-error@+1 {{requires exactly one of 'pure_fn' or a body region}}
  %r = skeleton.reduce
    ins(%in_t : tensor<?xf32>)
    outs(%init_t : tensor<f32>)
  return
}

// -----

// ReduceOp: both pure_fn and body → error

func.func @reduce_hybrid(%input: memref<?xf32>, %init: memref<f32>) {
  %in_t = bufferization.to_tensor %input : memref<?xf32> to tensor<?xf32>
  %init_t = bufferization.to_tensor %init : memref<f32> to tensor<f32>
  // expected-error@+1 {{requires exactly one of 'pure_fn' or a body region}}
  %r = skeleton.reduce pure_fn = @my_sum
    ins(%in_t : tensor<?xf32>)
    outs(%init_t : tensor<f32>) {
  ^bb0(%acc: f32, %elem: f32):
    %s = arith.addf %acc, %elem : f32
    skeleton.yield %s : f32
  }
  return
}

// -----

// ReduceOp: init not scalar → error

func.func @reduce_non_scalar_init(%input: memref<?xf32>, %init: memref<?xf32>) {
  %in_t = bufferization.to_tensor %input : memref<?xf32> to tensor<?xf32>
  %init_t = bufferization.to_tensor %init : memref<?xf32> to tensor<?xf32>
  // expected-error@+1 {{reduce init must be a scalar tensor (rank 0)}}
  %r = skeleton.reduce pure_fn = @my_sum
    ins(%in_t : tensor<?xf32>)
    outs(%init_t : tensor<?xf32>)
  return
}

// -----

// ReduceOp: init element type mismatch

func.func @reduce_elem_type_mismatch(%input: memref<?xf32>, %init: memref<i32>) {
  %in_t = bufferization.to_tensor %input : memref<?xf32> to tensor<?xf32>
  %init_t = bufferization.to_tensor %init : memref<i32> to tensor<i32>
  // expected-error@+1 {{reduce init element type must match input element type}}
  %r = skeleton.reduce pure_fn = @my_sum
    ins(%in_t : tensor<?xf32>)
    outs(%init_t : tensor<i32>)
  return
}

// -----

// VectorAddOp: rank not 1 → error

func.func @vecadd_bad_rank(%A: memref<8x8xf32>, %B: memref<8x8xf32>, %C: memref<8x8xf32>) {
  %A_t = bufferization.to_tensor %A : memref<8x8xf32> to tensor<8x8xf32>
  %B_t = bufferization.to_tensor %B : memref<8x8xf32> to tensor<8x8xf32>
  %C_t = bufferization.to_tensor %C : memref<8x8xf32> to tensor<8x8xf32>
  // expected-error@+1 {{expects all operands to be 1D tensors}}
  %r = skeleton.vector_add
    ins(%A_t, %B_t : tensor<8x8xf32>, tensor<8x8xf32>)
    outs(%C_t : tensor<8x8xf32>)
  return
}

// -----

// YieldOp: not inside MapOp or ReduceOp → error at parse time (via HasParent)

func.func @yield_outside_skeleton(%val: f32) {
  // expected-error@+1 {{expects parent op to be one of 'skeleton.map, skeleton.reduce'}}
  skeleton.yield %val : f32
  return
}

// -----

// MapOp: input rank differs from init/result rank

func.func @my_add(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @map_rank_mismatch(%A: memref<2x3xf32>, %B: memref<2x3xf32>, %C: memref<2xf32>) {
  %A_t = bufferization.to_tensor %A : memref<2x3xf32> to tensor<2x3xf32>
  %B_t = bufferization.to_tensor %B : memref<2x3xf32> to tensor<2x3xf32>
  %C_t = bufferization.to_tensor %C : memref<2xf32> to tensor<2xf32>
  // expected-error@+1 {{input 0 has rank 2 but init/result has rank 1}}
  %r = skeleton.map pure_fn = @my_add
    ins(%A_t, %B_t : tensor<2x3xf32>, tensor<2x3xf32>)
    outs(%C_t : tensor<2xf32>)
  return
}

// -----

// MapOp: static input dimension disagrees with init/result

func.func @my_add(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @map_shape_mismatch(%A: memref<8xf32>, %B: memref<8xf32>, %C: memref<16xf32>) {
  %A_t = bufferization.to_tensor %A : memref<8xf32> to tensor<8xf32>
  %B_t = bufferization.to_tensor %B : memref<8xf32> to tensor<8xf32>
  %C_t = bufferization.to_tensor %C : memref<16xf32> to tensor<16xf32>
  // expected-error@+1 {{input 0 dimension 0 has static size 8 but init/result has static size 16}}
  %r = skeleton.map pure_fn = @my_add
    ins(%A_t, %B_t : tensor<8xf32>, tensor<8xf32>)
    outs(%C_t : tensor<16xf32>)
  return
}

// -----

// ReduceOp: rank-0 input → error

func.func @my_sum(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @reduce_rank0_input(%input: memref<f32>, %init: memref<f32>) {
  %in_t = bufferization.to_tensor %input : memref<f32> to tensor<f32>
  %init_t = bufferization.to_tensor %init : memref<f32> to tensor<f32>
  // expected-error@+1 {{reduce input must be at least 1D, got rank 0}}
  %r = skeleton.reduce pure_fn = @my_sum
    ins(%in_t : tensor<f32>)
    outs(%init_t : tensor<f32>)
  return
}

// -----

// PreferenceAttr: invalid value → error at parse time

func.func @my_add(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

func.func @pref_invalid(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error@+1 {{expected 'CPU' or 'GPU' for preference attribute, got 'TPU'}}
  %r = skeleton.map preference = #skeleton.preference<"TPU"> pure_fn = @my_add
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>)
  return %r : tensor<8xf32>
}
