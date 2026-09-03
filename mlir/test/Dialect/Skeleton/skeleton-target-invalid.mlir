// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// TargetAttr: invalid device value "TPU" is rejected by the verifier.
// expected-error@+1 {{expected 'CPU' or 'GPU' for target attribute, got 'TPU'}}
func.func @invalid_target_tpu() attributes {skeleton.target = #skeleton.target<"TPU">} {
  return
}

// -----

// TargetAttr: case-sensitive — "gpu" is not accepted.
// expected-error@+1 {{expected 'CPU' or 'GPU' for target attribute, got 'gpu'}}
func.func @invalid_target_lower() attributes {skeleton.target = #skeleton.target<"gpu">} {
  return
}