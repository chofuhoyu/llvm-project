// REQUIRES: mlir-skeleton-driver
// RUN: mlir-skeleton-driver %s -entry main --emit %t.prog \
// RUN:   --runtime-lib=%mlir_lib_dir/libMLIRSkeletonRuntime.a \
// RUN:   --link-arg=-L%mlir_lib_dir --link-arg=-Wl,-rpath,%mlir_lib_dir \
// RUN:   --link-arg=-lmlir_runner_utils --link-arg=-lmlir_c_runner_utils
// RUN: %t.prog | FileCheck %s
// RUN: mlir-skeleton-driver %t.prog -entry main --trace \
// RUN:   --shared-libs=%mlir_c_runner_utils,%mlir_runner_utils \
// RUN:   2>&1 >/dev/null | FileCheck %s --check-prefix=TRACE
// RUN: mlir-skeleton-driver %t.prog -entry main \
// RUN:   --shared-libs=%mlir_c_runner_utils,%mlir_runner_utils | FileCheck %s

// The emitted artifact runs on its own (CPU implementations only) and is also
// a valid driver input: the driver reads the embedded skeleton IR, recompiles
// the operator on demand, and executes.

// TRACE: [skeleton-rt] op=0 declare pref=CPU
// TRACE: [skeleton-rt] op=0 compile device=CPU
// TRACE: [skeleton-rt] op=0 pref=CPU probe=CPU choose=CPU

func.func @main() {
  %A = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %B = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
  %C = arith.constant dense<0.0> : tensor<2x2xf32>

  %result = linalg.matmul {skeleton.preference = #skeleton.preference<"CPU">}
      ins(%A, %B : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%C : tensor<2x2xf32>) -> tensor<2x2xf32>

  %unranked = tensor.cast %result : tensor<2x2xf32> to tensor<*xf32>
  call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()

  // CHECK: Unranked Memref
  // CHECK-SAME: rank = 2
  // CHECK-SAME: sizes = [2, 2]
  // CHECK: {{[[]}}19,   22]
  // CHECK: [43,   50]

  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
