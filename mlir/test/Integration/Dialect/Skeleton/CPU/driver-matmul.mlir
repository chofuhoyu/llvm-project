// RUN: mlir-skeleton-driver %s -entry main --shared-libs=%mlir_c_runner_utils,%mlir_runner_utils | FileCheck %s

// RUN: mlir-skeleton-driver %s -entry main --trace --shared-libs=%mlir_c_runner_utils,%mlir_runner_utils 2>&1 >/dev/null | FileCheck %s --check-prefix=TRACE
// RUN: env SKELETON_TRACE=1 mlir-skeleton-driver %s -entry main --shared-libs=%mlir_c_runner_utils,%mlir_runner_utils 2>&1 >/dev/null | FileCheck %s --check-prefix=ENV

// The driver outlines the matmul into an operator unit, compiles it on demand
// for the CPU when the runtime resolves it, and executes the host.

// TRACE: [skeleton-rt] op=0 declare pref=CPU
// TRACE: [skeleton-rt] op=0 compile device=CPU
// TRACE: [skeleton-rt] op=0 pref=CPU probe=CPU choose=CPU

// The runtime honours SKELETON_TRACE even though the driver did not pass
// --trace; the driver must not force tracing off.
// ENV: [skeleton-rt] op=0 declare pref=CPU
// ENV: [skeleton-rt] op=0 compile device=CPU
// ENV: [skeleton-rt] op=0 pref=CPU probe=CPU choose=CPU

// A = [[1, 2], [3, 4]]   (2x2)
// B = [[5, 6], [7, 8]]   (2x2)
// Expected C = A * B = [[19, 22], [43, 50]]

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
