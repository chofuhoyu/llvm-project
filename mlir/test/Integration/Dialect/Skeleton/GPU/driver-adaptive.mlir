// REQUIRES: mlir-skeleton-driver
// RUN: mlir-skeleton-driver %s -entry main --trace \
// RUN:   --shared-libs=%mlir_cuda_runtime,%mlir_runner_utils,%mlir_c_runner_utils \
// RUN:   2>&1 >/dev/null | FileCheck %s --check-prefix=TRACE
// RUN: mlir-skeleton-driver %s -entry main \
// RUN:   --shared-libs=%mlir_cuda_runtime,%mlir_runner_utils,%mlir_c_runner_utils | FileCheck %s

// The same operator runs three times in one process. The first call takes the
// GPU because the preference is available; the second sees the GPU marked
// unavailable and falls back to the CPU; the third sees it available again and
// reuses the GPU implementation. Both the decision and the earlier JIT engine
// must survive the round trip.

// TRACE: [skeleton-rt] op=0 declare pref=GPU
// TRACE: [skeleton-rt] op=0 compile device=GPU
// TRACE: [skeleton-rt] op=0 pref=GPU probe=GPU{{.*}}choose=GPU
// TRACE: [skeleton-rt] device=GPU available=0 (forced)
// TRACE: [skeleton-rt] op=0 compile device=CPU
// TRACE: [skeleton-rt] op=0 pref=GPU probe=CPU choose=CPU
// TRACE: [skeleton-rt] device=GPU available=1 (forced)
// TRACE: [skeleton-rt] op=0 pref=GPU probe=GPU{{.*}}choose=GPU

func.func private @skeleton_rt_set_device_available(i32, i32)

func.func private @printMemrefF32(tensor<*xf32>)

// A = [[1, 2], [3, 4]]   (2x2)
// B = [[5, 6], [7, 8]]   (2x2)
// Expected C = A * B = [[19, 22], [43, 50]]

func.func @host(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: tensor<2x2xf32>)
    -> tensor<2x2xf32> {
  %r = linalg.matmul {skeleton.preference = #skeleton.preference<"GPU">}
      ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%c : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}

func.func @main() {
  %A = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %B = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
  %C = arith.constant dense<0.0> : tensor<2x2xf32>
  %gpu = arith.constant 1 : i32
  %no = arith.constant 0 : i32
  %yes = arith.constant 1 : i32

  %first = call @host(%A, %B, %C) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  call @skeleton_rt_set_device_available(%gpu, %no) : (i32, i32) -> ()
  %second = call @host(%A, %B, %C) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  call @skeleton_rt_set_device_available(%gpu, %yes) : (i32, i32) -> ()
  %third = call @host(%A, %B, %C) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>

  %unranked = tensor.cast %third : tensor<2x2xf32> to tensor<*xf32>
  call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()

  // CHECK: Unranked Memref
  // CHECK-SAME: rank = 2
  // CHECK-SAME: sizes = [2, 2]
  // CHECK: {{[[]}}19,   22]
  // CHECK: [43,   50]

  return
}
