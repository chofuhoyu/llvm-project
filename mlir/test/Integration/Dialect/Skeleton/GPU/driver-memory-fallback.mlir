// REQUIRES: mlir-skeleton-driver
// RUN: mlir-skeleton-driver %s -entry main --trace \
// RUN:   --shared-libs=%mlir_c_runner_utils,%mlir_runner_utils 2>&1 >/dev/null | \
// RUN:   FileCheck %s --check-prefix=TRACE
// RUN: mlir-skeleton-driver %s -entry main \
// RUN:   --shared-libs=%mlir_c_runner_utils,%mlir_runner_utils | FileCheck %s

// A GPU-preferred operator whose operands do not fit in the free device memory
// falls back to the CPU instead of failing inside the allocation. The device is
// forced available and its free-byte count is forced below the operand size,
// so the decision does not depend on the machine's real memory.

// The matmul moves 3 * 2 * 2 * 4 = 48 bytes of operands; 1 byte of free device
// memory is below that, so the GPU placement is rejected before any compilation.

// TRACE: [skeleton-rt] op=0 declare pref=GPU
// TRACE: [skeleton-rt] device=GPU available=1 (forced)
// TRACE: [skeleton-rt] device=GPU free_bytes=1 (forced)
// TRACE: [skeleton-rt] op=0 compile device=CPU
// TRACE: [skeleton-rt] op=0 pref=GPU probe=GPU free=0MB choose=CPU reason=insufficient-memory

func.func private @skeleton_rt_set_device_available(i32, i32)
func.func private @skeleton_rt_set_free_bytes(i32, i64)

// A = [[1, 2], [3, 4]]   (2x2)
// B = [[5, 6], [7, 8]]   (2x2)
// Expected C = A * B = [[19, 22], [43, 50]]

func.func @main() {
  %gpu = arith.constant 1 : i32
  %oneByte = arith.constant 1 : i64
  call @skeleton_rt_set_device_available(%gpu, %gpu) : (i32, i32) -> ()
  call @skeleton_rt_set_free_bytes(%gpu, %oneByte) : (i32, i64) -> ()

  %A = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %B = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
  %C = arith.constant dense<0.0> : tensor<2x2xf32>

  %result = linalg.matmul {skeleton.preference = #skeleton.preference<"GPU">}
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
