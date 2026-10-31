// REQUIRES: mlir-skeleton-driver
// RUN: mlir-skeleton-driver %s -entry main \
// RUN:   --shared-libs=%mlir_c_runner_utils,%mlir_runner_utils | FileCheck %s

// A program with no operator units is still a valid host program: the driver
// must not require the operator table to be non-empty.

func.func private @printMemrefF32(memref<*xf32>)

func.func @main() {
  %c0 = arith.constant 0 : index
  %value = arith.constant 42.0 : f32
  %buffer = memref.alloc() : memref<1xf32>
  memref.store %value, %buffer[%c0] : memref<1xf32>
  %unranked = memref.cast %buffer : memref<1xf32> to memref<*xf32>
  call @printMemrefF32(%unranked) : (memref<*xf32>) -> ()
  memref.dealloc %buffer : memref<1xf32>
  return
}

// CHECK: Unranked Memref
// CHECK-SAME: rank = 1
// CHECK: [42]
