// REQUIRES: mlir-skeleton-driver
// End-to-end run of the manual path with a compile-time extent, lowering to
// the GPU: the operators are defined in real C++ (cpp-map-add-static.cpp, whose
// host carries the "GPU" preference) and compiled through clang(-fclangir) ->
// cir-opt(-cir-call-to-skeleton), then lowered with the NVVM tail instead of
// the CPU one. The hand-written MLIR @main is the CPU test's driver, since the
// host's signature -- static memrefs in, tensor out -- is the same either way.
// Expected C = A + B = [6, 8, 10, 12].
//
// RUN: clang -cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -x c++ -I %mlir_src_root/include %S/cpp-map-add-static.cpp -emit-cir -o %t.cir
// RUN: cir-opt %t.cir -cir-call-to-skeleton -o %t.skel.mlir
// RUN: sh %S/../CPU/skeleton-peel-module.sh %t.skel.mlir > %t.funcs.mlir
// RUN: cat %t.funcs.mlir %S/../CPU/cpp-map-add-static.driver > %t.all.mlir
// RUN: mlir-skeleton-driver %t.all.mlir -entry main \
// RUN:   --shared-libs=%mlir_cuda_runtime,%mlir_runner_utils,%mlir_c_runner_utils | FileCheck %s

// CHECK: Unranked Memref
// CHECK-SAME: rank = 1
// CHECK-SAME: sizes = [4]
// CHECK: [6,{{ +}}8,{{ +}}10,{{ +}}12]
