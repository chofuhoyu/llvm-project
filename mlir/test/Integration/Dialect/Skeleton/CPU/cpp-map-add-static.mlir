// End-to-end run of the manual path with a compile-time extent: the operators
// (pure fn my_add + host wrapper vec_add) are defined in real C++ and compiled
// through clang(-fclangir) -> cir-opt(-cir-call-to-skeleton). The host's
// parameters are array references, so the pass gives the rewritten function
// memref<4xf32> arguments and a tensor<4xf32> result; the hand-written MLIR
// @main (cpp-map-add-static.driver) allocates buffers of that shape, calls
// vec_add and prints the returned tensor. Expected C = A + B = [6, 8, 10, 12].
//
// skeleton-peel-module.sh peels the `module {...}` wrapper cir-opt prints off
// the C++-compiled functions so that `cat` can join them with the driver into
// a single implicit module for the skeleton lowering pipeline. The peel
// validates cir-opt's output shape and fails loudly instead of silently
// splicing a corrupt module.
//
// RUN: clang -cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -x c++ -I %mlir_src_root/include %S/cpp-map-add-static.cpp -emit-cir -o %t.cir
// RUN: cir-opt %t.cir -cir-call-to-skeleton -o %t.skel.mlir
// RUN: sh %S/skeleton-peel-module.sh %t.skel.mlir > %t.funcs.mlir
// RUN: cat %t.funcs.mlir %S/cpp-map-add-static.driver > %t.all.mlir
// RUN: mlir-skeleton-driver %t.all.mlir -entry main \
// RUN:   --shared-libs=%mlir_c_runner_utils,%mlir_runner_utils | FileCheck %s

// CHECK: Unranked Memref
// CHECK-SAME: rank = 1
// CHECK-SAME: sizes = [4]
// CHECK: [6,{{ +}}8,{{ +}}10,{{ +}}12]
