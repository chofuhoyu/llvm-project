// REQUIRES: mlir-skeleton-driver
// End-to-end run of the semi-automatic path: a single annotated element-wise
// for loop is compiled through clang(-fclangir) -> cir-opt(-cir-loop-to-
// skeleton), which turns it into a pure func.func + a host func.func carrying a
// skeleton.map; a hand-written MLIR @main (loop-map-add.driver) allocates the
// memref inputs, calls annotated_for and prints the returned tensor. Expected
// C = A + B = [6, 8, 10, 12].
//
// skeleton-peel-module.sh peels the `module {...}` wrapper cir-opt prints off
// the C++-compiled function so that `cat` can join it with the driver into a
// single implicit module for the skeleton lowering pipeline.
//
// RUN: clang -cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -x c++ %S/loop-map-add.cpp -emit-cir -o %t.cir
// RUN: cir-opt %t.cir -cir-loop-to-skeleton -o %t.skel.mlir
// RUN: sh %S/skeleton-peel-module.sh %t.skel.mlir > %t.funcs.mlir
// RUN: cat %t.funcs.mlir %S/loop-map-add.driver > %t.all.mlir
// RUN: mlir-skeleton-driver %t.all.mlir -entry main \
// RUN:   --shared-libs=%mlir_c_runner_utils,%mlir_runner_utils | FileCheck %s

// CHECK: Unranked Memref
// CHECK-SAME: rank = 1
// CHECK-SAME: sizes = [4]
// CHECK: [6,{{ +}}8,{{ +}}10,{{ +}}12]
