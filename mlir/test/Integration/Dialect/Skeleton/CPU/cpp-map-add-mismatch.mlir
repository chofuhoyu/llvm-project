// End-to-end run of the manual path where the two inputs have different
// lengths: the operators are the plain-pointer ones from cpp-map-add.cpp, whose
// memrefs are dynamically shaped, so nothing in the type says how long the
// arrays are. Two runtime lengths are only known to agree at run time, and a
// shorter second input would be read out of bounds by the map, so the rewritten
// function carries a check of them. This driver (lengths 4 and 8) is what trips
// it.
//
// RUN: clang -cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -x c++ -I %mlir_src_root/include %S/cpp-map-add.cpp -emit-cir -o %t.cir
// RUN: cir-opt %t.cir -cir-call-to-skeleton -o %t.skel.mlir
// RUN: sh %S/skeleton-peel-module.sh %t.skel.mlir > %t.funcs.mlir
// RUN: cat %t.funcs.mlir %S/cpp-map-add-mismatch.driver > %t.all.mlir
// RUN: mlir-opt %t.all.mlir -skeleton-to-linalg -o - | FileCheck %s --check-prefix=SKELETON
// RUN: not --crash mlir-skeleton-driver %t.all.mlir -entry main --dump-host-ir \
// RUN:   --shared-libs=%mlir_c_runner_utils,%mlir_runner_utils > /dev/null 2> %t.host.mlir
// RUN: FileCheck %s --check-prefix=LLVM < %t.host.mlir
// RUN: not --crash mlir-skeleton-driver %t.all.mlir -entry main \
// RUN:   --shared-libs=%mlir_c_runner_utils,%mlir_runner_utils > %t.out 2>&1

// The check rides in the host function, next to the op it guards ...
//
// SKELETON: func.func @vec_add
// SKELETON: cf.assert
// SKELETON: linalg.generic

// ... it survives the lowering down to the abort it stands for ...
//
// LLVM: @assert_msg
// LLVM: llvm.call @abort

// ... and running with two different lengths is stopped by a signal instead of
// producing a result from them. The message it carries goes to stdout, which
// the abort leaves unflushed, so only the exit is observable from here.
