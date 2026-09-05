// End-to-end run of the manual path: the operators (pure fn my_add + host
// wrapper vec_add calling skeleton_map) are defined in real C++ and compiled
// through clang(-fclangir) -> cir-opt(-cir-call-to-skeleton); a hand-written
// MLIR @main (cpp-vector-add-e2e.driver) allocates the memref inputs, calls
// vec_add and prints the returned tensor. Expected C = A + B = [6, 8, 10, 12].
//
// The awk filter peels the `module {...}` wrapper cir-opt prints off the
// C++-compiled functions so that `cat` can join them with the driver into a
// single implicit module for the skeleton lowering pipeline.
//
// RUN: clang -cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -x c++ -I %mlir_src_root/include %S/cpp-vector-add-e2e.cpp -emit-cir -o %t.cir
// RUN: cir-opt %t.cir -cir-call-to-skeleton -o %t.skel.mlir
// RUN: awk 'NR==1{next} {l[++n]=$0} END{while(n>0 && l[n]=="")n--; if(n>0 && l[n]=="}")n--; for(i=1;i<=n;i++)print l[i]}' %t.skel.mlir > %t.funcs.mlir
// RUN: cat %t.funcs.mlir %S/cpp-vector-add-e2e.driver > %t.all.mlir
// RUN: mlir-opt %t.all.mlir -skeleton-to-linalg -skeleton-preference-partition -skeleton-target-lower \
// RUN:   -convert-scf-to-cf -expand-strided-metadata -convert-arith-to-llvm \
// RUN:   -finalize-memref-to-llvm -convert-func-to-llvm -convert-cf-to-llvm \
// RUN:   -reconcile-unrealized-casts -o - | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils | \
// RUN: FileCheck %s

// CHECK: Unranked Memref
// CHECK-SAME: rank = 1
// CHECK-SAME: sizes = [4]
// CHECK: [6,{{ +}}8,{{ +}}10,{{ +}}12]
