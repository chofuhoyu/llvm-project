//===- CirCallAnalysis.h - analyze CIR for the manual path -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Phase-1 analysis for cir-call-to-skeleton: read the CIR input and answer the
// questions the rewriting phase needs. What pure functions and skeleton-op
// declarations the user marked with annotations, which preference a host
// function carries, and — for a given skeleton-helper call — which pure
// function its first argument refers to.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRCALLANALYSIS_H
#define LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRCALLANALYSIS_H

#include <string>

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class DominanceInfo;
class ModuleOp;
} // namespace mlir

namespace cir {

class CallOp;
class FuncOp;

/// Collect function names marked with the "skeleton.pure" annotation.
llvm::DenseSet<llvm::StringRef> collectPureFunctions(mlir::ModuleOp module);

/// Collect skeleton op declarations: external functions carrying the
/// "skeleton.op" annotation. Returns map: function name → op type
/// ("map", "reduce"), taken from the annotation's first argument.
llvm::DenseMap<llvm::StringRef, llvm::StringRef>
collectSkeletonOpDecls(mlir::ModuleOp module);

/// Extract the preference string from the "skeleton.region" annotation on a
/// cir.func. Falls back to "CPU" when the annotation is absent.
std::string extractPreference(cir::FuncOp func);

/// Extract a FlatSymbolRefAttr from a call argument that represents a
/// function pointer (e.g. via cir.get_global @some_fn).
mlir::FlatSymbolRefAttr
extractPureFnRef(cir::CallOp callOp, unsigned argIdx,
                 const llvm::DenseSet<llvm::StringRef> &pureFns,
                 mlir::DominanceInfo &domInfo);

} // namespace cir

#endif // LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRCALLANALYSIS_H
