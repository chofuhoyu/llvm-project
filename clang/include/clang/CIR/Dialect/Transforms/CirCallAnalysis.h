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
// declarations the user marked with annotations, and — for a given
// skeleton-helper call — which pure function its first argument refers to.
// The skeleton annotation semantics (the enum types and their typed readers)
// live in CirSkeletonAnnotations.h.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRCALLANALYSIS_H
#define LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRCALLANALYSIS_H

#include "clang/CIR/Dialect/Transforms/CirSkeletonAnnotations.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class BlockArgument;
class DominanceInfo;
class ModuleOp;
class Value;
} // namespace mlir

namespace cir {

class CallOp;
class FuncOp;

/// Collect function names marked with the "skeleton.pure" annotation.
llvm::DenseSet<llvm::StringRef> collectPureFunctions(mlir::ModuleOp module);

/// Collect skeleton op declarations: external functions carrying the
/// "skeleton.op" annotation. Returns a map from function name to the validated
/// operator type; an annotation that does not name a supported operator is
/// reported on its function and the whole result fails.
mlir::FailureOr<llvm::DenseMap<llvm::StringRef, SkeletonOpType>>
collectSkeletonOpDecls(mlir::ModuleOp module);

/// Extract a FlatSymbolRefAttr from a call argument that represents a
/// function pointer (e.g. via cir.get_global @some_fn).
mlir::FlatSymbolRefAttr
extractPureFnRef(cir::CallOp callOp, unsigned argIdx,
                 const llvm::DenseSet<llvm::StringRef> &pureFns,
                 mlir::DominanceInfo &domInfo);

/// Resolve a value that should name one of the host's data parameters to the
/// host entry block argument it was copied from. Clang's codegen stages every
/// function parameter through an alloca (`store` at entry, `load` before
/// use), so a call operand is often a `cir.load` of a parameter slot rather
/// than the entry block argument itself. This follows that
/// `store -> alloca -> load` form; anything that is not (derived from) a
/// direct parameter fails.
mlir::FailureOr<mlir::BlockArgument> resolveToHostArg(mlir::Value operand,
                                                      cir::FuncOp host);

} // namespace cir

#endif // LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRCALLANALYSIS_H
