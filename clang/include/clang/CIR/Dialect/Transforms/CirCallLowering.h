//===- CirCallLowering.h - rewrite CIR calls to func/skeleton -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Phase-2 rewriting for cir-call-to-skeleton: turn the analyzed manual-path
// input into standard functions and skeleton ops. Pure functions marked with
// "skeleton.pure" become func.func (declaration-only ones stay declarations;
// ones carrying a body get it translated to arith by CirFuncToArith). A host
// cir.func that wraps one skeleton-helper call becomes a func.func whose body
// builds a skeleton.map / skeleton.reduce and returns its tensor result.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRCALL_LOWERING_H
#define LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRCALL_LOWERING_H

#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class ModuleOp;
class OpBuilder;
} // namespace mlir

namespace cir {

/// A work item extracted from the original cir.func before rewriting.
struct SkeletonCallInfo {
  cir::FuncOp cirFunc;
  cir::CallOp callOp;
  llvm::StringRef opType;     // "map" or "reduce"
  llvm::StringRef pureFnName; // the referenced pure function
  std::string preference;     // "CPU" or "GPU"
  // Number of data operands in the call. Return-value style: every data
  // operand is an input; the output is a fresh tensor returned from the
  // function, not a caller-supplied buffer.
  unsigned numInputs;
};

/// Convert pure functions from cir.func to func.func so that skeleton ops can
/// reference them via pure_fn. A declaration-only pure function becomes a
/// func.func declaration. One that carries a body has it translated into an
/// arith body by populateArithFuncBody, so SkeletonToLinalg can clone it. A
/// body that cannot be translated is reported and the whole pass fails rather
/// than silently dropping the function.
mlir::LogicalResult
convertPureFunctionsToFunc(mlir::ModuleOp module,
                           const llvm::DenseSet<llvm::StringRef> &pureFns);

/// Rewrite the cir.func that hosts a skeleton call to a func.func whose
/// parameters are memrefs and whose result is the skeleton op's output tensor.
mlir::func::FuncOp rewriteToStandardFunc(cir::FuncOp cirFunc,
                                         mlir::OpBuilder &rewriter,
                                         llvm::StringRef opType);

/// Process a map call: create a skeleton.map whose result is the function's
/// return value. Returns the skeleton op result (nullptr on failure).
mlir::Value lowerMapCall(SkeletonCallInfo &info, mlir::func::FuncOp newFunc,
                         mlir::OpBuilder &builder);

/// Process a reduce call: create a skeleton.reduce whose result (a rank-0
/// tensor) is the function's return value. Returns the skeleton op result
/// (nullptr on failure).
mlir::Value lowerReduceCall(SkeletonCallInfo &info, mlir::func::FuncOp newFunc,
                            mlir::OpBuilder &builder);

} // namespace cir

#endif // LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRCALL_LOWERING_H
