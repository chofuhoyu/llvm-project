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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Transforms/CirSkeletonAnnotations.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class Location;
class ModuleOp;
class OpBuilder;
class RankedTensorType;
class Type;
class Value;
} // namespace mlir

namespace cir {

/// A work item extracted from the original cir.func before rewriting.
struct SkeletonCallInfo {
  cir::FuncOp cirFunc;
  cir::CallOp callOp;
  SkeletonOpType opType;
  llvm::StringRef pureFnName; // the referenced pure function
  SkeletonPreference preference;
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

/// Create bufferization.to_tensor from a memref. Shared host-body helper: used
/// by the manual path's lowerMapCall/lowerReduceCall and by the semi-automatic
/// loop path's host rewriting.
mlir::Value createToTensor(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value memref);

/// The skeleton op's output tensor type determined by the op's semantics: map
/// is element-wise, so its output is 1-D with a dynamic extent matching the
/// input; reduce collapses its input to a rank-0 scalar. The element type
/// equals the input's (map/reduce verifiers require output elt == input elt).
mlir::RankedTensorType outputTensorType(SkeletonOpType opType,
                                        mlir::Type eltTy);

/// Create a tensor.empty whose type matches \p newFunc's result tensor type
/// (the skeleton op's output). Dynamic extents in the result shape are filled
/// from the corresponding dimension of \p firstInputTensor, since map/reduce
/// outputs share the input's shape.
mlir::Value createEmptyOutput(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::func::FuncOp newFunc,
                              mlir::Value firstInputTensor);

/// Rewrite the cir.func that hosts a skeleton call to a func.func whose
/// parameters are memrefs and whose result is the skeleton op's output tensor.
mlir::func::FuncOp rewriteToStandardFunc(cir::FuncOp cirFunc,
                                         mlir::OpBuilder &rewriter,
                                         SkeletonOpType opType);

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
