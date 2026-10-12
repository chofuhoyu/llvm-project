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
#include "mlir/IR/BuiltinTypes.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Transforms/CirSkeletonAnnotations.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
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

  /// The host parameter each data operand of the call was passed from, in call
  /// order (operand 0 is the pure function reference and has no entry). The
  /// call's argument order need not match the host's parameter order, so this
  /// is recorded rather than assumed positional. Filled by
  /// resolveDataOperands, which also checks that the parameters agree.
  llvm::SmallVector<unsigned> dataOperandHostParams;

  /// The memref the data operands denote, which the rewritten function's result
  /// tensor is shaped from. Filled by resolveDataOperands.
  mlir::MemRefType dataMemRef;
};

/// The memref a skeleton host parameter of CIR type \p paramTy denotes, or
/// null when the type is not a data array. A pointer to a constant array
/// carries a compile-time extent per dimension, so `float (&a)[4]` denotes
/// memref<4xf32> and `float (&a)[2][4]` denotes memref<2x4xf32>; a
/// skeleton_detail::span carries a runtime length in its size member and a
/// bare pointer carries no extent at all, so both denote a one-dimensional
/// memref with a dynamic extent. SkeletonOps.h is where a host writes these
/// forms.
mlir::FailureOr<mlir::MemRefType> hostParamToMemRef(mlir::Type paramTy);

/// Resolve the skeleton call's data operands against the host's parameters:
/// record which parameter each operand came from and check that the parameters
/// denote one and the same type. The check lives here rather than in the
/// skeleton.map verifier because that verifier only rejects a static dimension
/// disagreeing with another static one, so a static extent against a dynamic
/// one would pass it and read out of bounds later. Reports on the call and
/// fails when an operand is not backed by a host data array, and when the
/// operands' types disagree.
mlir::LogicalResult resolveDataOperands(SkeletonCallInfo &info);

/// Emit a runtime length check for \p inputTensors: on each dynamic dimension
/// of \p resultTy, every input must have the same extent as the first one. A
/// static extent is already checked against the type by resolveDataOperands,
/// but a dynamic one is only known at run time, and the skeleton op reads every
/// input at the indices the first one bounds, so a shorter input would be read
/// out of bounds. Emits nothing for a fully static shape or a single input.
void emitExtentCheck(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::RankedTensorType resultTy,
                     llvm::ArrayRef<mlir::Value> inputTensors);

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

/// The skeleton op's output tensor type for an input of shape \p inputShape:
/// map is element-wise, so its output keeps that input's shape at any rank;
/// reduce collapses its input to a rank-0 scalar. The element type equals the
/// input's (map/reduce verifiers require output elt == input elt).
mlir::RankedTensorType outputTensorType(SkeletonOpType opType,
                                        llvm::ArrayRef<int64_t> inputShape,
                                        mlir::Type eltTy);

/// Create a tensor.empty of \p resultTy, filling its dynamic extents from the
/// corresponding dimension of \p firstInputTensor: a skeleton op's output
/// shares the shape of the input it is created from.
mlir::Value createEmptyOutput(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::RankedTensorType resultTy,
                              mlir::Value firstInputTensor);

/// Rewrite the cir.func that hosts a skeleton call to a func.func whose
/// parameters are memrefs and whose result is the skeleton op's output tensor.
/// Requires resolveDataOperands to have run on \p info.
mlir::func::FuncOp rewriteToStandardFunc(SkeletonCallInfo &info,
                                         mlir::OpBuilder &rewriter);

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
