//===- CirCallLowering.h - shape and ABI of a skeleton helper call -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The C++ boundary of one skeleton operator: a call to a skeleton helper
// (declared with the "skeleton.op" annotation) inside a host function marked
// with "skeleton.region". This file owns how a C++ data argument's type becomes
// a memref -- that is where an operator's shapes come from -- and the check
// that every data argument really is a parameter of the host.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRCALL_LOWERING_H
#define LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRCALL_LOWERING_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Transforms/CirSkeletonAnnotations.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class Location;
class ModuleOp;
class OpBuilder;
class Type;
class Value;
} // namespace mlir

namespace cir {

/// The members of skeleton_detail::span<T>: the data pointer and the length.
/// Their roles are part of the project's own type, so the pass reads them from
/// that declaration instead of guessing at the layout.
constexpr unsigned kSpanDataMember = 0;
constexpr unsigned kSpanSizeMember = 1;

/// True when \p ty is the skeleton_detail::span<T> the front-end header
/// declares. It is recognized by name together with both members' roles, so a
/// struct which merely looks similar is not taken for one.
bool isSkeletonSpan(mlir::Type ty);

/// One data argument of a skeleton helper call: the host parameter it came
/// from, and the memref that parameter denotes.
struct DataOperandRef {
  unsigned hostParam = 0;
  mlir::MemRefType type;
  /// The parameter is a span, so the operand's dynamic extent is the span's
  /// size member read at run time. False for an array reference (compile-time
  /// extents) and for a single-element destination.
  bool hasRuntimeExtent = false;
};

/// A skeleton helper call prepared for the split: which operator it is, the
/// pure function it applies, and where its data and scalar arguments come
/// from. The call's operand order is the unit's operand order.
struct SkeletonCallInfo {
  cir::FuncOp cirFunc;
  cir::CallOp callOp;
  SkeletonOpType opType;
  llvm::StringRef pureFnName;
  SkeletonPreference preference;

  /// The data arguments in call order. The first one is the destination whose
  /// shape is the operator's result shape.
  llvm::SmallVector<DataOperandRef> dataOperands;

  /// The call operand indices of the scalar arguments (init values and the
  /// like), in call order. Their values are passed to the unit as they are.
  llvm::SmallVector<unsigned> scalarOperands;
};

/// The memref the host parameter \p paramTy denotes, or failure when the type
/// is not a data argument. An array reference carries one compile-time extent
/// per dimension, so `float (&a)[2][4]` denotes memref<2x4xf32>; a
/// skeleton_detail::span carries its length at run time, so it denotes a
/// one-dimensional memref with a dynamic extent. A pointer to a single element
/// is a rank-0 memref and is only a destination: as an input it says nothing
/// about how many elements follow, which is what a span is for.
mlir::FailureOr<mlir::MemRefType> hostParamToMemRef(mlir::Type paramTy,
                                                    bool isDestination);

/// Resolve the call's arguments against the host's parameters and record where
/// each data argument comes from. \p info.dataOperands and
/// \p info.scalarOperands are filled in call order; the check that all data
/// arguments agree on the element type lives here, because the skeleton op's
/// verifier accepts a static extent against a dynamic one and a host mixing the
/// two would read out of bounds.
mlir::LogicalResult resolveDataOperands(SkeletonCallInfo &info);

/// Create bufferization.to_tensor from a memref: the tensor view the skeleton
/// operator reads (and, for a destination, writes through).
mlir::Value createToTensor(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value memref);

} // namespace cir

#endif // LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRCALL_LOWERING_H
