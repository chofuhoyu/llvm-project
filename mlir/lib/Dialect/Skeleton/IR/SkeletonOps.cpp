//===- SkeletonOps.cpp - Skeleton dialect ops implementation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Skeleton/IR/SkeletonOps.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonDialect.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::skeleton;

#define GET_OP_CLASSES
#include "mlir/Dialect/Skeleton/IR/SkeletonOps.cpp.inc"

//===----------------------------------------------------------------------===//
// CustomMatmulOp
//===----------------------------------------------------------------------===//

LogicalResult CustomMatmulOp::verify() {
  // Verify that lhs, rhs, and output have compatible shapes for matmul.
  // lhs: M x K, rhs: K x N, output: M x N
  auto lhsType = cast<RankedTensorType>(getLhs().getType());
  auto rhsType = cast<RankedTensorType>(getRhs().getType());
  auto outputType = cast<RankedTensorType>(getOutput().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());

  unsigned lhsRank = lhsType.getRank();
  unsigned rhsRank = rhsType.getRank();
  unsigned outputRank = outputType.getRank();
  unsigned resultRank = resultType.getRank();

  if (lhsRank != 2 || rhsRank != 2 || outputRank != 2 || resultRank != 2)
    return emitOpError("expects all operands to be 2D tensors");

  if (outputType != resultType)
    return emitOpError("expects output type to match result type");

  // Check element type compatibility across all operands and result.
  auto lhsEltType = lhsType.getElementType();
  if (rhsType.getElementType() != lhsEltType ||
      outputType.getElementType() != lhsEltType ||
      resultType.getElementType() != lhsEltType)
    return emitOpError("expects all operands and result to have the same "
                       "element type");

  // Check K dimension compatibility: lhs dim 1 == rhs dim 0
  if (!lhsType.isDynamicDim(1) && !rhsType.isDynamicDim(0))
    if (lhsType.getDimSize(1) != rhsType.getDimSize(0))
      return emitOpError("K dimension mismatch between lhs and rhs");

  // Check M dimension compatibility: lhs dim 0 == output dim 0
  if (!lhsType.isDynamicDim(0) && !outputType.isDynamicDim(0))
    if (lhsType.getDimSize(0) != outputType.getDimSize(0))
      return emitOpError("M dimension mismatch between lhs and output");

  // Check N dimension compatibility: rhs dim 1 == output dim 1
  if (!rhsType.isDynamicDim(1) && !outputType.isDynamicDim(1))
    if (rhsType.getDimSize(1) != outputType.getDimSize(1))
      return emitOpError("N dimension mismatch between rhs and output");

  return success();
}

//===----------------------------------------------------------------------===//
// VectorAddOp
//===----------------------------------------------------------------------===//

LogicalResult VectorAddOp::verify() {
  // Verify that lhs, rhs, output are 1D tensors (vectors).
  auto lhsType = cast<RankedTensorType>(getLhs().getType());
  auto rhsType = cast<RankedTensorType>(getRhs().getType());
  auto outputType = cast<RankedTensorType>(getOutput().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());

  unsigned lhsRank = lhsType.getRank();
  unsigned rhsRank = rhsType.getRank();
  unsigned outputRank = outputType.getRank();
  unsigned resultRank = resultType.getRank();

  if (lhsRank != 1 || rhsRank != 1 || outputRank != 1 || resultRank != 1)
    return emitOpError("expects all operands to be 1D tensors (vectors)");

  if (outputType != resultType)
    return emitOpError("expects output type to match result type");

  // Check element type compatibility.
  auto lhsEltType = lhsType.getElementType();
  if (rhsType.getElementType() != lhsEltType ||
      outputType.getElementType() != lhsEltType ||
      resultType.getElementType() != lhsEltType)
    return emitOpError("expects all operands and result to have the same "
                       "element type");

  // Check length compatibility: all vectors must have the same size.
  if (!lhsType.isDynamicDim(0) && !rhsType.isDynamicDim(0))
    if (lhsType.getDimSize(0) != rhsType.getDimSize(0))
      return emitOpError("vector length mismatch between lhs and rhs");

  if (!lhsType.isDynamicDim(0) && !outputType.isDynamicDim(0))
    if (lhsType.getDimSize(0) != outputType.getDimSize(0))
      return emitOpError("vector length mismatch between lhs and output");

  return success();
}
