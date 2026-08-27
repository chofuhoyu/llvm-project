//===- SkeletonOps.cpp - Skeleton dialect ops implementation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Skeleton/IR/SkeletonOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace mlir::skeleton;

#define GET_OP_CLASSES
#include "mlir/Dialect/Skeleton/IR/SkeletonOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Common pure_fn verification helper
//===----------------------------------------------------------------------===//

/// Verify that a FlatSymbolRefAttr refers to a func.func with a signature
/// compatible with the given expected parameter types and result type.
static LogicalResult verifyPureFnSignature(Operation *op,
    FlatSymbolRefAttr pureFn,
    ArrayRef<Type> expectedParamTypes,
    Type expectedResultType) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return op->emitOpError("expected a parent ModuleOp to resolve pure_fn");

  auto fn = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, pureFn);
  if (!fn)
    return op->emitOpError("pure_fn @")
           << pureFn.getValue() << " not found in symbol table";

  auto fnType = fn.getFunctionType();

  // Check parameter count.
  if (fnType.getNumInputs() != expectedParamTypes.size())
    return op->emitOpError("pure_fn @")
           << pureFn.getValue()
           << " expects " << fnType.getNumInputs()
           << " parameters but skeleton op provides "
           << expectedParamTypes.size() << " inputs";

  // Check each parameter type against the expected element type.
  for (unsigned i = 0; i < expectedParamTypes.size(); ++i) {
    if (fnType.getInput(i) != expectedParamTypes[i])
      return op->emitOpError("pure_fn parameter ")
             << i << " type mismatch: expected "
             << expectedParamTypes[i] << " but got "
             << fnType.getInput(i);
  }

  // Check result type.
  if (fnType.getNumResults() != 1)
    return op->emitOpError("pure_fn @")
           << pureFn.getValue()
           << " must have exactly one result, got "
           << fnType.getNumResults();

  if (fnType.getResult(0) != expectedResultType)
    return op->emitOpError("pure_fn result type mismatch: expected ")
           << expectedResultType << " but got "
           << fnType.getResult(0);

  return success();
}

//===----------------------------------------------------------------------===//
// MapOp
//===----------------------------------------------------------------------===//

LogicalResult MapOp::verify() {
  bool hasPureFn = getPureFnAttr() != nullptr;
  bool hasBody = !getBody().empty();

  // Exactly one of pure_fn or body must be present.
  if (hasPureFn == hasBody)
    return emitOpError("requires exactly one of 'pure_fn' or a body region");

  auto outputType = cast<RankedTensorType>(getOutput().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());

  // Output and result must have the same shape.
  if (outputType != resultType)
    return emitOpError("output type must match result type");

  auto outputEltType = outputType.getElementType();

  // Verify pure_fn signature if present.
  if (hasPureFn) {
    // Collect expected parameter types from input tensor element types.
    SmallVector<Type> expectedParamTypes;
    for (auto input : getInputs()) {
      auto inputTensorType = cast<RankedTensorType>(input.getType());
      expectedParamTypes.push_back(inputTensorType.getElementType());
    }
    if (failed(verifyPureFnSignature(getOperation(), getPureFnAttr(),
                                     expectedParamTypes, outputEltType)))
      return failure();
  }

  // Verify body region if present.
  if (hasBody) {
    auto &body = getBody();
    if (!body.hasOneBlock())
      return emitOpError("body region must have exactly one block");

    Block &entry = body.front();
    unsigned numInputs = getInputs().size();

    // Block arguments: one per input element type + one for output element
    // type.
    unsigned expectedBlockArgs = numInputs + 1;
    if (entry.getNumArguments() != expectedBlockArgs)
      return emitOpError("body block expects ")
             << expectedBlockArgs << " arguments ("
             << numInputs << " input element types + 1 output element type) "
             << "but has " << entry.getNumArguments();

    // Check block arg types match input element types.
    for (unsigned i = 0; i < numInputs; ++i) {
      auto inputEltType =
          cast<RankedTensorType>(getInputs()[i].getType()).getElementType();
      if (entry.getArgument(i).getType() != inputEltType)
        return emitOpError("body block argument ")
               << i << " type mismatch";
    }
    // Check last block arg matches output element type.
    if (entry.getArgument(numInputs).getType() != outputEltType)
      return emitOpError("body block output initializer argument type "
                         "mismatch");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceOp::verify() {
  bool hasPureFn = getPureFnAttr() != nullptr;
  bool hasBody = !getBody().empty();

  // Exactly one of pure_fn or body must be present.
  if (hasPureFn == hasBody)
    return emitOpError("requires exactly one of 'pure_fn' or a body region");

  auto inputType = cast<RankedTensorType>(getInput().getType());
  auto outputType = cast<RankedTensorType>(getOutput().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());

  // Output and result must match.
  if (outputType != resultType)
    return emitOpError("output type must match result type");

  auto eltType = inputType.getElementType();

  // Input must be at least 1D. Output must be 0D (scalar tensor) or same
  // element type with reduced rank.
  if (outputType.getRank() != 0)
    return emitOpError("reduce output must be a scalar tensor (rank 0), got "
                       "rank ")
           << outputType.getRank();

  if (outputType.getElementType() != eltType)
    return emitOpError("reduce output element type must match input element "
                       "type");

  // Verify pure_fn signature if present.
  if (hasPureFn) {
    // Reduce pure_fn is binary: (T, T) -> T.
    SmallVector<Type> expectedParamTypes = {eltType, eltType};
    if (failed(verifyPureFnSignature(getOperation(), getPureFnAttr(),
                                     expectedParamTypes, eltType)))
      return failure();
  }

  // Verify body region if present.
  if (hasBody) {
    auto &body = getBody();
    if (!body.hasOneBlock())
      return emitOpError("body region must have exactly one block");

    Block &entry = body.front();
    // Reduce body: accumulator + element → (not directly modeled here; the
    // body is an alternative to pure_fn, so it accepts the same type
    // signature).
    if (entry.getNumArguments() != 2)
      return emitOpError("reduce body block expects 2 arguments "
                         "(accumulator, element) but has ")
             << entry.getNumArguments();

    if (entry.getArgument(0).getType() != eltType ||
        entry.getArgument(1).getType() != eltType)
      return emitOpError("reduce body block arguments must have element "
                         "type ")
             << eltType;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// VectorAddOp
//===----------------------------------------------------------------------===//

LogicalResult VectorAddOp::verify() {
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

  auto lhsEltType = lhsType.getElementType();
  if (rhsType.getElementType() != lhsEltType ||
      outputType.getElementType() != lhsEltType ||
      resultType.getElementType() != lhsEltType)
    return emitOpError("expects all operands and result to have the same "
                       "element type");

  // Check length compatibility.
  if (!lhsType.isDynamicDim(0) && !rhsType.isDynamicDim(0))
    if (lhsType.getDimSize(0) != rhsType.getDimSize(0))
      return emitOpError("vector length mismatch between lhs and rhs");

  if (!lhsType.isDynamicDim(0) && !outputType.isDynamicDim(0))
    if (lhsType.getDimSize(0) != outputType.getDimSize(0))
      return emitOpError("vector length mismatch between lhs and output");

  return success();
}
