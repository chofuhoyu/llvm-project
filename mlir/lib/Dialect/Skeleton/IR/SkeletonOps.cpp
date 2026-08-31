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

// Common pure_fn verification helper

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

// MapOp

LogicalResult MapOp::verify() {
  bool hasPureFn = getPureFnAttr() != nullptr;
  bool hasBody = !getBody().empty();

  // Exactly one of pure_fn or body must be present.
  if (hasPureFn == hasBody)
    return emitOpError("requires exactly one of 'pure_fn' or a body region");

  auto initType = cast<RankedTensorType>(getInit().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());

  // Init and result must have the same shape.
  if (initType != resultType)
    return emitOpError("init type must match result type");

  auto initEltType = initType.getElementType();

  // Every input must have the same rank as init/result: this op is
  // element-wise, so inputs map element-for-element onto the output. A rank
  // mismatch would silently misalign ConvertMapOp's indexing maps, which are
  // built from the result rank.
  unsigned resultRank = resultType.getRank();
  for (unsigned i = 0, e = getInputs().size(); i < e; ++i) {
    auto inputType = cast<RankedTensorType>(getInputs()[i].getType());
    if (inputType.getRank() != resultRank)
      return emitOpError("input ")
             << i << " has rank " << inputType.getRank()
             << " but init/result has rank " << resultRank;

    // Static dimensions must agree where both are known.
    for (unsigned d = 0; d < resultRank; ++d) {
      int64_t inSize = inputType.getDimSize(d);
      int64_t resSize = resultType.getDimSize(d);
      if (!ShapedType::isDynamic(inSize) &&
          !ShapedType::isDynamic(resSize) && inSize != resSize)
        return emitOpError("input ")
               << i << " dimension " << d << " has static size " << inSize
               << " but init/result has static size " << resSize;
    }
  }

  // Verify pure_fn signature if present.
  if (hasPureFn) {
    // Collect expected parameter types from input tensor element types.
    SmallVector<Type> expectedParamTypes;
    for (auto input : getInputs()) {
      auto inputTensorType = cast<RankedTensorType>(input.getType());
      expectedParamTypes.push_back(inputTensorType.getElementType());
    }
    if (failed(verifyPureFnSignature(getOperation(), getPureFnAttr(),
                                     expectedParamTypes, initEltType)))
      return failure();
  }

  // Verify body region if present.
  if (hasBody) {
    auto &body = getBody();
    if (!body.hasOneBlock())
      return emitOpError("body region must have exactly one block");

    Block &entry = body.front();
    unsigned numInputs = getInputs().size();

    // Block arguments: one per input element type + one for init element
    // type.
    unsigned expectedBlockArgs = numInputs + 1;
    if (entry.getNumArguments() != expectedBlockArgs)
      return emitOpError("body block expects ")
             << expectedBlockArgs << " arguments ("
             << numInputs << " input element types + 1 init element type) "
             << "but has " << entry.getNumArguments();

    // Check block arg types match input element types.
    for (unsigned i = 0; i < numInputs; ++i) {
      auto inputEltType =
          cast<RankedTensorType>(getInputs()[i].getType()).getElementType();
      if (entry.getArgument(i).getType() != inputEltType)
        return emitOpError("body block argument ")
               << i << " type mismatch";
    }
    // Check last block arg matches init element type.
    if (entry.getArgument(numInputs).getType() != initEltType)
      return emitOpError("body block init argument type "
                         "mismatch");
  }

  return success();
}

// ReduceOp

LogicalResult ReduceOp::verify() {
  bool hasPureFn = getPureFnAttr() != nullptr;
  bool hasBody = !getBody().empty();

  // Exactly one of pure_fn or body must be present.
  if (hasPureFn == hasBody)
    return emitOpError("requires exactly one of 'pure_fn' or a body region");

  auto inputType = cast<RankedTensorType>(getInput().getType());
  auto initType = cast<RankedTensorType>(getInit().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());

  // Input must be at least 1D. A rank-0 input would pass validation but
  // ConvertReduceOp would then build a linalg.reduce with an empty
  // dimensions list.
  if (inputType.getRank() < 1)
    return emitOpError("reduce input must be at least 1D, got rank ")
           << inputType.getRank();

  // Init and result must match.
  if (initType != resultType)
    return emitOpError("init type must match result type");

  auto eltType = inputType.getElementType();

  // Input must be at least 1D. Init must be 0D (scalar tensor) or same
  // element type with reduced rank.
  if (initType.getRank() != 0)
    return emitOpError("reduce init must be a scalar tensor (rank 0), got "
                       "rank ")
           << initType.getRank();

  if (initType.getElementType() != eltType)
    return emitOpError("reduce init element type must match input element "
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

// VectorAddOp

LogicalResult VectorAddOp::verify() {
  auto lhsType = cast<RankedTensorType>(getLhs().getType());
  auto rhsType = cast<RankedTensorType>(getRhs().getType());
  auto initType = cast<RankedTensorType>(getInit().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());

  unsigned lhsRank = lhsType.getRank();
  unsigned rhsRank = rhsType.getRank();
  unsigned initRank = initType.getRank();
  unsigned resultRank = resultType.getRank();

  if (lhsRank != 1 || rhsRank != 1 || initRank != 1 || resultRank != 1)
    return emitOpError("expects all operands to be 1D tensors (vectors)");

  if (initType != resultType)
    return emitOpError("expects init type to match result type");

  auto lhsEltType = lhsType.getElementType();
  if (rhsType.getElementType() != lhsEltType ||
      initType.getElementType() != lhsEltType ||
      resultType.getElementType() != lhsEltType)
    return emitOpError("expects all operands and result to have the same "
                       "element type");

  // Check length compatibility.
  if (!lhsType.isDynamicDim(0) && !rhsType.isDynamicDim(0))
    if (lhsType.getDimSize(0) != rhsType.getDimSize(0))
      return emitOpError("vector length mismatch between lhs and rhs");

  if (!lhsType.isDynamicDim(0) && !initType.isDynamicDim(0))
    if (lhsType.getDimSize(0) != initType.getDimSize(0))
      return emitOpError("vector length mismatch between lhs and init");

  return success();
}

// Custom assembly: the result type is derived from `init` (destination-passing,
// like linalg.map), so the printed form omits `-> type($result)` and parse
// fills the result type from the `outs` operand.

ParseResult MapOp::parse(OpAsmParser &parser, OperationState &result) {
  PreferenceAttr preferenceAttr;
  FlatSymbolRefAttr pureFnAttr;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputsOperands;
  SMLoc inputsOperandsLoc;
  SmallVector<Type, 1> inputsTypes;
  OpAsmParser::UnresolvedOperand initRawOperand{};
  ArrayRef<OpAsmParser::UnresolvedOperand> initOperands(&initRawOperand, 1);
  SMLoc initOperandsLoc;
  Type initRawType{};
  ArrayRef<Type> initTypes(&initRawType, 1);
  std::unique_ptr<Region> bodyRegion = std::make_unique<Region>();

  if (succeeded(parser.parseOptionalKeyword("preference"))) {
    if (parser.parseEqual() ||
        parser.parseCustomAttributeWithFallback(preferenceAttr, Type{}))
      return failure();
    if (preferenceAttr)
      result.getOrAddProperties<MapOp::Properties>().preference = preferenceAttr;
  }
  if (succeeded(parser.parseOptionalKeyword("pure_fn"))) {
    if (parser.parseEqual() ||
        parser.parseCustomAttributeWithFallback(
            pureFnAttr, parser.getBuilder().getType<NoneType>()))
      return failure();
    if (pureFnAttr)
      result.getOrAddProperties<MapOp::Properties>().pure_fn = pureFnAttr;
  }
  if (parser.parseKeyword("ins") || parser.parseLParen())
    return failure();
  inputsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputsOperands) || parser.parseColon() ||
      parser.parseTypeList(inputsTypes) || parser.parseRParen())
    return failure();
  if (parser.parseKeyword("outs") || parser.parseLParen())
    return failure();
  initOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(initRawOperand) || parser.parseColon())
    return failure();
  {
    RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return failure();
    initRawType = type;
  }
  if (parser.parseRParen())
    return failure();
  {
    auto parseResult = parser.parseOptionalRegion(*bodyRegion);
    if (parseResult.has_value() && failed(*parseResult))
      return failure();
  }
  {
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalAttrDict(result.attributes))
      return failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
          return parser.emitError(loc)
                 << "'" << result.name.getStringRef() << "' op ";
        })))
      return failure();
  }
  result.addRegion(std::move(bodyRegion));
  // Result type is the init type (destination-passing).
  result.addTypes(initRawType);
  if (parser.resolveOperands(inputsOperands, inputsTypes, inputsOperandsLoc,
                             result.operands) ||
      parser.resolveOperands(initOperands, initTypes, initOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

void MapOp::print(OpAsmPrinter &p) {
  if (getPreferenceAttr()) {
    p << " preference = ";
    p.printStrippedAttrOrType(getPreferenceAttr());
  }
  if (getPureFnAttr()) {
    p << " pure_fn = ";
    p.printAttributeWithoutType(getPureFnAttr());
  }
  p << " ins(" << getInputs() << " : " << getInputs().getTypes() << ")";
  p << " outs(" << getInit() << " : " << getInit().getType() << ")";
  if (!getBody().empty()) {
    p << ' ';
    p.printRegion(getBody());
  }
  p.printOptionalAttrDict((*this)->getAttrs(), {"preference", "pure_fn"});
}

ParseResult ReduceOp::parse(OpAsmParser &parser, OperationState &result) {
  PreferenceAttr preferenceAttr;
  FlatSymbolRefAttr pureFnAttr;
  OpAsmParser::UnresolvedOperand inputRawOperand{};
  ArrayRef<OpAsmParser::UnresolvedOperand> inputOperands(&inputRawOperand, 1);
  SMLoc inputOperandsLoc;
  Type inputRawType{};
  ArrayRef<Type> inputTypes(&inputRawType, 1);
  OpAsmParser::UnresolvedOperand initRawOperand{};
  ArrayRef<OpAsmParser::UnresolvedOperand> initOperands(&initRawOperand, 1);
  SMLoc initOperandsLoc;
  Type initRawType{};
  ArrayRef<Type> initTypes(&initRawType, 1);
  std::unique_ptr<Region> bodyRegion = std::make_unique<Region>();

  if (succeeded(parser.parseOptionalKeyword("preference"))) {
    if (parser.parseEqual() ||
        parser.parseCustomAttributeWithFallback(preferenceAttr, Type{}))
      return failure();
    if (preferenceAttr)
      result.getOrAddProperties<ReduceOp::Properties>().preference = preferenceAttr;
  }
  if (succeeded(parser.parseOptionalKeyword("pure_fn"))) {
    if (parser.parseEqual() ||
        parser.parseCustomAttributeWithFallback(
            pureFnAttr, parser.getBuilder().getType<NoneType>()))
      return failure();
    if (pureFnAttr)
      result.getOrAddProperties<ReduceOp::Properties>().pure_fn = pureFnAttr;
  }
  if (parser.parseKeyword("ins") || parser.parseLParen())
    return failure();
  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperand) || parser.parseColon())
    return failure();
  {
    RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return failure();
    inputRawType = type;
  }
  if (parser.parseRParen())
    return failure();
  if (parser.parseKeyword("outs") || parser.parseLParen())
    return failure();
  initOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(initRawOperand) || parser.parseColon())
    return failure();
  {
    RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return failure();
    initRawType = type;
  }
  if (parser.parseRParen())
    return failure();
  {
    auto parseResult = parser.parseOptionalRegion(*bodyRegion);
    if (parseResult.has_value() && failed(*parseResult))
      return failure();
  }
  {
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalAttrDict(result.attributes))
      return failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
          return parser.emitError(loc)
                 << "'" << result.name.getStringRef() << "' op ";
        })))
      return failure();
  }
  result.addRegion(std::move(bodyRegion));
  // Result type is the init type (destination-passing).
  result.addTypes(initRawType);
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc,
                             result.operands) ||
      parser.resolveOperands(initOperands, initTypes, initOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

void ReduceOp::print(OpAsmPrinter &p) {
  if (getPreferenceAttr()) {
    p << " preference = ";
    p.printStrippedAttrOrType(getPreferenceAttr());
  }
  if (getPureFnAttr()) {
    p << " pure_fn = ";
    p.printAttributeWithoutType(getPureFnAttr());
  }
  p << " ins(" << getInput() << " : " << getInput().getType() << ")";
  p << " outs(" << getInit() << " : " << getInit().getType() << ")";
  if (!getBody().empty()) {
    p << ' ';
    p.printRegion(getBody());
  }
  p.printOptionalAttrDict((*this)->getAttrs(), {"preference", "pure_fn"});
}

ParseResult VectorAddOp::parse(OpAsmParser &parser, OperationState &result) {
  PreferenceAttr preferenceAttr;
  OpAsmParser::UnresolvedOperand lhsRawOperand{};
  ArrayRef<OpAsmParser::UnresolvedOperand> lhsOperands(&lhsRawOperand, 1);
  SMLoc lhsOperandsLoc;
  OpAsmParser::UnresolvedOperand rhsRawOperand{};
  ArrayRef<OpAsmParser::UnresolvedOperand> rhsOperands(&rhsRawOperand, 1);
  SMLoc rhsOperandsLoc;
  Type lhsRawType{};
  ArrayRef<Type> lhsTypes(&lhsRawType, 1);
  Type rhsRawType{};
  ArrayRef<Type> rhsTypes(&rhsRawType, 1);
  OpAsmParser::UnresolvedOperand initRawOperand{};
  ArrayRef<OpAsmParser::UnresolvedOperand> initOperands(&initRawOperand, 1);
  SMLoc initOperandsLoc;
  Type initRawType{};
  ArrayRef<Type> initTypes(&initRawType, 1);

  if (succeeded(parser.parseOptionalKeyword("preference"))) {
    if (parser.parseEqual() ||
        parser.parseCustomAttributeWithFallback(preferenceAttr, Type{}))
      return failure();
    if (preferenceAttr)
      result.getOrAddProperties<VectorAddOp::Properties>().preference =
          preferenceAttr;
  }
  if (parser.parseKeyword("ins") || parser.parseLParen())
    return failure();
  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperand) || parser.parseComma())
    return failure();
  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperand) || parser.parseColon())
    return failure();
  {
    RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return failure();
    lhsRawType = type;
  }
  if (parser.parseComma())
    return failure();
  {
    RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return failure();
    rhsRawType = type;
  }
  if (parser.parseRParen())
    return failure();
  if (parser.parseKeyword("outs") || parser.parseLParen())
    return failure();
  initOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(initRawOperand) || parser.parseColon())
    return failure();
  {
    RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return failure();
    initRawType = type;
  }
  if (parser.parseRParen())
    return failure();
  {
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalAttrDict(result.attributes))
      return failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
          return parser.emitError(loc)
                 << "'" << result.name.getStringRef() << "' op ";
        })))
      return failure();
  }
  // Result type is the init type (destination-passing).
  result.addTypes(initRawType);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc,
                             result.operands) ||
      parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc,
                             result.operands) ||
      parser.resolveOperands(initOperands, initTypes, initOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

void VectorAddOp::print(OpAsmPrinter &p) {
  if (getPreferenceAttr()) {
    p << " preference = ";
    p.printStrippedAttrOrType(getPreferenceAttr());
  }
  p << " ins(" << getLhs() << ", " << getRhs() << " : " << getLhs().getType()
    << ", " << getRhs().getType() << ")";
  p << " outs(" << getInit() << " : " << getInit().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), {"preference"});
}
