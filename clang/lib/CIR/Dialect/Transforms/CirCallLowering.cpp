//===- CirCallLowering.cpp - shape and ABI of a skeleton helper call -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Transforms/CirCallLowering.h"

#include "clang/CIR/Dialect/Transforms/CirCallAnalysis.h"
#include "clang/CIR/Dialect/Transforms/CirScalarTypeConverter.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace cir;

namespace cir {

bool isSkeletonSpan(Type ty) {
  auto structTy = dyn_cast<cir::StructType>(ty);
  if (!structTy)
    return false;
  StringAttr name = structTy.getName();
  return name && name.getValue().starts_with("skeleton_detail::span<") &&
         structTy.getNumElements() == 2 &&
         isa<cir::PointerType>(structTy.getElementType(kSpanDataMember)) &&
         isa<cir::IntType>(structTy.getElementType(kSpanSizeMember));
}

Value createToTensor(OpBuilder &builder, Location loc, Value memref,
                     bool writable) {
  auto memrefType = cast<MemRefType>(memref.getType());
  auto tensorType =
      RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
  // The host may pass the same buffer as an input and as the destination (an
  // in-place step of the DAG), so neither view claims to be unrestricted: the
  // write stays in place because the destination is writable, not because the
  // buffers are known not to alias.
  return bufferization::ToTensorOp::create(builder, loc, tensorType, memref,
                                           /*restrict=*/false,
                                           /*writable=*/writable);
}

FailureOr<MemRefType> hostParamToMemRef(Type paramTy, bool isDestination) {
  auto ptrTy = dyn_cast<cir::PointerType>(paramTy);
  if (!ptrTy)
    return failure();

  // Each array layer of the pointee contributes one compile-time extent,
  // outermost first, which is memref's own dimension order.
  Type elemTy = ptrTy.getPointee();
  SmallVector<int64_t> shape;
  while (auto arrTy = dyn_cast<cir::ArrayType>(elemTy)) {
    shape.push_back(arrTy.getSize());
    elemTy = arrTy.getElementType();
  }

  bool hasRuntimeExtent = false;
  if (isSkeletonSpan(elemTy)) {
    // A span is itself the data array, so no array layer may sit above it. Its
    // element type comes from the data member and its length is a run-time
    // value, i.e. one dynamic extent.
    if (!shape.empty())
      return failure();
    elemTy = cast<cir::PointerType>(
                 cast<cir::StructType>(elemTy).getElementType(kSpanDataMember))
                 .getPointee();
    shape.push_back(ShapedType::kDynamic);
    hasRuntimeExtent = true;
  } else if (shape.empty()) {
    // A single element. Only a destination can say what it means: an input
    // pointer has no length with it, and a bare pointer parameter cannot carry
    // one.
    if (!isDestination)
      return failure();
  }

  CirScalarTypeConverter converter;
  auto mappedElt = converter.convertType(elemTy);
  if (!mappedElt)
    return failure();
  return MemRefType::get(shape, mappedElt);
}

LogicalResult resolveDataOperands(SkeletonCallInfo &info) {
  cir::CallOp callOp = info.callOp;
  cir::FuncOp host = info.cirFunc;

  // Operand 0 is the pure function reference; every other argument is either a
  // data argument (a pointer the ABI thunk turns into a memref descriptor) or a
  // scalar the unit takes as it is.
  for (unsigned op = 1, e = callOp.getNumOperands(); op < e; ++op) {
    Value operand = callOp.getOperand(op);
    unsigned argIdx = op - 1;

    if (!isa<cir::PointerType>(operand.getType())) {
      CirScalarTypeConverter converter;
      if (!converter.convertType(operand.getType())) {
        callOp.emitError() << "skeleton call argument " << argIdx
                           << " has type " << operand.getType()
                           << ", which is neither a data argument nor a scalar";
        return failure();
      }
      info.scalarOperands.push_back(op);
      continue;
    }

    auto hostArg = resolveToHostArg(operand, host);
    if (failed(hostArg)) {
      callOp.emitError() << "skeleton call argument " << argIdx
                         << " is not a direct parameter of the host function; "
                            "a skeleton region host must pass its own "
                            "parameters straight through to the operator";
      return failure();
    }

    unsigned param = hostArg->arg.getArgNumber();
    Type paramTy = host.getArgument(param).getType();
    // The first data argument is the destination. It decides the result shape,
    // and it is the only position where a pointer to a single element is a
    // shape rather than a missing length.
    bool isDestination = info.dataOperands.empty();
    auto memrefTy = hostParamToMemRef(paramTy, isDestination);
    if (failed(memrefTy)) {
      callOp.emitError()
          << "skeleton call argument " << argIdx << " is host parameter "
          << param << " of type " << paramTy
          << ", which is not a data argument: write a compile-time extent as "
             "an array reference (float (&a)[4]), a runtime one as "
             "skeleton_detail::span<float>, and a single-element destination "
             "as float &";
      return failure();
    }

    // A member names part of its parameter. The only one that is a data array
    // is the span's data pointer; its size member is a length.
    if (hostArg->memberIndex && *hostArg->memberIndex != kSpanDataMember) {
      callOp.emitError() << "skeleton call argument " << argIdx << " is member "
                         << *hostArg->memberIndex << " of host parameter "
                         << param << " of type " << paramTy
                         << ", which is not a data array: pass the parameter "
                            "itself or its data member";
      return failure();
    }

    // Every data argument must agree on the element type. The extents may
    // differ: an input of a lower rank is broadcast over the destination, and
    // a reduce's input is larger than its destination by the reduced
    // dimensions. Element types that disagree would be silently reinterpreted.
    if (!info.dataOperands.empty() &&
        memrefTy->getElementType() !=
            info.dataOperands.front().type.getElementType()) {
      callOp.emitError() << "skeleton call argument " << argIdx << " denotes "
                         << *memrefTy << " but the destination denotes "
                         << info.dataOperands.front().type
                         << "; all data arguments must have the same element "
                            "type";
      return failure();
    }

    info.dataOperands.push_back(
        DataOperandRef{param, *memrefTy,
                       isSkeletonSpan(cast<cir::PointerType>(paramTy)
                                          .getPointee())});
  }

  if (info.dataOperands.empty()) {
    callOp.emitError("skeleton call has no data arguments");
    return failure();
  }

  // A reduce's destination may not have more dimensions than its input: the
  // operator reduces the trailing dimensions the destination does not have.
  if (info.opType == SkeletonOpType::Reduce &&
      info.dataOperands.size() > 1 &&
      info.dataOperands[0].type.getRank() >
          info.dataOperands[1].type.getRank()) {
    callOp.emitError() << "skeleton.reduce destination "
                       << info.dataOperands[0].type
                       << " has more dimensions than its input "
                       << info.dataOperands[1].type;
    return failure();
  }

  return success();
}

} // namespace cir
