//===- CirCallLowering.cpp - rewrite CIR calls to func/skeleton -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Transforms/CirCallLowering.h"

#include "clang/CIR/Dialect/Transforms/CirCallAnalysis.h"
#include "clang/CIR/Dialect/Transforms/CirFuncToArith.h"
#include "clang/CIR/Dialect/Transforms/CirScalarTypeConverter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonAttrs.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace cir;

namespace cir {

LogicalResult convertPureFunctionsToFunc(ModuleOp module,
                                         const DenseSet<StringRef> &pureFns) {
  SmallVector<cir::FuncOp> toConvert;
  module.walk([&](cir::FuncOp func) {
    if (pureFns.contains(func.getSymName()))
      toConvert.push_back(func);
  });

  for (cir::FuncOp cirFunc : toConvert) {
    OpBuilder builder(cirFunc);
    auto newFunc = createFuncFromCirBody(cirFunc, cirFunc.getSymName(), builder);
    if (!newFunc)
      return failure();
    cirFunc.erase();
  }
  return success();
}

/// Create bufferization.to_tensor from a memref.
Value createToTensor(OpBuilder &builder, Location loc, Value memref) {
  auto memrefType = cast<MemRefType>(memref.getType());
  auto tensorType =
      RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
  return bufferization::ToTensorOp::create(builder, loc, tensorType, memref,
                                           /*restrict=*/true,
                                           /*writable=*/true);
}

/// The two members of skeleton_detail::span<T>, in the order the manual-path
/// header declares them (see SkeletonOps.h): the data pointer and the length.
/// Their roles are part of the project's own type, so the pass reads the layout
/// off that declaration instead of guessing at it.
static constexpr unsigned kSpanDataMember = 0;
static constexpr unsigned kSpanSizeMember = 1;

/// Whether \p ty is the skeleton_detail::span<T> the manual-path header
/// declares. It is recognized by name together with both members' roles, so
/// that a struct which merely looks similar is not taken for one.
static bool isSkeletonSpan(cir::StructType ty) {
  StringAttr name = ty.getName();
  return name && name.getValue().starts_with("skeleton_detail::span<") &&
         ty.getNumElements() == 2 &&
         isa<cir::PointerType>(ty.getElementType(kSpanDataMember)) &&
         isa<cir::IntType>(ty.getElementType(kSpanSizeMember));
}

FailureOr<MemRefType> hostParamToMemRef(Type paramTy) {
  auto ptrTy = dyn_cast<cir::PointerType>(paramTy);
  if (!ptrTy)
    return failure();

  // Each array layer of the pointee contributes one static extent, outermost
  // first, which is memref's own dimension order.
  Type elemTy = ptrTy.getPointee();
  SmallVector<int64_t> shape;
  while (auto arrTy = dyn_cast<cir::ArrayType>(elemTy)) {
    shape.push_back(arrTy.getSize());
    elemTy = arrTy.getElementType();
  }

  if (auto structTy = dyn_cast<cir::StructType>(elemTy)) {
    // A span is itself the data array, so no array layer may sit above it.
    if (!isSkeletonSpan(structTy) || !shape.empty())
      return failure();
    // The element type comes from the data member, and the length in the size
    // member is a run-time value: one dynamic extent. The length's value
    // reaches the skeleton op as that extent of the memref the caller of the
    // rewritten function passes.
    elemTy = cast<cir::PointerType>(structTy.getElementType(kSpanDataMember))
                 .getPointee();
    shape.push_back(ShapedType::kDynamic);
  } else if (shape.empty()) {
    // A bare pointer says nothing about its length and has no member to say it
    // with, so its extent is dynamic as well.
    shape.push_back(ShapedType::kDynamic);
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

  SmallVector<unsigned> hostParams;
  MemRefType dataTy;
  for (unsigned op = 1, e = callOp.getNumOperands(); op < e; ++op) {
    unsigned dataIdx = op - 1;
    auto hostArg = resolveToHostArg(callOp.getOperand(op), host);
    if (failed(hostArg)) {
      callOp.emitError() << "skeleton call data operand " << dataIdx
                         << " is not a direct parameter of the host function; "
                            "only thin-wrapper hosts are supported";
      return failure();
    }

    unsigned param = hostArg->arg.getArgNumber();
    Type paramTy = host.getArgument(param).getType();
    auto memrefTy = hostParamToMemRef(paramTy);
    if (failed(memrefTy)) {
      callOp.emitError()
          << "skeleton call data operand " << dataIdx << " is host parameter "
          << param << " of type " << paramTy
          << ", which is not a data array: write a compile-time extent as an "
             "array reference (float (&a)[4]) and a runtime one as "
             "skeleton_detail::span<float>";
      return failure();
    }

    // A member names part of its parameter. The only one that is a data array
    // is the span's data pointer; its size member is a length.
    if (hostArg->memberIndex && *hostArg->memberIndex != kSpanDataMember) {
      callOp.emitError() << "skeleton call data operand " << dataIdx
                         << " is member " << *hostArg->memberIndex
                         << " of host parameter " << param << " of type "
                         << paramTy << ", which is not a data array: pass the "
                         << "parameter itself or its data member";
      return failure();
    }

    // Every data operand must denote one and the same type. The skeleton.map
    // verifier accepts a static dimension against a dynamic one, so a host
    // mixing `float (&a)[4]` with `float *b` would verify and then read out of
    // bounds.
    if (dataIdx == 0) {
      dataTy = *memrefTy;
    } else if (*memrefTy != dataTy) {
      callOp.emitError() << "skeleton call data operand " << dataIdx
                         << " denotes " << *memrefTy << " but data operand 0 "
                         << "denotes " << dataTy
                         << "; all data operands must have the same element "
                            "type and shape";
      return failure();
    }

    hostParams.push_back(param);
  }

  if (hostParams.empty()) {
    callOp.emitError("skeleton call has no data operands");
    return failure();
  }

  info.dataOperandHostParams = std::move(hostParams);
  info.dataMemRef = dataTy;
  return success();
}

/// The skeleton op's output tensor type, determined by the op's semantics.
/// map is element-wise, so its output keeps its input's shape at any rank;
/// reduce collapses its input to a rank-0 scalar. The element type equals the
/// input's (map/reduce verifiers require output elt == input elt). Future
/// operators add their own output-shape rule here.
RankedTensorType outputTensorType(SkeletonOpType opType,
                                  ArrayRef<int64_t> inputShape, Type eltTy) {
  if (opType == SkeletonOpType::Reduce)
    return RankedTensorType::get({}, eltTy);
  return RankedTensorType::get(inputShape, eltTy);
}

/// Create a tensor.empty of \p resultTy. Dynamic extents are filled from the
/// corresponding dimension of the input tensor the output shares its shape
/// with.
Value createEmptyOutput(OpBuilder &builder, Location loc,
                        RankedTensorType resultTy, Value firstInputTensor) {
  ArrayRef<int64_t> shape = resultTy.getShape();
  SmallVector<Value> dynSizes;
  for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i) {
    if (!ShapedType::isDynamic(shape[i]))
      continue;
    auto index = arith::ConstantIndexOp::create(builder, loc, i);
    dynSizes.push_back(
        tensor::DimOp::create(builder, loc, firstInputTensor, index));
  }
  return tensor::EmptyOp::create(builder, loc, shape, resultTy.getElementType(),
                                 dynSizes);
}

void emitExtentCheck(OpBuilder &builder, Location loc,
                     RankedTensorType resultTy, ArrayRef<Value> inputTensors) {
  if (inputTensors.size() < 2)
    return;

  ArrayRef<int64_t> shape = resultTy.getShape();
  for (int64_t d = 0; d < static_cast<int64_t>(shape.size()); ++d) {
    if (!ShapedType::isDynamic(shape[d]))
      continue;
    auto dimOf = [&](Value tensor) {
      auto index = arith::ConstantIndexOp::create(builder, loc, d);
      return tensor::DimOp::create(builder, loc, tensor, index).getResult();
    };
    Value first = dimOf(inputTensors.front());
    for (Value other : inputTensors.drop_front()) {
      Value equal = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::eq, first, dimOf(other));
      cf::AssertOp::create(builder, loc, equal,
                           "skeleton.map inputs must have the same length");
    }
  }
}

// TODO(cir-call-to-skeleton, host-function-wrapper): Rewriting the host
// function below assumes it is a thin wrapper around a single skeleton call.
// The whole original body is discarded and the return type is invented by this
// pass, so a host with anything besides the one skeleton call, or with several
// skeleton calls, is silently mis-rewritten (extra statements dropped,
// same-named func.func collisions). Validate the wrapper shape and error out on
// non-wrapper hosts, or grow the rewrite to keep non-skeleton statements and
// merge several skeleton calls into one func.func.

/// Rewrite the cir.func that hosts a skeleton call to a func.func.
///
/// This treats the host function as a thin wrapper around a single skeleton
/// call (the shape a user's skeleton region currently must take): the call's
/// data operands are all inputs, so every data-array parameter becomes the
/// memref its type denotes (see hostParamToMemRef) and every other parameter
/// maps to its scalar type.
///
/// The original body is discarded and the function is given a return type by
/// this pass rather than read from the original cir.func. That return type is
/// the skeleton op's result tensor (map/reduce's `outs` type), shaped from the
/// memref the data operands denote (info.dataMemRef, recorded by
/// resolveDataOperands): map keeps that input's shape, since the op is
/// element-wise, and reduce collapses it to a rank-0 tensor. This is only
/// sound while the host body really is just the one skeleton call; other
/// statements would be silently dropped and several skeleton calls would each
/// emit a same-named func.func (see review doc front-end follow-ups).
func::FuncOp rewriteToStandardFunc(SkeletonCallInfo &info,
                                   OpBuilder &rewriter) {
  cir::FuncOp cirFunc = info.cirFunc;
  auto loc = cirFunc.getLoc();
  auto *ctx = rewriter.getContext();
  CirScalarTypeConverter converter;

  SmallVector<Type> newInputTypes;
  for (unsigned i = 0; i < cirFunc.getNumArguments(); ++i) {
    Type argTy = cirFunc.getArgument(i).getType();
    if (isa<cir::PointerType>(argTy)) {
      auto memrefTy = hostParamToMemRef(argTy);
      if (failed(memrefTy)) {
        cirFunc.emitError()
            << "unsupported CIR type in skeleton function: " << argTy
            << "; a data parameter is written as an array reference "
               "(float (&a)[4]) for a compile-time extent or as "
               "skeleton_detail::span<float> for a runtime one";
        return nullptr;
      }
      newInputTypes.push_back(*memrefTy);
    } else {
      auto scalarTy = converter.convertType(argTy);
      if (!scalarTy) {
        cirFunc.emitError()
            << "unsupported CIR type in skeleton function: " << argTy;
        return nullptr;
      }
      newInputTypes.push_back(scalarTy);
    }
  }

  auto resultTy = outputTensorType(info.opType, info.dataMemRef.getShape(),
                                   info.dataMemRef.getElementType());
  auto funcType = FunctionType::get(ctx, newInputTypes, {resultTy});

  auto newFunc =
      func::FuncOp::create(rewriter, loc, cirFunc.getSymName(), funcType);
  newFunc.setSymVisibility(cirFunc.getSymVisibility());

  auto *entryBlock = newFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  return newFunc;
}

Value lowerMapCall(SkeletonCallInfo &info, func::FuncOp newFunc,
                   OpBuilder &builder) {
  auto loc = info.callOp.getLoc();
  auto *ctx = builder.getContext();
  auto resultTy = cast<RankedTensorType>(newFunc.getResultTypes().front());

  // Each recorded host parameter backs one data operand; visiting them in the
  // recorded order keeps the map's inputs in the call's argument order
  // (return-value style: every data operand is an input). The rewritten
  // function keeps the host's parameter list in order, so the parameter index
  // is the argument index.
  SmallVector<Value> inputTensors;
  for (unsigned hostParam : info.dataOperandHostParams)
    inputTensors.push_back(
        createToTensor(builder, loc, newFunc.getArgument(hostParam)));

  // Inputs of a runtime length are only known to agree at run time.
  emitExtentCheck(builder, loc, resultTy, inputTensors);

  // Output is a fresh tensor.empty (the outs/destination), not a
  // caller-supplied buffer.
  Value init = createEmptyOutput(builder, loc, resultTy, inputTensors[0]);

  auto pureFnAttr = SymbolRefAttr::get(ctx, info.pureFnName);
  auto prefAttr =
      skeleton::PreferenceAttr::get(ctx, StringAttr::get(ctx, toString(info.preference)));

  auto mapOp = skeleton::MapOp::create(
      builder, loc, init.getType(), inputTensors, init, pureFnAttr, prefAttr);

  return mapOp.getResult();
}

Value lowerReduceCall(SkeletonCallInfo &info, func::FuncOp newFunc,
                      OpBuilder &builder) {
  auto loc = info.callOp.getLoc();
  auto *ctx = builder.getContext();
  auto resultTy = cast<RankedTensorType>(newFunc.getResultTypes().front());

  // The reduce call's single data operand.
  auto inputTensor = createToTensor(
      builder, loc, newFunc.getArgument(info.dataOperandHostParams.front()));

  // Output is a fresh rank-0 tensor.empty.
  Value init = createEmptyOutput(builder, loc, resultTy, inputTensor);

  auto pureFnAttr = SymbolRefAttr::get(ctx, info.pureFnName);
  auto prefAttr =
      skeleton::PreferenceAttr::get(ctx, StringAttr::get(ctx, toString(info.preference)));

  auto reduceOp = skeleton::ReduceOp::create(
      builder, loc, init.getType(), inputTensor, init, pureFnAttr, prefAttr);

  return reduceOp.getResult();
}

} // namespace cir
