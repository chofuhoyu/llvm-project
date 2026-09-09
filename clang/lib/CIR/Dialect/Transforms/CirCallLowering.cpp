//===- CirCallLowering.cpp - rewrite CIR calls to func/skeleton -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Transforms/CirCallLowering.h"

#include "clang/CIR/Dialect/Transforms/CirFuncToArith.h"
#include "clang/CIR/Dialect/Transforms/CirScalarTypeConverter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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

  auto *ctx = module.getContext();
  CirScalarTypeConverter converter;
  for (cir::FuncOp cirFunc : toConvert) {
    SmallVector<Type> inputs, results;
    bool unsupported = false;
    for (Type t : cirFunc.getArgumentTypes()) {
      Type mapped = converter.convertType(t);
      if (!mapped) {
        cirFunc.emitError() << "unsupported CIR type in pure function: " << t;
        unsupported = true;
        break;
      }
      inputs.push_back(mapped);
    }
    if (unsupported)
      continue;
    for (Type t : cirFunc.getResultTypes()) {
      Type mapped = converter.convertType(t);
      if (!mapped) {
        cirFunc.emitError() << "unsupported CIR type in pure function: " << t;
        unsupported = true;
        break;
      }
      results.push_back(mapped);
    }
    if (unsupported)
      continue;

    auto funcType = FunctionType::get(ctx, inputs, results);
    OpBuilder builder(cirFunc);
    auto newFunc = func::FuncOp::create(builder, cirFunc.getLoc(),
                                        cirFunc.getSymName(), funcType);
    newFunc.setSymVisibility(cirFunc.getSymVisibility());

    // A pure function that carries a body needs it translated into an arith
    // body (populateArithFuncBody) so SkeletonToLinalg can clone it. On
    // failure populateArithFuncBody already emitted a diagnostic; roll the
    // partial func.func back and fail the pass.
    if (!cirFunc.isExternal()) {
      if (failed(populateArithFuncBody(cirFunc, newFunc, builder))) {
        newFunc.erase();
        return failure();
      }
    }

    cirFunc.erase();
  }
  return success();
}

/// Create bufferization.to_tensor from a memref.
static Value createToTensor(OpBuilder &builder, Location loc, Value memref) {
  auto memrefType = cast<MemRefType>(memref.getType());
  auto tensorType =
      RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
  return bufferization::ToTensorOp::create(builder, loc, tensorType, memref,
                                           /*restrict=*/true,
                                           /*writable=*/true);
}

/// Return the memref block argument of the rewritten function that backs the
/// `operandIndex`-th data operand of the skeleton call, or null if that operand
/// does not come straight from a host parameter.
///
/// The rewritten function keeps the host's parameter list in order
/// (rewriteToStandardFunc maps every cir.func parameter one-to-one), so a call
/// operand that is the host entry block's argument `k` is carried by `newFunc`'s
/// argument `k`. Resolving through the call's operand rather than grabbing
/// memref arguments positionally honors the call's actual argument order: a
/// host that passes its parameters in a different order, or does not use all of
/// them, must not shift or drop the map inputs.
static Value mapDataOperandToMemref(SkeletonCallInfo &info,
                                    func::FuncOp newFunc,
                                    unsigned operandIndex) {
  Value operand = info.callOp.getOperand(operandIndex);
  auto blockArg = dyn_cast<BlockArgument>(operand);
  if (!blockArg ||
      blockArg.getOwner()->getParentOp() != info.cirFunc.getOperation()) {
    info.callOp.emitWarning(
        "skeleton call data operand is not a direct parameter of the host "
        "function; only thin-wrapper hosts are supported");
    return {};
  }
  Value memref = newFunc.getArgument(blockArg.getArgNumber());
  if (!isa<MemRefType>(memref.getType())) {
    info.callOp.emitWarning(
        "skeleton call data operand maps to a non-pointer host parameter");
    return {};
  }
  return memref;
}

/// The skeleton op's output tensor type, determined by the op's semantics.
/// map is element-wise, so its output is 1-D with a dynamic extent matching
/// the input; reduce collapses its input to a rank-0 scalar. The element type
/// equals the input's (map/reduce verifiers require output elt == input elt).
/// Future operators add their own output-shape rule here.
static RankedTensorType outputTensorType(SkeletonOpType opType, Type eltTy) {
  if (opType == SkeletonOpType::Reduce)
    return RankedTensorType::get({}, eltTy);
  return RankedTensorType::get({ShapedType::kDynamic}, eltTy);
}

/// Create a tensor.empty whose type matches the rewritten function's output
/// tensor type (the skeleton op's result). Dynamic extents in the result
/// shape are filled from the corresponding dimension of the first input
/// tensor, since map/reduce outputs share the input's shape.
static Value createEmptyOutput(OpBuilder &builder, Location loc,
                               func::FuncOp newFunc, Value firstInputTensor) {
  auto resultTy = cast<RankedTensorType>(newFunc.getResultTypes().front());
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
/// data operands are all inputs, so every pointer parameter becomes a 1-D
/// memref with a dynamic extent (a bare C pointer does not carry its length).
/// Non-pointer parameters map to their scalar type.
///
/// The original body is discarded and the function is given a return type by
/// this pass rather than read from the original cir.func. That return type is
/// the skeleton op's result tensor (map/reduce's `outs` type): shape from the
/// op semantics, element type from the first pointer's pointee, since the op
/// verifiers require the output elt to equal the input's. This is only sound
/// while the host body really is just the one skeleton call; other statements
/// would be silently dropped and several skeleton calls would each emit a
/// same-named func.func (see review doc front-end follow-ups).
func::FuncOp rewriteToStandardFunc(cir::FuncOp cirFunc, OpBuilder &rewriter,
                                   SkeletonOpType opType) {
  auto loc = cirFunc.getLoc();
  auto *ctx = rewriter.getContext();
  CirScalarTypeConverter converter;

  SmallVector<Type> newInputTypes;
  Type eltTy;
  for (unsigned i = 0; i < cirFunc.getNumArguments(); ++i) {
    Type argTy = cirFunc.getArgument(i).getType();
    if (auto ptrTy = dyn_cast<cir::PointerType>(argTy)) {
      auto mappedElt = converter.convertType(ptrTy.getPointee());
      if (!mappedElt) {
        cirFunc.emitError() << "unsupported CIR type in skeleton function: "
                            << ptrTy.getPointee();
        return nullptr;
      }
      newInputTypes.push_back(
          MemRefType::get({ShapedType::kDynamic}, mappedElt));
      if (!eltTy)
        eltTy = mappedElt;
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

  // A skeleton call always has at least one data (vector) operand, so its
  // element type is available to type the returned tensor.
  if (!eltTy) {
    cirFunc.emitError() << "skeleton function needs a pointer (vector) "
                           "argument to infer the output element type";
    return nullptr;
  }

  auto funcType =
      FunctionType::get(ctx, newInputTypes, {outputTensorType(opType, eltTy)});

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

  // Map each data operand of the skeleton call (operand 0 is the pure function
  // reference) to the memref holding the host parameter it was passed from,
  // then view that memref as a tensor. Walking the call's operands keeps the
  // map's inputs in the call's argument order (return-value style: every data
  // operand is an input).
  SmallVector<Value> inputTensors;
  for (unsigned i = 1; i < info.callOp.getNumOperands(); ++i) {
    Value memref = mapDataOperandToMemref(info, newFunc, i);
    if (!memref)
      return {};
    inputTensors.push_back(createToTensor(builder, loc, memref));
  }
  if (inputTensors.empty()) {
    info.callOp.emitWarning("skeleton map call has no data operands");
    return {};
  }

  // Output is a fresh tensor.empty (the outs/destination), not a
  // caller-supplied buffer.
  Value init = createEmptyOutput(builder, loc, newFunc, inputTensors[0]);

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

  // The reduce call's single data operand, resolved the same way as
  // lowerMapCall's (operand 0 is the pure function reference).
  Value memref = mapDataOperandToMemref(info, newFunc, 1);
  if (!memref)
    return {};
  auto inputTensor = createToTensor(builder, loc, memref);

  // Output is a fresh rank-0 tensor.empty.
  Value init = createEmptyOutput(builder, loc, newFunc, inputTensor);

  auto pureFnAttr = SymbolRefAttr::get(ctx, info.pureFnName);
  auto prefAttr =
      skeleton::PreferenceAttr::get(ctx, StringAttr::get(ctx, toString(info.preference)));

  auto reduceOp = skeleton::ReduceOp::create(
      builder, loc, init.getType(), inputTensor, init, pureFnAttr, prefAttr);

  return reduceOp.getResult();
}

} // namespace cir
