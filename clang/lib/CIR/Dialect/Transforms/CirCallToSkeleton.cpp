//===- CirCallToSkeleton.cpp - CIR calls to Skeleton ops --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Full manual path: Recognizes calls to skeleton helper functions (declared
// with the "skeleton_op" annotation in headers) and replaces them with
// Skeleton dialect structured computation operations.
//
// Semi-automatic path (TODO): annotated for-loops with inline computation.
// Lambda support (TODO): captureless lambdas as pure_fn references.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonAttrs.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonDialect.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_CIRCALLTOSKELETON
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

//===----------------------------------------------------------------------===//
// Metadata collection (Phase 1: gather info from original CIR IR)
//===----------------------------------------------------------------------===//

/// Collect function names marked with skeleton.pure.
static DenseSet<StringRef> collectPureFunctions(ModuleOp module) {
  DenseSet<StringRef> pureFns;
  module.walk([&](cir::FuncOp func) {
    if (func->hasAttr("skeleton.pure"))
      pureFns.insert(func.getSymName());
  });
  return pureFns;
}

/// Collect skeleton op declarations: external functions with "skeleton.op"
/// attribute. Returns map: function name → op type ("map", "reduce").
static DenseMap<StringRef, StringRef> collectSkeletonOpDecls(ModuleOp module) {
  DenseMap<StringRef, StringRef> opDecls;
  module.walk([&](cir::FuncOp func) {
    if (!func.isExternal())
      return;
    if (auto opAttr = func->getAttrOfType<StringAttr>("skeleton.op"))
      opDecls[func.getSymName()] = opAttr.getValue();
  });
  return opDecls;
}

/// Extract a FlatSymbolRefAttr from a call argument that represents a
/// function pointer (e.g. via cir.get_global @some_fn).
static FlatSymbolRefAttr extractPureFnRef(cir::CallOp callOp, unsigned argIdx,
                                          const DenseSet<StringRef> &pureFns) {
  if (argIdx >= callOp.getNumOperands())
    return {};
  Value arg = callOp.getOperand(argIdx);

  // Walk through cir.cast ops.
  while (auto castOp = arg.getDefiningOp<cir::CastOp>())
    arg = castOp.getSrc();

  // cir.get_global gives a direct reference.
  if (auto getGlobal = arg.getDefiningOp<cir::GetGlobalOp>()) {
    StringRef name = getGlobal.getName();
    if (pureFns.contains(name))
      return SymbolRefAttr::get(callOp.getContext(), name);
  }

  // TODO: Handle indirect references through cir.load chains.
  // TODO: Lambda-to-function-pointer conversion.
  return {};
}

/// Extract preference string from a cir.func's annotations attribute.
static std::string extractPreference(cir::FuncOp func) {
  auto annotations = func->getAttrOfType<ArrayAttr>("annotations");
  if (annotations && !annotations.empty()) {
    if (auto ann = dyn_cast<cir::AnnotationAttr>(annotations[0])) {
      if (auto args = ann.getArgs()) {
        if (!args.empty())
          if (auto pref = dyn_cast<StringAttr>(args[0]))
            return pref.getValue().str();
      }
    }
  }
  return "CPU";
}

/// Map a CIR scalar type to standard MLIR type.
static Type cirTypeToStdType(Type cirTy, MLIRContext *ctx) {
  if (cirTy.isInteger(32))
    return IntegerType::get(ctx, 32);
  // cir.float → f32, cir.double → f64
  if (cirTy.isF32() || isa<cir::SingleType>(cirTy))
    return Float32Type::get(ctx);
  if (cirTy.isF64() || isa<cir::DoubleType>(cirTy))
    return Float64Type::get(ctx);
  return Float32Type::get(ctx); // default
}

/// A work item extracted from the original cir.func before rewriting.
struct SkeletonCallInfo {
  cir::FuncOp cirFunc;
  cir::CallOp callOp;
  StringRef opType;       // "map" or "reduce"
  StringRef pureFnName;   // the referenced pure function
  std::string preference; // "CPU" or "GPU"
  // Number of input operands and output operand positions in the call.
  unsigned numInputs;
  unsigned numDataOps; // total data operands (inputs + output)
};

//===----------------------------------------------------------------------===//
// Phase 2: Rewrite cir.func → func.func and create skeleton ops
//===----------------------------------------------------------------------===//

/// Create bufferization.to_tensor from a memref.
static Value createToTensor(OpBuilder &builder, Location loc, Value memref) {
  auto memrefType = cast<MemRefType>(memref.getType());
  auto tensorType = RankedTensorType::get(memrefType.getShape(),
                                          memrefType.getElementType());
  return builder.create<bufferization::ToTensorOp>(loc, tensorType, memref,
                                                    /*restrict=*/true,
                                                    /*writable=*/true);
}

/// Create bufferization.materialize_in_destination.
static void createMaterializeInDest(OpBuilder &builder, Location loc,
                                     Value tensor, Value memref) {
  builder.create<bufferization::MaterializeInDestinationOp>(
      loc, tensor, memref);
}

/// Rewrite a cir.func to a func.func with memref parameters.
/// Returns the new func.func.
static func::FuncOp rewriteToStandardFunc(cir::FuncOp cirFunc,
                                           OpBuilder &rewriter) {
  auto loc = cirFunc.getLoc();
  auto ctx = rewriter.getContext();

  SmallVector<Type> newInputTypes;
  for (unsigned i = 0; i < cirFunc.getNumArguments(); ++i) {
    Type argTy = cirFunc.getArgument(i).getType();
    if (auto ptrTy = dyn_cast<cir::PointerType>(argTy)) {
      auto eltTy = cirTypeToStdType(ptrTy.getPointee(), ctx);
      newInputTypes.push_back(
          MemRefType::get({ShapedType::kDynamic}, eltTy));
    } else {
      newInputTypes.push_back(IndexType::get(ctx));
    }
  }

  auto funcType = FunctionType::get(ctx, newInputTypes, {});

  auto newFunc = rewriter.create<func::FuncOp>(loc, cirFunc.getSymName(),
                                                funcType);
  newFunc.setSymVisibility(cirFunc.getSymVisibility());

  auto *entryBlock = newFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  return newFunc;
}

/// Process a map call: create skeleton.map from the collected info.
static LogicalResult lowerMapCall(SkeletonCallInfo &info,
                                   func::FuncOp newFunc,
                                   OpBuilder &builder) {
  auto loc = info.callOp.getLoc();
  auto ctx = builder.getContext();

  // Map call args to new function's memref block arguments.
  // We need to find the memref block args that correspond to the original
  // cir.ptr params. In the rewritten function, all cir.ptr<T> params become
  // memref<?xT>. We walk all memref block args and assign them positionally.
  SmallVector<Value> memrefArgs;
  for (unsigned i = 0; i < newFunc.getNumArguments(); ++i) {
    if (isa<MemRefType>(newFunc.getArgument(i).getType()))
      memrefArgs.push_back(newFunc.getArgument(i));
  }

  // The call has: arg0=pure_fn, arg1..argN-1=inputs, argN=output
  // We need numDataOps memrefs.
  if (memrefArgs.size() < info.numDataOps) {
    info.callOp.emitWarning("not enough memref arguments in rewritten function");
    return failure();
  }

  // Create to_tensor for each input memref.
  SmallVector<Value> inputTensors;
  for (unsigned i = 0; i < info.numInputs; ++i)
    inputTensors.push_back(createToTensor(builder, loc, memrefArgs[i]));

  // Output is the last data operand.
  Value outputMemref = memrefArgs[info.numInputs];
  Value outputTensor = createToTensor(builder, loc, outputMemref);

  // Create skeleton.map.
  auto pureFnAttr = SymbolRefAttr::get(ctx, info.pureFnName);
  auto prefAttr = skeleton::PreferenceAttr::get(
      ctx, StringAttr::get(ctx, info.preference));

  auto mapOp = builder.create<skeleton::MapOp>(
      loc, outputTensor.getType(), inputTensors, outputTensor, pureFnAttr,
      prefAttr);

  createMaterializeInDest(builder, loc, mapOp.getResult(), outputMemref);
  return success();
}

/// Process a reduce call.
static LogicalResult lowerReduceCall(SkeletonCallInfo &info,
                                      func::FuncOp newFunc,
                                      OpBuilder &builder) {
  auto loc = info.callOp.getLoc();
  auto ctx = builder.getContext();

  SmallVector<Value> memrefArgs;
  for (unsigned i = 0; i < newFunc.getNumArguments(); ++i) {
    if (isa<MemRefType>(newFunc.getArgument(i).getType()))
      memrefArgs.push_back(newFunc.getArgument(i));
  }

  if (memrefArgs.size() < info.numDataOps) {
    info.callOp.emitWarning("not enough memref arguments");
    return failure();
  }

  Value inputMemref = memrefArgs[0];
  Value outputMemref = memrefArgs[1];

  auto inputTensor = createToTensor(builder, loc, inputMemref);
  auto outputTensor = createToTensor(builder, loc, outputMemref);

  auto pureFnAttr = SymbolRefAttr::get(ctx, info.pureFnName);
  auto prefAttr = skeleton::PreferenceAttr::get(
      ctx, StringAttr::get(ctx, info.preference));

  auto reduceOp = builder.create<skeleton::ReduceOp>(
      loc, outputTensor.getType(), inputTensor, outputTensor, pureFnAttr,
      prefAttr);

  createMaterializeInDest(builder, loc, reduceOp.getResult(), outputMemref);
  return success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class CIRCallToSkeletonPass
    : public impl::CIRCallToSkeletonBase<CIRCallToSkeletonPass> {
public:
  using CIRCallToSkeletonBase::CIRCallToSkeletonBase;

  void runOnOperation() override {
    auto module = getOperation();
    auto pureFns = collectPureFunctions(module);
    auto opDecls = collectSkeletonOpDecls(module);

    if (opDecls.empty())
      return;

    // Phase 1: Collect all skeleton call info from the original CIR IR.
    SmallVector<SkeletonCallInfo> worklist;

    module.walk([&](cir::CallOp callOp) {
      auto calleeOpt = callOp.getCallee();
      if (!calleeOpt)
        return;
      StringRef callee = *calleeOpt;

      auto it = opDecls.find(callee);
      if (it == opDecls.end())
        return;

      StringRef opType = it->second;

      auto pureFnRef = extractPureFnRef(callOp, 0, pureFns);
      if (!pureFnRef) {
        callOp.emitWarning("skeleton op call first argument must be a pure "
                           "function reference; semi-automatic path not yet "
                           "supported");
        return;
      }

      auto cirFunc = callOp->getParentOfType<cir::FuncOp>();
      if (!cirFunc)
        return;

      unsigned numArgs = callOp.getNumOperands();
      unsigned numDataOps = numArgs - 1; // minus pure_fn
      unsigned numInputs = numArgs - 2;  // minus pure_fn and output

      SkeletonCallInfo info{cirFunc, callOp, opType,
                            pureFnRef.getRootReference().getValue(),
                            extractPreference(cirFunc), numInputs, numDataOps};
      worklist.push_back(info);
    });

    if (worklist.empty())
      return;

    // Phase 2: Rewrite functions and create skeleton ops.
    OpBuilder builder(&getContext());
    for (auto &info : worklist) {
      if (!info.cirFunc->getParentOp())
        continue;

      builder.setInsertionPoint(info.cirFunc);
      auto newFunc = rewriteToStandardFunc(info.cirFunc, builder);
      builder.setInsertionPointToStart(&newFunc.getBody().front());

      LogicalResult result = failure();
      if (info.opType == "map")
        result = lowerMapCall(info, newFunc, builder);
      else if (info.opType == "reduce")
        result = lowerReduceCall(info, newFunc, builder);
      else
        info.callOp.emitWarning("unknown skeleton op type '")
            << info.opType << "'";

      if (succeeded(result))
        builder.create<func::ReturnOp>(newFunc.getLoc());
    }

    // Erase original cir.func ops (now replaced by func.func ops).
    for (auto &info : worklist) {
      if (info.cirFunc && info.cirFunc->getParentOp())
        info.cirFunc.erase();
    }

    // Erase skeleton op declarations (external, no body).
    SmallVector<cir::FuncOp> toErase;
    module.walk([&](cir::FuncOp func) {
      if (func.isExternal() && func->hasAttr("skeleton.op"))
        toErase.push_back(func);
    });
    for (auto func : toErase)
      func.erase();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createCIRCallToSkeletonPass() {
  return std::make_unique<CIRCallToSkeletonPass>();
}
