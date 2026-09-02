//===- CirCallToSkeleton.cpp - CIR calls to Skeleton ops --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Full manual path: Recognizes calls to skeleton helper functions (declared
// with the "skeleton.op" annotation in headers) and replaces them with
// Skeleton dialect structured computation operations.
//
// Semi-automatic path (TODO): annotated for-loops with inline computation.
// Lambda support (TODO): captureless lambdas as pure_fn references.
//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonAttrs.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonDialect.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
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

// Metadata collection (Phase 1: gather info from original CIR IR)

/// Find a cir.annotation with the given name on an operation's annotations
/// attribute. Returns an empty attribute when not found.
static cir::AnnotationAttr getAnnotationByName(Operation *op, StringRef name) {
  auto annotations = op->getAttrOfType<ArrayAttr>("annotations");
  if (!annotations)
    return {};
  for (Attribute attr : annotations) {
    auto ann = dyn_cast<cir::AnnotationAttr>(attr);
    if (ann && ann.getName().getValue() == name)
      return ann;
  }
  return {};
}

/// Collect function names marked with the "skeleton.pure" annotation.
static DenseSet<StringRef> collectPureFunctions(ModuleOp module) {
  DenseSet<StringRef> pureFns;
  module.walk([&](cir::FuncOp func) {
    if (getAnnotationByName(func, "skeleton.pure"))
      pureFns.insert(func.getSymName());
  });
  return pureFns;
}

/// Collect skeleton op declarations: external functions carrying the
/// "skeleton.op" annotation. Returns map: function name → op type
/// ("map", "reduce"), taken from the annotation's first argument.
static DenseMap<StringRef, StringRef> collectSkeletonOpDecls(ModuleOp module) {
  DenseMap<StringRef, StringRef> opDecls;
  module.walk([&](cir::FuncOp func) {
    if (!func.isExternal())
      return;
    auto ann = getAnnotationByName(func, "skeleton.op");
    if (!ann)
      return;
    StringRef opType;
    if (auto args = ann.getArgs())
      if (!args.empty())
        if (auto arg = dyn_cast<StringAttr>(args[0]))
          opType = arg.getValue();
    // TODO: Validate opType against the set of supported operators ("map",
    // "reduce") here. An unknown type currently flows all the way into the
    // rewrite phase, where it leaves an empty func.func and only surfaces as
    // the "unknown skeleton op type" warning at dispatch time. The supported
    // set should live in one place shared with outputMemrefShape and the
    // map/reduce dispatch below.
    if (!opType.empty())
      opDecls[func.getSymName()] = opType;
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

/// Extract the preference string from the "skeleton.region" annotation on a
/// cir.func. Falls back to "CPU" when the annotation is absent.
static std::string extractPreference(cir::FuncOp func) {
  if (auto ann = getAnnotationByName(func, "skeleton.region"))
    if (auto args = ann.getArgs())
      if (!args.empty())
        if (auto pref = dyn_cast<StringAttr>(args[0]))
          return pref.getValue().str();
  return "CPU";
}

/// Map a CIR scalar type to standard MLIR type. Returns a null type for
/// unsupported CIR types so callers emit a diagnostic instead of silently
/// lowering to the wrong type.
/// TODO: Replace this with a more robust type converter covering all CIR
/// scalar types.
static Type cirTypeToStdType(Type cirTy, MLIRContext *ctx) {
  if (cirTy.isInteger(32))
    return IntegerType::get(ctx, 32);
  // cir.float → f32, cir.double → f64
  if (cirTy.isF32() || isa<cir::SingleType>(cirTy))
    return Float32Type::get(ctx);
  if (cirTy.isF64() || isa<cir::DoubleType>(cirTy))
    return Float64Type::get(ctx);
  return Type();
}

// TODO: This drops the pure function's body and keeps only its signature.
// That satisfies the skeleton op verifier (pure_fn must resolve to a func.func
// symbol), but it stops the manual path at skeleton IR: SkeletonToLinalg needs
// pure_fn to carry a real body, which it clones into the linalg region and
// rejects with "pure_fn ... must be defined" when the function is only a
// declaration. So a user's pure function (which does have a body in CIR)
// cannot yet run through to linalg/execution via this path.
//
// Root cause: no CIR → standard-MLIR function lowering exists to reuse.
// ClangIR keeps its own op set (cir.fadd, ...) and lowers straight to LLVM
// (DirectToLLVM), so nothing upstream turns a CIR body into the arith body a
// linalg region can accept. CirCallToSkeleton is the bridge between CIR and
// the standard-MLIR world skeleton lives in, so the translation has to live
// here (or in shared CIR tooling).
//
// Options when picked up:
//   - Translate single-block, straight-line scalar pure_fn bodies from CIR to
//     arith (constants, scalar arith, return) — the shape SkeletonToLinalg
//     already requires.
//   - Or recognize simple bodies (e.g. add) in CIR and emit named ops
//     directly.
//   - CirLoopToSkeleton needs the same CIR→func.func translation for the
//     loop bodies it extracts; share the tooling between both paths.
// Until then, pure functions that actually carry a body are rejected here
// with an explicit diagnostic instead of being silently erased.

/// Convert pure functions from cir.func to func.func declarations so that
/// skeleton ops can reference them via pure_fn. Only declaration-only pure
/// functions are supported; a pure function that carries a body is rejected,
/// because translating its CIR body to a func.func body is not implemented
/// yet (see the TODO above).
static LogicalResult convertPureFunctionsToFunc(
    ModuleOp module, const DenseSet<StringRef> &pureFns) {
  SmallVector<cir::FuncOp> toConvert;
  module.walk([&](cir::FuncOp func) {
    if (!pureFns.contains(func.getSymName()))
      return;
    if (!func.isExternal()) {
      func.emitError()
          << "pure function '" << func.getSymName()
          << "' has a body; translating its CIR body to func.func is not "
             "implemented yet";
      return;
    }
    toConvert.push_back(func);
  });

  auto *ctx = module.getContext();
  for (cir::FuncOp cirFunc : toConvert) {
    SmallVector<Type> inputs, results;
    bool unsupported = false;
    for (Type t : cirFunc.getArgumentTypes()) {
      Type mapped = cirTypeToStdType(t, ctx);
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
      Type mapped = cirTypeToStdType(t, ctx);
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
    cirFunc.erase();
  }
  return success();
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

// Phase 2: Rewrite cir.func → func.func and create skeleton ops

/// Create bufferization.to_tensor from a memref.
static Value createToTensor(OpBuilder &builder, Location loc, Value memref) {
  auto memrefType = cast<MemRefType>(memref.getType());
  auto tensorType = RankedTensorType::get(memrefType.getShape(),
                                          memrefType.getElementType());
  return bufferization::ToTensorOp::create(builder, loc, tensorType, memref,
                                           /*restrict=*/true,
                                           /*writable=*/true);
}

/// Create bufferization.materialize_in_destination.
static void createMaterializeInDest(OpBuilder &builder, Location loc,
                                     Value tensor, Value memref) {
  bufferization::MaterializeInDestinationOp::create(
      builder, loc, /*result=*/Type(), /*source=*/tensor, /*dest=*/memref,
      /*restrict=*/false, /*writable=*/true);
}

/// The output memref shape for a skeleton op, determined by the op's
/// semantics. map is element-wise, so its output matches the 1-D input shape;
/// reduce collapses its input to a rank-0 scalar. Future operators add their
/// own output-shape rule here.
static SmallVector<int64_t> outputMemrefShape(StringRef opType) {
  if (opType == "reduce")
    return {};
  return {ShapedType::kDynamic};
}

/// Rewrite a cir.func to a func.func with memref parameters.
/// Returns the new func.func.
///
/// Every pointer parameter becomes a memref whose element type is the
/// pointer's pointee. Input pointers are 1-D with a dynamic extent (a bare C
/// pointer does not carry its length); the output pointer (the last one) gets
/// the op-specific shape from outputMemrefShape. Non-pointer parameters map to
/// their scalar type via cirTypeToStdType, and the original body is discarded.
static func::FuncOp rewriteToStandardFunc(cir::FuncOp cirFunc,
                                           OpBuilder &rewriter,
                                           StringRef opType) {
  auto loc = cirFunc.getLoc();
  auto *ctx = rewriter.getContext();

  unsigned numPtrArgs = 0;
  for (unsigned i = 0; i < cirFunc.getNumArguments(); ++i)
    if (isa<cir::PointerType>(cirFunc.getArgument(i).getType()))
      ++numPtrArgs;

  SmallVector<Type> newInputTypes;
  unsigned ptrIdx = 0;
  for (unsigned i = 0; i < cirFunc.getNumArguments(); ++i) {
    Type argTy = cirFunc.getArgument(i).getType();
    if (auto ptrTy = dyn_cast<cir::PointerType>(argTy)) {
      auto eltTy = cirTypeToStdType(ptrTy.getPointee(), ctx);
      if (!eltTy) {
        cirFunc.emitError() << "unsupported CIR type in skeleton function: "
                            << ptrTy.getPointee();
        return nullptr;
      }
      bool isOutput = (ptrIdx == numPtrArgs - 1);
      ++ptrIdx;
      if (isOutput)
        newInputTypes.push_back(
            MemRefType::get(outputMemrefShape(opType), eltTy));
      else
        newInputTypes.push_back(
            MemRefType::get({ShapedType::kDynamic}, eltTy));
    } else {
      auto scalarTy = cirTypeToStdType(argTy, ctx);
      if (!scalarTy) {
        cirFunc.emitError() << "unsupported CIR type in skeleton function: "
                            << argTy;
        return nullptr;
      }
      newInputTypes.push_back(scalarTy);
    }
  }

  auto funcType = FunctionType::get(ctx, newInputTypes, {});

  auto newFunc =
      func::FuncOp::create(rewriter, loc, cirFunc.getSymName(), funcType);
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
  auto *ctx = builder.getContext();

  // Map call args to new function's memref block arguments.
  // We need to find the memref block args that correspond to the original
  // cir.ptr params. In the rewritten function, all cir.ptr<T> params become
  // memref<?xT>. We walk all memref block args and assign them positionally.
  // TODO: This positional matching assumes the rewritten function's memref
  // params appear in the same order as the skeleton call's data operands,
  // and that the function has no other pointer params — an extra or
  // reordered param silently binds the wrong operand. Recover the
  // correspondence from the original call's argument values instead.
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

  auto mapOp =
      skeleton::MapOp::create(builder, loc, outputTensor.getType(),
                              inputTensors, outputTensor, pureFnAttr, prefAttr);

  createMaterializeInDest(builder, loc, mapOp.getResult(), outputMemref);
  return success();
}

/// Process a reduce call.
static LogicalResult lowerReduceCall(SkeletonCallInfo &info,
                                      func::FuncOp newFunc,
                                      OpBuilder &builder) {
  auto loc = info.callOp.getLoc();
  auto *ctx = builder.getContext();

  // TODO: Same positional memref-arg matching assumption as lowerMapCall.
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

  auto reduceOp = skeleton::ReduceOp::create(
      builder, loc, outputTensor.getType(), inputTensor, outputTensor,
      pureFnAttr, prefAttr);

  createMaterializeInDest(builder, loc, reduceOp.getResult(), outputMemref);
  return success();
}

// Pass

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

    // Convert pure functions from cir.func to func.func so that the skeleton
    // ops created below can reference them via pure_fn. Fail when a pure
    // function carries a body we cannot translate (see the TODO above);
    // continuing would silently drop it.
    if (failed(convertPureFunctionsToFunc(module, pureFns))) {
      signalPassFailure();
      return;
    }

    // Phase 2: Rewrite functions and create skeleton ops.
    OpBuilder builder(&getContext());
    for (auto &info : worklist) {
      if (!info.cirFunc->getParentOp())
        continue;

      builder.setInsertionPoint(info.cirFunc);
      auto newFunc = rewriteToStandardFunc(info.cirFunc, builder, info.opType);
      if (!newFunc)
        continue;
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
        func::ReturnOp::create(builder, newFunc.getLoc());
    }

    // Erase original cir.func ops (now replaced by func.func ops).
    for (auto &info : worklist) {
      if (info.cirFunc && info.cirFunc->getParentOp())
        info.cirFunc.erase();
    }

    // Erase skeleton op declarations (external, no body).
    SmallVector<cir::FuncOp> toErase;
    module.walk([&](cir::FuncOp func) {
      if (func.isExternal() && getAnnotationByName(func, "skeleton.op"))
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
