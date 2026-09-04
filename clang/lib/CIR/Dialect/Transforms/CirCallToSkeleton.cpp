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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonAttrs.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonDialect.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
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

/// Converts CIR scalar types to the standard MLIR scalar types that skeleton
/// ops (and the arith bodies they inline) accept. Unsupported CIR types
/// convert to a null type, so callers emit a diagnostic instead of silently
/// lowering to the wrong type. Newly supported types are added here as
/// additional conversions rather than an if/else chain.
class CirScalarTypeConverter : public mlir::TypeConverter {
public:
  CirScalarTypeConverter() {
    addConversion([](cir::IntType ty) -> std::optional<Type> {
      // Project to a signless MLIR integer (the arith convention); signedness
      // is a property of the CIR arithmetic that produced the value, not of
      // the integer type the skeleton layer consumes.
      return IntegerType::get(ty.getContext(), ty.getWidth());
    });
    addConversion([](cir::SingleType ty) -> std::optional<Type> {
      return Float32Type::get(ty.getContext());
    });
    addConversion([](cir::DoubleType ty) -> std::optional<Type> {
      return Float64Type::get(ty.getContext());
    });
    addConversion([](cir::FP16Type ty) -> std::optional<Type> {
      return Float16Type::get(ty.getContext());
    });
    addConversion([](cir::BF16Type ty) -> std::optional<Type> {
      return BFloat16Type::get(ty.getContext());
    });
  }
};

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
static LogicalResult
convertPureFunctionsToFunc(ModuleOp module,
                           const DenseSet<StringRef> &pureFns) {
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
  // Number of data operands in the call. Return-value style: every data
  // operand is an input; the output is a fresh tensor returned from the
  // function, not a caller-supplied buffer.
  unsigned numInputs;
};

// Phase 2: Rewrite cir.func → func.func and create skeleton ops

/// Create bufferization.to_tensor from a memref.
static Value createToTensor(OpBuilder &builder, Location loc, Value memref) {
  auto memrefType = cast<MemRefType>(memref.getType());
  auto tensorType =
      RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
  return bufferization::ToTensorOp::create(builder, loc, tensorType, memref,
                                           /*restrict=*/true,
                                           /*writable=*/true);
}

/// The skeleton op's output tensor type, determined by the op's semantics.
/// map is element-wise, so its output is 1-D with a dynamic extent matching
/// the input; reduce collapses its input to a rank-0 scalar. The element type
/// equals the input's (map/reduce verifiers require output elt == input elt).
/// Future operators add their own output-shape rule here.
static RankedTensorType outputTensorType(StringRef opType, Type eltTy) {
  if (opType == "reduce")
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
/// Non-pointer parameters map to their scalar type via cirTypeToStdType.
///
/// The original body is discarded and the function is given a return type by
/// this pass rather than read from the original cir.func. That return type is
/// the skeleton op's result tensor (map/reduce's `outs` type): shape from the
/// op semantics, element type from the first pointer's pointee, since the op
/// verifiers require the output elt to equal the input's. This is only sound
/// while the host body really is just the one skeleton call; other statements
/// would be silently dropped and several skeleton calls would each emit a
/// same-named func.func (see review doc front-end follow-ups).
static func::FuncOp rewriteToStandardFunc(cir::FuncOp cirFunc,
                                          OpBuilder &rewriter,
                                          StringRef opType) {
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

/// Process a map call: create a skeleton.map whose result is the function's
/// return value. Returns the skeleton op result (nullptr on failure).
static Value lowerMapCall(SkeletonCallInfo &info, func::FuncOp newFunc,
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

  // The call has: arg0=pure_fn, arg1..argN = inputs. We need numInputs
  // memrefs; all data operands are inputs (return-value style).
  if (memrefArgs.size() < info.numInputs) {
    info.callOp.emitWarning(
        "not enough memref arguments in rewritten function");
    return {};
  }

  // Create to_tensor for each input memref.
  SmallVector<Value> inputTensors;
  for (unsigned i = 0; i < info.numInputs; ++i)
    inputTensors.push_back(createToTensor(builder, loc, memrefArgs[i]));

  // Output is a fresh tensor.empty (the outs/destination), not a
  // caller-supplied buffer.
  Value init = createEmptyOutput(builder, loc, newFunc, inputTensors[0]);

  auto pureFnAttr = SymbolRefAttr::get(ctx, info.pureFnName);
  auto prefAttr =
      skeleton::PreferenceAttr::get(ctx, StringAttr::get(ctx, info.preference));

  auto mapOp = skeleton::MapOp::create(
      builder, loc, init.getType(), inputTensors, init, pureFnAttr, prefAttr);

  return mapOp.getResult();
}

/// Process a reduce call: create a skeleton.reduce whose result (a rank-0
/// tensor) is the function's return value. Returns the skeleton op result
/// (nullptr on failure).
static Value lowerReduceCall(SkeletonCallInfo &info, func::FuncOp newFunc,
                             OpBuilder &builder) {
  auto loc = info.callOp.getLoc();
  auto *ctx = builder.getContext();

  // TODO: Same positional memref-arg matching assumption as lowerMapCall.
  SmallVector<Value> memrefArgs;
  for (unsigned i = 0; i < newFunc.getNumArguments(); ++i) {
    if (isa<MemRefType>(newFunc.getArgument(i).getType()))
      memrefArgs.push_back(newFunc.getArgument(i));
  }

  if (memrefArgs.size() < info.numInputs) {
    info.callOp.emitWarning("not enough memref arguments");
    return {};
  }

  Value inputMemref = memrefArgs[0];
  auto inputTensor = createToTensor(builder, loc, inputMemref);

  // Output is a fresh rank-0 tensor.empty.
  Value init = createEmptyOutput(builder, loc, newFunc, inputTensor);

  auto pureFnAttr = SymbolRefAttr::get(ctx, info.pureFnName);
  auto prefAttr =
      skeleton::PreferenceAttr::get(ctx, StringAttr::get(ctx, info.preference));

  auto reduceOp = skeleton::ReduceOp::create(
      builder, loc, init.getType(), inputTensor, init, pureFnAttr, prefAttr);

  return reduceOp.getResult();
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
      unsigned numInputs = numArgs - 1; // minus pure_fn

      SkeletonCallInfo info{cirFunc,
                            callOp,
                            opType,
                            pureFnRef.getRootReference().getValue(),
                            extractPreference(cirFunc),
                            numInputs};
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

      Value result;
      if (info.opType == "map")
        result = lowerMapCall(info, newFunc, builder);
      else if (info.opType == "reduce")
        result = lowerReduceCall(info, newFunc, builder);
      else
        info.callOp.emitWarning("unknown skeleton op type '")
            << info.opType << "'";

      if (result)
        func::ReturnOp::create(builder, newFunc.getLoc(), ValueRange(result));
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
