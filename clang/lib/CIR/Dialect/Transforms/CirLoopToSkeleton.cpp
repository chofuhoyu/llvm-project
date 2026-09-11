//===- CirLoopToSkeleton.cpp - Annotated CIR loops to Skeleton -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Semi-automatic path: recognizes `cir.for` loops carrying the
// "skeleton.region" annotation and rewrites them into Skeleton dialect
// operations. The operator
// (map or reduce) is inferred from the loop shape, not named by the annotation.
//
// For an element-wise loop (`C[i] = f(A[i], ...)`) the pass:
//   1. validates that the loop is a canonical ascending unit-stride `cir.for`
//      over array elements with a straight-line scalar body,
//   2. extracts the body's scalar computation into a func.func pure function
//      (an arith body, via CirFuncToArith),
//   3. rewrites the enclosing host cir.func into a func.func taking one
//      `memref<?xelt>` per input array and returning the output
//      `tensor<?xelt>`,
//   4. emits a `skeleton.map` that references the extracted pure function.
//
// While/do-while loops, accumulation loops (reduce) and any non-canonical
// shape are reported and left untouched.
//
// The full manual path (skeleton helper function calls → skeleton ops) is
// implemented in CirCallToSkeleton.cpp.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/CirCallLowering.h"
#include "clang/CIR/Dialect/Transforms/CirFuncToArith.h"
#include "clang/CIR/Dialect/Transforms/CirScalarTypeConverter.h"
#include "clang/CIR/Dialect/Transforms/CirSkeletonAnnotations.h"

#include "llvm/ADT/STLExtras.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonAttrs.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonDialect.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_CIRLOOPTOSKELETON
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

/// What an element-wise (map) loop pattern carries: the loop itself, the
/// host function it lives in, the input arrays (host entry block arguments,
/// ordered by their parameter index) and the single output array.
struct MapPattern {
  cir::ForOp loop;
  cir::FuncOp host;
  SmallVector<BlockArgument> inputArgs;
  SmallVector<Type> inputCirTypes; // element CIR type per input
  BlockArgument outputArg;
  Type outputCirType; // element CIR type of the output
};

/// Reverse-trace a pointer value to the host function entry block argument it
/// was copied from, unwinding the `store -> alloca -> load` form clang's
/// codegen emits for function parameters. Returns null on failure.
static Value traceToHostValueImpl(Value v, cir::FuncOp host,
                                  DenseSet<Value> &visited) {
  if (!v || !visited.insert(v).second)
    return {};
  if (auto blockArg = dyn_cast<BlockArgument>(v))
    if (blockArg.getOwner()->getParentOp() == host.getOperation())
      return v;

  Operation *def = v.getDefiningOp();
  if (!def)
    return {};
  auto load = dyn_cast<cir::LoadOp>(def);
  if (!load)
    return {};
  if (!load.getAddr().getDefiningOp<cir::AllocaOp>())
    return {};
  // The address is a local alloca; its value is set by the stores feeding it.
  // Straight-line parameter copy: exactly one store carries the parameter.
  Value stored;
  for (OpOperand &use : load.getAddr().getUses()) {
    auto store = dyn_cast<cir::StoreOp>(use.getOwner());
    if (!store || use.getOperandNumber() != cir::StoreOp::odsIndex_addr)
      continue;
    if (stored)
      return {}; // more than one write: not a plain parameter copy
    stored = store.getValue();
  }
  if (!stored)
    return {};
  return traceToHostValueImpl(stored, host, visited);
}

static Value traceToHostValue(Value v, cir::FuncOp host) {
  DenseSet<Value> visited;
  return traceToHostValueImpl(v, host, visited);
}

/// Peel the unit-increment produced by `++i`: an operand of a `cir.inc`, or
/// the non-1 side of a `cir.add` with 1. Returns null when the value is not a
/// unit step.
static Value stripUnitIncrement(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return {};
  if (auto inc = dyn_cast<cir::IncOp>(def))
    return inc.getInput();
  if (auto add = dyn_cast<cir::AddOp>(def)) {
    auto isOne = [](Value side) {
      auto cst = side.getDefiningOp<cir::ConstantOp>();
      if (!cst)
        return false;
      auto intAttr = cst.getValueAttr<cir::IntAttr>();
      return intAttr && intAttr.getValue().isOne();
    };
    if (isOne(add.getLhs()))
      return add.getRhs();
    if (isOne(add.getRhs()))
      return add.getLhs();
  }
  return {};
}

/// True when \p stride is the loop induction variable (a load from its alloca)
/// possibly widened by integral casts — the offset shape a canonical
/// element-wise loop indexes arrays with.
static bool isArrayIndex(Value stride, Value ivAllocaAddr) {
  if (!stride)
    return false;
  if (auto load = stride.getDefiningOp<cir::LoadOp>())
    return load.getAddr() == ivAllocaAddr;
  auto castOp = stride.getDefiningOp<cir::CastOp>();
  if (!castOp || castOp.getKind() != cir::CastKind::integral)
    return false;
  return isArrayIndex(castOp.getSrc(), ivAllocaAddr);
}

/// Recognize the canonical ascending `for (i = 0; i < N; ++i)` shape and the
/// induction variable's alloca address. The bound must be one of the host's
/// parameters: the iteration count is later taken from the input tensor's
/// extent, so a bound that is anything else — a constant, or an expression
/// such as `N - 1` — would silently change how many elements are covered.
///
/// Requiring a parameter is necessary but not sufficient. Nothing here checks
/// that the caller passes the array length as that parameter, so a call such
/// as `f(3, A, B, C)` over a ten-element `A` is still rewritten to cover all
/// ten. CIR carries no link between a loop bound and an array's extent, so the
/// assumption cannot be discharged at this level.
static bool recognizeInduction(cir::ForOp loop, cir::FuncOp host,
                               Value &ivAllocaAddr) {
  // The condition region compares the induction variable against a bound.
  auto &condReg = loop.getCond();
  if (!condReg.hasOneBlock())
    return false;
  cir::CmpOp cmp;
  for (Operation &o : condReg.front()) {
    if (auto c = dyn_cast<cir::CmpOp>(&o))
      cmp = c;
  }
  if (!cmp)
    return false;

  // The step region stores iv+1 back into the induction variable's alloca.
  auto &stepReg = loop.getStep();
  if (!stepReg.hasOneBlock())
    return false;
  cir::StoreOp stepStore;
  for (Operation &o : stepReg.front())
    if (auto st = dyn_cast<cir::StoreOp>(&o))
      stepStore = st;
  if (!stepStore)
    return false;

  Value stepIn = stripUnitIncrement(stepStore.getValue());
  if (!stepIn)
    return false;
  auto stepLoad = stepIn.getDefiningOp<cir::LoadOp>();
  if (!stepLoad || stepLoad.getAddr() != stepStore.getAddr())
    return false;
  ivAllocaAddr = stepStore.getAddr();

  // The condition must count up while `i < bound` (either orientation). The
  // induction load in the condition region is a different SSA value from the
  // one in the step region, so compare the loaded addresses, not the values.
  auto loadsIv = [&](Value v) {
    auto load = v.getDefiningOp<cir::LoadOp>();
    return load && load.getAddr() == ivAllocaAddr;
  };
  Value bound;
  auto kind = cmp.getKind();
  if (loadsIv(cmp.getLhs()) && kind == cir::CmpOpKind::lt)
    bound = cmp.getRhs(); // i < N
  else if (loadsIv(cmp.getRhs()) && kind == cir::CmpOpKind::gt)
    bound = cmp.getLhs(); // N > i
  else
    return false;

  return static_cast<bool>(traceToHostValue(bound, host));
}

/// The loop must start from a constant zero stored into the induction variable
/// right before it (so the iterations cover the whole input tensor).
static bool hasZeroInit(cir::ForOp loop, Value ivAllocaAddr) {
  for (Operation &o : loop->getBlock()->getOperations()) {
    if (&o == loop.getOperation())
      break;
    auto store = dyn_cast<cir::StoreOp>(&o);
    if (!store || store.getAddr() != ivAllocaAddr)
      continue;
    auto cst = store.getValue().getDefiningOp<cir::ConstantOp>();
    if (!cst)
      return false;
    auto intAttr = cst.getValueAttr<cir::IntAttr>();
    return intAttr && intAttr.getValue().isZero();
  }
  return false;
}

/// True for the CIR scalar ops CirFuncToArith can translate (the ones a map
/// leaf body may clone).
static bool isScalarArithOp(Operation *op) {
  return isa<cir::ConstantOp, cir::AddOp, cir::SubOp, cir::MulOp, cir::DivOp,
             cir::RemOp, cir::FAddOp, cir::FSubOp, cir::FMulOp, cir::FDivOp,
             cir::FMulAddOp, cir::FNegOp, cir::MinusOp>(op);
}

/// Analyze an annotated loop as an element-wise (map) pattern. The loop body
/// must be a single straight-line block over array elements: reads are
/// `load(ptr_stride(base, i))` on host parameters, the body writes exactly one
/// such array element once, and everything else is translatable scalar
/// arithmetic.
static bool analyzeMapPattern(cir::ForOp loop, cir::FuncOp host,
                              MapPattern &pattern) {
  if (loop.maybeGetCleanup())
    return false;
  auto &bodyReg = loop.getBody();
  if (!bodyReg.hasOneBlock())
    return false;
  auto &body = bodyReg.front();

  Value ivAllocaAddr;
  if (!recognizeInduction(loop, host, ivAllocaAddr))
    return false;
  if (!hasZeroInit(loop, ivAllocaAddr))
    return false;

  // Collect reads (by host argument) and the single output, validating the
  // body op by op. `reads` maps a host argument to its element CIR type.
  DenseMap<BlockArgument, Type> reads;
  BlockArgument outputArg;

  bool sawStore = false;
  for (Operation &o : body) {
    // Body terminator.
    if (isa<cir::YieldOp>(&o))
      continue;

    // A value read: either an array element (host parameter indexed by i) or
    // the loop bookkeeping (induction variable / pointer handle reload).
    if (auto load = dyn_cast<cir::LoadOp>(&o)) {
      auto stride = load.getAddr().getDefiningOp<cir::PtrStrideOp>();
      if (!stride) {
        // Bookkeeping loads read an alloca (the iv or a pointer handle); only
        // those are tolerated.
        if (load.getAddr().getDefiningOp<cir::AllocaOp>())
          continue;
        return false;
      }
      if (!isArrayIndex(stride.getStride(), ivAllocaAddr))
        return false;
      Value base = traceToHostValue(stride.getBase(), host);
      if (!base)
        return false;
      auto baseArg = cast<BlockArgument>(base);
      reads[baseArg] = stride.getElementType();
      continue;
    }

    // An array store writes exactly one element once.
    if (auto store = dyn_cast<cir::StoreOp>(&o)) {
      if (sawStore)
        return false;
      auto stride = store.getAddr().getDefiningOp<cir::PtrStrideOp>();
      if (!stride)
        return false; // a store into a scalar alloca is an accumulator
      if (!isArrayIndex(stride.getStride(), ivAllocaAddr))
        return false;
      Value base = traceToHostValue(stride.getBase(), host);
      if (!base)
        return false;
      outputArg = cast<BlockArgument>(base);
      sawStore = true;
      continue;
    }

    // An index bookkeeping cast widens the induction variable for a stride
    // operand; any other cast is scalar conversion, out of scope.
    if (auto castOp = dyn_cast<cir::CastOp>(&o)) {
      if (castOp.getKind() != cir::CastKind::integral)
        return false;
      bool strideOnly = true;
      for (OpOperand &use : castOp.getResult().getUses()) {
        auto stride = dyn_cast<cir::PtrStrideOp>(use.getOwner());
        if (!stride || use.getOperandNumber() != 1) {
          strideOnly = false;
          break;
        }
      }
      if (!strideOnly)
        return false;
      continue;
    }
    // A pointer stride on its own (it is consumed by the load/store above).
    if (isa<cir::PtrStrideOp>(&o))
      continue;

    // Straight-line scalar arithmetic.
    if (isScalarArithOp(&o))
      continue;

    // Everything else (branches, calls, allocas, nesting, ...) is out of
    // scope.
    return false;
  }

  if (!sawStore || reads.empty())
    return false;

  // The output array must not also be read (no in-place element update), and
  // all arrays must share one element type.
  if (reads.count(outputArg))
    return false;
  auto outputCirType =
      cast<cir::PointerType>(outputArg.getType()).getPointee();
  SmallVector<BlockArgument> inputArgs;
  SmallVector<Type> inputCirTypes;
  for (auto &[arg, eltTy] : reads) {
    inputArgs.push_back(arg);
    inputCirTypes.push_back(eltTy);
  }
  for (Type t : inputCirTypes)
    if (t != outputCirType)
      return false;

  // Order inputs by their host parameter index for deterministic output.
  SmallVector<unsigned> order(inputArgs.size());
  for (unsigned i = 0; i != inputArgs.size(); ++i)
    order[i] = i;
  llvm::sort(order, [&](unsigned a, unsigned b) {
    return inputArgs[a].getArgNumber() < inputArgs[b].getArgNumber();
  });
  SmallVector<BlockArgument> sortedArgs;
  SmallVector<Type> sortedTypes;
  for (unsigned i : order) {
    sortedArgs.push_back(inputArgs[i]);
    sortedTypes.push_back(inputCirTypes[i]);
  }

  pattern.loop = loop;
  pattern.host = host;
  pattern.inputArgs = sortedArgs;
  pattern.inputCirTypes = sortedTypes;
  pattern.outputArg = outputArg;
  pattern.outputCirType = outputCirType;
  return true;
}

/// Extract the loop body's scalar computation into a func.func pure function
/// (an arith body), placed before \p host. The pure function takes one scalar
/// per input array element type and returns the output element.
static func::FuncOp buildPureLeaf(MapPattern &pattern, OpBuilder &builder,
                                  StringRef pureName) {
  Location loc = pattern.loop.getLoc();

  // Temporary CIR function carrying the extracted straight-line body.
  auto funcType =
      cir::FuncType::get(pattern.inputCirTypes, pattern.outputCirType,
                         /*isVarArg=*/false);
  auto tempName = (Twine("__skeleton_temp_") + pureName).str();
  auto temp = cir::FuncOp::create(builder, loc, tempName, funcType);
  temp.setSymVisibility("private");

  auto &tempBody = temp.getBody();
  while (!tempBody.empty())
    tempBody.front().erase();
  Block *entry = new Block();
  tempBody.push_back(entry);
  for (Type t : pattern.inputCirTypes)
    entry->addArgument(t, loc);

  IRMapping mapping;
  DenseMap<BlockArgument, unsigned> inputIndex;
  for (auto [i, arg] : llvm::enumerate(pattern.inputArgs))
    inputIndex[arg] = i;

  Value outputValue;
  auto &body = pattern.loop.getBody().front();
  bool failed = false;
  for (Operation &o : body) {
    builder.setInsertionPointToEnd(entry);
    if (isa<cir::YieldOp>(&o))
      continue;

    // Array-element load -> the corresponding pure function argument.
    if (auto load = dyn_cast<cir::LoadOp>(&o)) {
      auto stride = load.getAddr().getDefiningOp<cir::PtrStrideOp>();
      if (stride) {
        Value base = traceToHostValue(stride.getBase(), pattern.host);
        auto baseArg = cast<BlockArgument>(base);
        mapping.map(load.getResult(),
                    entry->getArgument(inputIndex.lookup(baseArg)));
      }
      continue; // bookkeeping loads are dropped
    }

    // The single output store: its value is the pure function's result.
    if (auto store = dyn_cast<cir::StoreOp>(&o)) {
      outputValue = store.getValue();
      continue;
    }

    // Index widening cast / ptr_stride: dropped (pure bookkeeping).
    if (isa<cir::CastOp, cir::PtrStrideOp>(&o))
      continue;

    // Straight-line scalar arithmetic: clone with operands resolved.
    bool opOk = true;
    for (Value operand : o.getOperands()) {
      if (!operand.getDefiningOp())
        opOk = false; // a host argument used directly in the body: unsupported
      if (opOk && !mapping.contains(operand))
        opOk = false; // value produced by a dropped op (e.g. the index)
    }
    if (!opOk) {
      failed = true;
      break;
    }
    builder.clone(o, mapping);
  }

  if (!failed && !outputValue)
    failed = true;

  if (failed) {
    temp.erase();
    return {};
  }

  builder.setInsertionPointToEnd(entry);
  auto retVal = mapping.lookupOrNull(outputValue);
  if (!retVal) {
    temp.erase();
    return {};
  }
  cir::ReturnOp::create(builder, loc, ValueRange(retVal));

  // Translate the temporary CIR function into an arith func.func.
  builder.setInsertionPoint(pattern.host);
  auto pureFn = createFuncFromCirBody(temp, pureName, builder);
  temp.erase();
  return pureFn;
}

/// Rewrite the host cir.func into a func.func taking one memref per input
/// array and returning the output tensor, with a skeleton.map in its body.
static func::FuncOp buildHostFunc(MapPattern &pattern, OpBuilder &builder,
                                  StringRef pureName,
                                  SkeletonPreference preference) {
  auto *ctx = builder.getContext();
  Location loc = pattern.host.getLoc();
  CirScalarTypeConverter converter;

  Type eltTy = converter.convertType(pattern.outputCirType);
  if (!eltTy)
    return {};

  SmallVector<Type> paramTypes;
  for (Type cirType : pattern.inputCirTypes) {
    Type mapped = converter.convertType(cirType);
    if (!mapped)
      return {};
    paramTypes.push_back(MemRefType::get({ShapedType::kDynamic}, mapped));
  }

  auto funcType =
      FunctionType::get(ctx, paramTypes,
                        {outputTensorType(SkeletonOpType::Map, eltTy)});
  auto hostFunc = func::FuncOp::create(builder, loc,
                                       pattern.host.getSymName(), funcType);
  hostFunc.setSymVisibility(pattern.host.getSymVisibility());

  auto *entryBlock = hostFunc.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  SmallVector<Value> inputTensors;
  for (Value param : entryBlock->getArguments()) {
    inputTensors.push_back(createToTensor(builder, loc, param));
  }
  Value init = createEmptyOutput(builder, loc, hostFunc, inputTensors.front());

  auto pureFnAttr = SymbolRefAttr::get(ctx, pureName);
  auto prefAttr = skeleton::PreferenceAttr::get(
      ctx, StringAttr::get(ctx, toString(preference)));
  auto mapOp = skeleton::MapOp::create(
      builder, loc, init.getType(), inputTensors, init, pureFnAttr, prefAttr);

  func::ReturnOp::create(builder, loc, ValueRange(mapOp.getResult()));
  return hostFunc;
}

/// Drop the CIR codegen attributes clang leaves on the module (the `cir.*`
/// family and the dlti data-layout spec), which name the CIR dialect that the
/// downstream MLIR tools consuming a skeleton module do not register.
static void dropCirModuleAttrs(ModuleOp module) {
  SmallVector<StringAttr> toDrop;
  for (const NamedAttribute &attr : module->getDiscardableAttrs()) {
    StringRef name = attr.getName().strref();
    if (name.starts_with("cir.") || name.starts_with("dlti."))
      toDrop.push_back(attr.getName());
  }
  for (StringAttr name : toDrop)
    module->removeAttr(name);
}

} // namespace

namespace {

class CIRLoopToSkeletonPass
    : public impl::CIRLoopToSkeletonBase<CIRLoopToSkeletonPass> {
public:
  using CIRLoopToSkeletonBase::CIRLoopToSkeletonBase;

  void runOnOperation() override {
    auto module = getOperation();

    // Phase A: find annotated loops, group them by host.
    SmallVector<cir::ForOp> annotatedLoops;
    module.walk([&](cir::ForOp forOp) {
      if (hasSkeletonRegionAnnotation(forOp))
        annotatedLoops.push_back(forOp);
    });
    if (annotatedLoops.empty())
      return;

    DenseMap<cir::FuncOp, SmallVector<cir::ForOp>> loopsByHost;
    for (cir::ForOp loop : annotatedLoops) {
      auto host = loop->getParentOfType<cir::FuncOp>();
      if (!host)
        continue;
      loopsByHost[host].push_back(loop);
    }

    struct HostRewrite {
      cir::FuncOp host;
      MapPattern pattern;
      SkeletonPreference preference;
    };
    SmallVector<HostRewrite> rewrites;
    bool badPreference = false;

    for (auto &kv : loopsByHost) {
      cir::FuncOp host = kv.first;
      if (kv.second.size() != 1) {
        host.emitWarning() << "host with multiple skeleton.region annotated "
                              "loops is not supported by the semi-automatic "
                              "path";
        continue;
      }
      cir::ForOp loop = kv.second.front();

      auto preference = getSkeletonPreference(loop);
      if (failed(preference)) {
        badPreference = true;
        continue;
      }

      MapPattern pattern;
      if (!analyzeMapPattern(loop, host, pattern)) {
        loop.emitWarning() << "semi-automatic path: this skeleton.region "
                              "annotated for loop does not match the supported "
                              "element-wise (map) shape; only a canonical "
                              "`for (i = 0; i < N; ++i)` over array elements "
                              "with a straight-line body is supported";
        continue;
      }
      rewrites.push_back({host, pattern, *preference});
    }

    if (badPreference) {
      signalPassFailure();
      return;
    }
    if (rewrites.empty())
      return; // nothing convertible: leave the module as it is

    // Phase B: rewrite each convertible host.
    unsigned pureCounter = 0;
    auto freshPureName = [&]() {
      return SymbolTable::generateSymbolName<32>(
          "__skeleton_pure",
          [&](StringRef name) {
            return static_cast<bool>(
                SymbolTable::lookupSymbolIn(module.getOperation(), name));
          },
          pureCounter);
    };

    OpBuilder builder(&getContext());
    for (HostRewrite &rewrite : rewrites) {
      builder.setInsertionPoint(rewrite.host);
      auto pureName = freshPureName();
      auto pureFn =
          buildPureLeaf(rewrite.pattern, builder, pureName);
      if (!pureFn)
        continue; // analysis already guaranteed the shape; keep host
      builder.setInsertionPoint(rewrite.host);
      if (!buildHostFunc(rewrite.pattern, builder, pureName,
                         rewrite.preference)) {
        pureFn.erase();
        continue;
      }
      rewrite.host.erase();
    }

    // Phase C: the module must be whole-skeleton by now; refuse loudly
    // otherwise (see CirCallToSkeleton for the same whole-module constraint).
    SmallVector<cir::FuncOp> leftovers;
    module.walk([&](cir::FuncOp func) { leftovers.push_back(func); });
    if (!leftovers.empty()) {
      for (cir::FuncOp func : leftovers)
        func.emitError() << "non-skeleton function " << func.getSymName()
                         << " is left in CIR: cir-loop-to-skeleton requires the "
                            "whole module to be skeleton code (annotated "
                            "element-wise loops and their hosts), surrounding "
                            "ordinary C++ code is not supported yet";
      signalPassFailure();
      return;
    }

    dropCirModuleAttrs(module);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createCIRLoopToSkeletonPass() {
  return std::make_unique<CIRLoopToSkeletonPass>();
}
