//===- CirSkeletonSplit.cpp - split skeleton hosts into operator units ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The front end turns a whole C++ program into CIR. This pass is where an
// operator leaves the host: a function marked with the "skeleton.region"
// annotation whose body is a single call to a skeleton helper keeps its symbol
// and its C++ signature, and grows two things next to it:
//
//   __skeleton_op_<id>(<memref>..., <scalar>...) -> ()
//       the operator unit: the skeleton operator itself, on memrefs, with no
//       return value. The driver compiles it for a device.
//
//   __skeleton_op_<id>_shim(ptr, i64..., <scalar>...) -> ()
//       the ABI thunk, a declaration here. `skeleton-lower-host-abi` fills its
//       body once the host has been lowered to the LLVM dialect: it turns the
//       pointer/length pairs into memref descriptors and calls the unit.
//
// The host keeps calling convention `void f(...)` and its body becomes a call to
// the thunk, so every C++ caller is unaffected and the placement decision stays
// at the call site. The unit's destination is the first data argument of the
// helper call, which is what makes the boundary destination-passing: the caller
// owns the buffer, the unit neither allocates nor frees.
//
// Everything else in the module -- `main`, ordinary functions, the pure
// functions the units reference -- is left for `cir-to-llvm`.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/CirCallAnalysis.h"
#include "clang/CIR/Dialect/Transforms/CirCallLowering.h"
#include "clang/CIR/Dialect/Transforms/CirFuncToArith.h"
#include "clang/CIR/Dialect/Transforms/CirScalarTypeConverter.h"
#include "clang/CIR/Dialect/Transforms/CirSkeletonAnnotations.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonAttrs.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonDialect.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_CIRSKELETONSPLIT
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

/// Symbols the split creates next to a host: the operator unit and the ABI
/// thunk that stands between the host and it.
constexpr StringLiteral kUnitPrefix = "__skeleton_op_";
constexpr StringLiteral kThunkSuffix = "_shim";

/// Attributes the split leaves behind for the driver and the ABI lowering.
/// `skeleton.op_id` / `skeleton.bytes` / `skeleton.preference` are the unit's
/// placement metadata, the same ones `skeleton-outline` puts on units it
/// outlines. `skeleton.host_abi` is the thunk's description: which unit to
/// call and which memref each pointer/length pair denotes.
constexpr StringLiteral kOpIdAttr = "skeleton.op_id";
constexpr StringLiteral kBytesAttr = "skeleton.bytes";
constexpr StringLiteral kPreferenceAttr = "skeleton.preference";
constexpr StringLiteral kHostABIAttr = "skeleton.host_abi";
/// Set on a unit whose data arguments are runtime buffers. Device lowering
/// leaves such a unit's operands alone: they are already memory the kernel can
/// reach, and giving each of them a device allocation plus a copy would defeat
/// the point of allocating them from the runtime.
constexpr StringLiteral kManagedAttr = "skeleton.managed";

/// Number of bytes the unit moves through its data arguments, when every one of
/// them has a static shape. 0 means "not known at compile time", which is what
/// the runtime expects for an unknown working set.
int64_t computeUnitBytes(ArrayRef<MemRefType> dataTypes) {
  int64_t total = 0;
  for (MemRefType type : dataTypes) {
    if (!type.hasStaticShape() || !type.getElementType().isIntOrFloat())
      return 0;
    total += type.getElementType().getIntOrFloatBitWidth() / 8 *
             type.getNumElements();
  }
  return total;
}

/// The value of a scalar argument of the skeleton call, converted to the type
/// the unit takes.
Type scalarUnitType(Type cirType) {
  CirScalarTypeConverter converter;
  return converter.convertType(cirType);
}

/// `cir.load` with every optional attribute left at its default.
Value loadValue(OpBuilder &builder, Location loc, Type type, Value addr) {
  return cir::LoadOp::create(builder, loc, type, addr, /*isDeref=*/false,
                             /*is_volatile=*/false, /*is_nontemporal=*/false,
                             /*alignment=*/IntegerAttr(),
                             /*sync_scope=*/cir::SyncScopeKindAttr(),
                             /*mem_order=*/cir::MemOrderAttr(),
                             /*invariant=*/false)
      .getResult();
}

/// The ops a thin skeleton host may contain besides its one skeleton call:
/// parameter staging and constants. Anything else would be silently dropped
/// when the body is replaced by the thunk call, so it is reported instead.
bool isThinWrapperOp(Operation *op, cir::CallOp skeletonCall) {
  if (op == skeletonCall)
    return true;
  return isa<cir::AllocaOp, cir::StoreOp, cir::LoadOp, cir::GetMemberOp,
             cir::CastOp, cir::ConstantOp, cir::GetGlobalOp, cir::ReturnOp>(op);
}

class CIRSkeletonSplitPass
    : public impl::CIRSkeletonSplitBase<CIRSkeletonSplitPass> {
public:
  using CIRSkeletonSplitBase::CIRSkeletonSplitBase;

  void runOnOperation() override;

private:
  /// Create the operator unit for \p info and return it.
  func::FuncOp createUnit(SkeletonCallInfo &info, int64_t id);

  /// Declare the ABI thunk for \p unit and return the declaration.
  cir::FuncOp createThunk(SkeletonCallInfo &info, int64_t id,
                          func::FuncOp unit);

  /// Replace the host's body with a call to \p thunk.
  LogicalResult rewriteHost(SkeletonCallInfo &info, cir::FuncOp thunk);

  OpBuilder *builder = nullptr;
  ModuleOp module;
};

/// Rewrite the host body: every data argument becomes the pointer (and, for a
/// span, the length) its parameter holds, the scalar arguments are passed
/// through, and the skeleton call is replaced by a call to the thunk.
LogicalResult CIRSkeletonSplitPass::rewriteHost(SkeletonCallInfo &info,
                                                cir::FuncOp thunk) {
  cir::FuncOp host = info.cirFunc;
  Location loc = info.callOp.getLoc();
  Block &entry = host.getBody().front();

  // The original ops are erased once the new call is in place. They are
  // collected first because the rewrite reads values they define.
  SmallVector<Operation *> originalOps;
  for (Operation &op : entry)
    originalOps.push_back(&op);

  Operation *oldCall = info.callOp.getOperation();
  Operation *terminator = entry.getTerminator();
  builder->setInsertionPoint(terminator);

  auto voidPtrTy = cir::PointerType::get(cir::VoidType::get(&getContext()));
  SmallVector<Value> args;
  for (const DataOperandRef &data : info.dataOperands) {
    Value param = host.getArgument(data.hostParam);
    if (data.hasRuntimeExtent) {
      // A span is {data, size}: read both members from the parameter the
      // caller passed by reference.
      auto spanPtrTy = cast<cir::PointerType>(param.getType());
      auto spanTy = cast<cir::StructType>(spanPtrTy.getPointee());
      Type dataMemberTy = spanTy.getElementType(kSpanDataMember);
      Type sizeMemberTy = spanTy.getElementType(kSpanSizeMember);
      auto dataPtr = cir::GetMemberOp::create(
          *builder, loc, cir::PointerType::get(dataMemberTy), param, "",
          kSpanDataMember);
      Value base = loadValue(*builder, loc, dataMemberTy, dataPtr.getResult());
      auto sizePtr = cir::GetMemberOp::create(
          *builder, loc, cir::PointerType::get(sizeMemberTy), param, "",
          kSpanSizeMember);
      Value size = loadValue(*builder, loc, sizeMemberTy, sizePtr.getResult());
      args.push_back(cir::CastOp::create(*builder, loc, voidPtrTy,
                                         cir::CastKind::bitcast, base));
      args.push_back(size);
      continue;
    }
    // An array reference or a single-element destination: the parameter is
    // already the address of the buffer.
    args.push_back(cir::CastOp::create(*builder, loc, voidPtrTy,
                                       cir::CastKind::bitcast, param));
  }
  for (const ScalarOperandRef &scalar : info.scalarOperands) {
    Value value = info.callOp.getOperand(scalar.callOperand);
    if (scalar.byReference)
      value = loadValue(*builder, loc, scalar.cirType, value);
    args.push_back(value);
  }

  cir::CallOp::create(*builder, loc,
                      FlatSymbolRefAttr::get(&getContext(), thunk.getSymName()),
                      cir::VoidType::get(&getContext()), ValueRange(args));
  cir::ReturnOp::create(*builder, loc, ValueRange{});

  // Drop the old body: the skeleton call and the return are replaced, and the
  // staging ops whose results nothing uses any more are dead. Only ops the thin
  // wrapper check allowed are candidates, so no user statement can go with
  // them. Ops without results stay: a store into a parameter slot is what
  // initializes a value a rewritten call reads back, and an unused one costs
  // nothing once the host is lowered.
  oldCall->erase();
  terminator->erase();
  for (Operation *op : llvm::reverse(originalOps)) {
    if (op == oldCall || op == terminator || op->getBlock() != &entry)
      continue;
    if (op->getNumResults() && op->use_empty())
      op->erase();
  }
  return success();
}

func::FuncOp CIRSkeletonSplitPass::createUnit(SkeletonCallInfo &info,
                                              int64_t id) {
  MLIRContext *ctx = &getContext();
  Location loc = info.callOp.getLoc();

  SmallVector<Type> argTypes;
  SmallVector<MemRefType> dataTypes;
  for (const DataOperandRef &data : info.dataOperands) {
    argTypes.push_back(data.type);
    dataTypes.push_back(data.type);
  }
  for (const ScalarOperandRef &scalar : info.scalarOperands)
    argTypes.push_back(scalarUnitType(scalar.cirType));

  auto funcType = FunctionType::get(ctx, argTypes, {});
  std::string name = (kUnitPrefix + Twine(id)).str();
  builder->setInsertionPointToEnd(module.getBody());
  auto unit = func::FuncOp::create(*builder, loc, name, funcType);
  unit.setPrivate();
  unit->setAttr(kOpIdAttr, builder->getI64IntegerAttr(id));
  unit->setAttr(kBytesAttr, builder->getI64IntegerAttr(computeUnitBytes(dataTypes)));
  unit->setDiscardableAttr(
      kPreferenceAttr,
      skeleton::PreferenceAttr::get(
          ctx, StringAttr::get(ctx, toString(info.preference))));
  if (info.managed)
    unit->setAttr(kManagedAttr, UnitAttr::get(ctx));

  Block *entry = unit.addEntryBlock();
  OpBuilder body = OpBuilder::atBlockBegin(entry);
  unsigned argIdx = 0;

  Value dstMemref = entry->getArgument(argIdx++);

  auto pureFnAttr = SymbolRefAttr::get(ctx, info.pureFnName);
  auto prefAttr = skeleton::PreferenceAttr::get(
      ctx, StringAttr::get(ctx, toString(info.preference)));

  // The operator writes into the destination buffer in place. A tensor-level
  // operator has no side effects, so the write that follows is also what keeps
  // the computation alive: without a use its result would be dead code and
  // would disappear before bufferization.
  Value result;
  bool resultRestrict = true;
  if (info.opType == SkeletonOpType::Map) {
    // The result is the destination, through a writable view of it: the map
    // writes the caller's buffer instead of a temporary that would then be
    // copied back. A temporary is not reachable from a device kernel, and
    // relying on empty tensor elimination to remove it only holds for the
    // shapes it recognizes.
    Value mapsInto = createToTensor(body, loc, dstMemref, /*writable=*/true);
    // One tensor per input data argument, then one rank-0 tensor per scalar:
    // skeleton.map broadcasts a scalar over the destination.
    SmallVector<Value> ins;
    for (unsigned i = 1; i < info.dataOperands.size(); ++i)
      ins.push_back(createToTensor(body, loc, entry->getArgument(argIdx++)));
    for (unsigned i = 0; i < info.scalarOperands.size(); ++i) {
      Value scalar = entry->getArgument(argIdx++);
      auto scalarTy = RankedTensorType::get({}, scalar.getType());
      ins.push_back(tensor::FromElementsOp::create(body, loc, scalarTy,
                                                   ValueRange{scalar}));
    }
    result = skeleton::MapOp::create(body, loc, ValueRange(ins), mapsInto,
                                     pureFnAttr, prefAttr)
                 .getResult();
    // The materialize below is a no-op for bufferization; it is what keeps the
    // map alive.
    resultRestrict = false;
  } else if (info.opType == SkeletonOpType::Matmul) {
    // A matrix multiply writes the destination as `outs`. The writable view
    // keeps the contraction writing into the caller's buffer, which is what a
    // device kernel can reach.
    Value lhs = createToTensor(body, loc, entry->getArgument(argIdx++));
    Value rhs = createToTensor(body, loc, entry->getArgument(argIdx++));
    Value accumulate = createToTensor(body, loc, dstMemref, /*writable=*/true);
    // linalg.matmul adds into its outs, so the destination is cleared first;
    // the operator's contract is C = A * B, not C += A * B.
    Value zero = arith::ConstantOp::create(
        body, loc, cast<MemRefType>(dstMemref.getType()).getElementType(),
        body.getZeroAttr(cast<MemRefType>(dstMemref.getType()).getElementType()));
    linalg::FillOp::create(body, loc, ValueRange{zero}, ValueRange{accumulate});
    result = linalg::MatmulOp::create(body, loc, ValueRange{lhs, rhs},
                                      ValueRange{accumulate})
                 .getResult(0);
    // The contraction accumulates in the destination, so there is nothing to
    // copy back; the write below only keeps it alive.
    resultRestrict = false;
  } else {
    // The reduction accumulates in its `outs`, so the identity is written into
    // the destination first and the reduction then runs there. That keeps a
    // reduction on the device reachable from the kernel, and the destination's
    // shape is what tells the operator how many trailing dimensions it
    // reduces.
    Value inTensor = createToTensor(body, loc, entry->getArgument(argIdx++));
    Value identity = entry->getArgument(argIdx++);
    // The destination is the accumulator: a writable view keeps the fill and
    // the reduction writing into the caller's buffer. A read-only view would
    // make bufferization allocate a temporary and copy it back, which on a
    // device means the kernel writes host memory instead of the buffer the
    // runtime placed there.
    Value init = createToTensor(body, loc, dstMemref, /*writable=*/true);
    linalg::FillOp::create(body, loc, ValueRange{identity}, ValueRange{init});
    result = skeleton::ReduceOp::create(body, loc, inTensor, init, pureFnAttr,
                                        prefAttr)
                 .getResult();
    // The accumulator already lives in the destination, so this write is a
    // no-op for bufferization; it is what keeps the reduction alive.
    resultRestrict = false;
  }

  bufferization::MaterializeInDestinationOp::create(
      body, loc, /*result=*/Type(), result, dstMemref, resultRestrict,
      /*writable=*/true);
  func::ReturnOp::create(body, loc, ValueRange{});
  return unit;
}

cir::FuncOp CIRSkeletonSplitPass::createThunk(SkeletonCallInfo &info, int64_t id,
                                              func::FuncOp unit) {
  MLIRContext *ctx = &getContext();
  Location loc = info.callOp.getLoc();

  auto voidPtrTy = cir::PointerType::get(cir::VoidType::get(ctx));
  auto lengthTy = cir::IntType::get(ctx, /*width=*/64, /*isSigned=*/false);
  SmallVector<Type> argTypes;
  SmallVector<Attribute> dataTypes;
  for (const DataOperandRef &data : info.dataOperands) {
    dataTypes.push_back(TypeAttr::get(data.type));
    argTypes.push_back(voidPtrTy);
    for (int64_t dim = 0; dim < data.type.getRank(); ++dim)
      if (ShapedType::isDynamic(data.type.getShape()[dim]))
        argTypes.push_back(lengthTy);
  }
  for (const ScalarOperandRef &scalar : info.scalarOperands)
    argTypes.push_back(scalar.cirType);

  auto funcType =
      cir::FuncType::get(argTypes, cir::VoidType::get(&getContext()),
                         /*isVarArg=*/false);
  std::string name = (kUnitPrefix + Twine(id) + kThunkSuffix).str();
  builder->setInsertionPointToEnd(module.getBody());
  auto thunk = cir::FuncOp::create(*builder, loc, name, funcType);
  thunk.setPrivate();

  NamedAttrList abi;
  abi.append("op_id", builder->getI64IntegerAttr(id));
  abi.append("unit", builder->getStringAttr(unit.getSymName()));
  abi.append("operands", ArrayAttr::get(ctx, dataTypes));
  thunk->setAttr(kHostABIAttr, DictionaryAttr::get(ctx, abi));
  return thunk;
}

void CIRSkeletonSplitPass::runOnOperation() {
  module = getOperation();
  MLIRContext *ctx = &getContext();
  OpBuilder opBuilder(ctx);
  builder = &opBuilder;

  auto pureFns = collectPureFunctions(module);
  auto opDecls = collectSkeletonOpDecls(module);
  if (failed(opDecls)) {
    signalPassFailure();
    return;
  }
  if (opDecls->empty())
    return;

  DominanceInfo domInfo(module);
  bool reportedError = false;

  // Every call to a skeleton helper is accounted for: the host it sits in must
  // be an annotated host and must contain exactly that one call.
  SmallVector<cir::FuncOp> hosts;
  DenseMap<Operation *, SmallVector<cir::CallOp>> callsByHost;
  module.walk([&](cir::CallOp callOp) {
    auto callee = callOp.getCallee();
    if (!callee || opDecls->find(*callee) == opDecls->end())
      return;
    auto host = callOp->getParentOfType<cir::FuncOp>();
    if (!host) {
      callOp.emitError() << "skeleton call is not inside a function";
      reportedError = true;
      return;
    }
    auto [it, inserted] = callsByHost.try_emplace(host.getOperation());
    if (inserted)
      hosts.push_back(host);
    it->second.push_back(callOp);
  });
  if (reportedError) {
    signalPassFailure();
    return;
  }

  SmallVector<SkeletonCallInfo> worklist;
  for (cir::FuncOp host : hosts) {
    if (!hasSkeletonRegionAnnotation(host)) {
      host.emitError() << "calls a skeleton operator but is not marked as a "
                          "skeleton region; add "
                          "__attribute__((annotate(\"skeleton.region\", "
                          "\"CPU\"))) to the function";
      reportedError = true;
      continue;
    }
    if (host.isExternal()) {
      host.emitError() << "a skeleton region host must have a body";
      reportedError = true;
      continue;
    }
    if (!host.getBody().hasOneBlock()) {
      host.emitError() << "a skeleton region host must be a single-block "
                          "function";
      reportedError = true;
      continue;
    }
    if (host.getNumResults() != 0) {
      host.emitError() << "a skeleton region host must return void: write the "
                          "operator's result into an output parameter";
      reportedError = true;
      continue;
    }

    SmallVector<cir::CallOp> &calls = callsByHost[host.getOperation()];
    if (calls.size() != 1) {
      host.emitError() << "a skeleton region host must contain exactly one "
                          "skeleton operator call, found "
                       << calls.size();
      reportedError = true;
      continue;
    }

    cir::CallOp callOp = calls.front();
    SkeletonCallInfo info;
    info.cirFunc = host;
    info.callOp = callOp;
    info.opType = opDecls->lookup(*callOp.getCallee());

    bool notThin = false;
    for (Operation &op : host.getBody().front())
      if (!isThinWrapperOp(&op, callOp)) {
        op.emitError() << "a skeleton region host must be a thin wrapper "
                          "around its one skeleton operator call; this "
                          "statement would be dropped";
        reportedError = true;
        notThin = true;
        break;
      }
    if (notThin)
      continue;

    // A matmul has no scalar body to name: the contraction itself is the
    // operator, and only linalg can express it.
    if (info.opType != SkeletonOpType::Matmul) {
      auto pureFnRef = extractPureFnRef(callOp, 0, pureFns, domInfo);
      if (!pureFnRef) {
        callOp.emitError()
            << "the first argument of a skeleton operator call must be a pure "
               "function reference (a function marked "
               "__attribute__((annotate(\"skeleton.pure\"))))";
        reportedError = true;
        continue;
      }
      info.pureFnName = pureFnRef.getRootReference().getValue();
    }

    auto preference = getSkeletonPreference(host);
    if (failed(preference)) {
      reportedError = true;
      continue;
    }
    info.preference = *preference;

    if (failed(resolveDataOperands(info))) {
      reportedError = true;
      continue;
    }
    if (info.opType == SkeletonOpType::Reduce &&
        (info.dataOperands.size() != 2 || info.scalarOperands.size() != 1)) {
      callOp.emitError() << "skeleton_reduce takes a destination, one input "
                            "and one identity value";
      reportedError = true;
      continue;
    }
    if (info.opType == SkeletonOpType::Matmul &&
        (info.dataOperands.size() != 3 || !info.scalarOperands.empty())) {
      callOp.emitError() << "skeleton_matmul takes a destination, a left "
                            "operand and a right operand";
      reportedError = true;
      continue;
    }
    worklist.push_back(info);
  }

  // A pure function is part of the operator, not of the host program: after the
  // split it exists only as the func.func an operator unit references, so a
  // call from host code would be left dangling.
  module.walk([&](cir::CallOp callOp) {
    auto callee = callOp.getCallee();
    if (!callee || pureFns.find(*callee) == pureFns.end())
      return;
    callOp.emitError() << "host code may not call the pure function '" << *callee
                       << "': a pure function only exists inside the operator "
                          "units that reference it. Make it an ordinary "
                          "function if the host needs it too.";
    reportedError = true;
  });

  if (reportedError || worklist.empty()) {
    if (reportedError)
      signalPassFailure();
    return;
  }

  // Pure functions become func.func so that the units can reference them.
  if (failed(convertPureFunctionsToFunc(module, pureFns))) {
    signalPassFailure();
    return;
  }

  for (auto [index, info] : llvm::enumerate(worklist)) {
    int64_t id = static_cast<int64_t>(index);
    func::FuncOp unit = createUnit(info, id);
    cir::FuncOp thunk = createThunk(info, id, unit);
    if (failed(rewriteHost(info, thunk))) {
      signalPassFailure();
      return;
    }
  }

  // The skeleton helper declarations only existed for the calls just rewritten.
  SmallVector<cir::FuncOp> toErase;
  module.walk([&](cir::FuncOp func) {
    if (func.isExternal() && opDecls->find(func.getSymName()) != opDecls->end())
      toErase.push_back(func);
  });
  for (cir::FuncOp func : toErase)
    func.erase();
}

} // namespace

std::unique_ptr<Pass> mlir::createCIRSkeletonSplitPass() {
  return std::make_unique<CIRSkeletonSplitPass>();
}
