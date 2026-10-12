//===- CirCallAnalysis.cpp - analyze CIR for the manual path ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Transforms/CirCallAnalysis.h"

#include <functional>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Transforms/CirSkeletonAnnotations.h"

using namespace mlir;
using namespace cir;

namespace {

/// Resolve a call argument that should name a pure function to the referenced
/// function, following cir.cast wrappers and forwarding through values staged
/// in a stack local (a cir.store into an alloca read back with a cir.load).
/// Only a "skeleton.pure"-annotated global resolves to a symbol; every other
/// shape returns an empty attribute and the caller reports the argument as not
/// being a pure-function reference. Lambda-to-function-pointer arguments are
/// not resolved here yet (see the lambda follow-up).
static FlatSymbolRefAttr tracePureFnRef(Value arg,
                                        const DenseSet<StringRef> &pureFns,
                                        DominanceInfo &domInfo,
                                        DenseSet<Value> &visited) {
  if (!visited.insert(arg).second)
    return {};
  // Walk through cir.cast ops.
  while (auto castOp = arg.getDefiningOp<cir::CastOp>())
    arg = castOp.getSrc();

  // cir.get_global gives a direct reference.
  if (auto getGlobal = arg.getDefiningOp<cir::GetGlobalOp>()) {
    StringRef name = getGlobal.getName();
    if (pureFns.contains(name))
      return SymbolRefAttr::get(arg.getContext(), name);
    return {};
  }

  // A function address may be staged in a stack local: forward through the
  // single cir.store into an alloca that dominates this cir.load. Other
  // address sources (function parameters, pointers loaded from elsewhere,
  // globals) cannot be pinned to one pure function here, so they fail
  // conservatively.
  if (auto loadOp = arg.getDefiningOp<cir::LoadOp>()) {
    Value addr = loadOp.getAddr();
    if (addr.getDefiningOp<cir::AllocaOp>()) {
      cir::StoreOp theStore;
      for (OpOperand &use : addr.getUses()) {
        auto store = dyn_cast<cir::StoreOp>(use.getOwner());
        if (!store || use.getOperandNumber() != cir::StoreOp::odsIndex_addr)
          continue;
        if (!domInfo.dominates(store, loadOp))
          continue;
        if (theStore) // several writes in flight: not a fixed address
          return {};
        theStore = store;
      }
      if (theStore)
        return tracePureFnRef(theStore.getValue(), pureFns, domInfo, visited);
    }
  }
  return {};
}

} // namespace

namespace cir {

DenseSet<StringRef> collectPureFunctions(ModuleOp module) {
  DenseSet<StringRef> pureFns;
  module.walk([&](cir::FuncOp func) {
    if (hasSkeletonPureAnnotation(func))
      pureFns.insert(func.getSymName());
  });
  return pureFns;
}

FailureOr<DenseMap<StringRef, SkeletonOpType>>
collectSkeletonOpDecls(ModuleOp module) {
  DenseMap<StringRef, SkeletonOpType> opDecls;
  bool invalid = false;
  module.walk([&](cir::FuncOp func) {
    if (!func.isExternal() || !hasSkeletonOpAnnotation(func))
      return;
    auto opType = getSkeletonOpType(func);
    if (failed(opType)) {
      invalid = true; // the error was already reported on func
      return;
    }
    opDecls[func.getSymName()] = *opType;
  });
  if (invalid)
    return failure();
  return opDecls;
}

FlatSymbolRefAttr extractPureFnRef(cir::CallOp callOp, unsigned argIdx,
                                   const DenseSet<StringRef> &pureFns,
                                   DominanceInfo &domInfo) {
  if (argIdx >= callOp.getNumOperands())
    return {};
  DenseSet<Value> visited;
  return tracePureFnRef(callOp.getOperand(argIdx), pureFns, domInfo, visited);
}

FailureOr<HostArg> resolveToHostArg(Value operand, cir::FuncOp host) {
  DenseSet<Value> visited;
  // The member the operand names, if it reaches the call through one. Only one
  // member access can appear on the way back to the parameter.
  std::optional<unsigned> memberIndex;
  std::function<FailureOr<BlockArgument>(Value)> walk =
      [&](Value v) -> FailureOr<BlockArgument> {
    if (!v || !visited.insert(v).second)
      return failure();
    if (auto blockArg = dyn_cast<BlockArgument>(v))
      if (blockArg.getOwner()->getParentOp() == host.getOperation())
        return blockArg;

    // Peel back to the host parameter the value is held in: a slot's contents
    // (a load), a member of what the slot holds (a span's data pointer or its
    // length), or the slot itself (the helper takes its data arguments by
    // reference, so a pointer parameter reaches the call as the address of its
    // slot).
    if (auto load = v.getDefiningOp<cir::LoadOp>())
      return walk(load.getAddr());
    if (auto member = v.getDefiningOp<cir::GetMemberOp>()) {
      memberIndex = static_cast<unsigned>(member.getIndex());
      return walk(member.getAddr());
    }

    // What is left is an address; a local alloca staging a host parameter is
    // the one to follow, and a straight-line parameter copy stores it exactly
    // once.
    if (!v.getDefiningOp<cir::AllocaOp>())
      return failure();
    Value stored;
    for (OpOperand &use : v.getUses()) {
      auto store = dyn_cast<cir::StoreOp>(use.getOwner());
      if (!store || use.getOperandNumber() != cir::StoreOp::odsIndex_addr)
        continue;
      if (stored)
        return failure(); // several writes: not a plain parameter copy
      stored = store.getValue();
    }
    if (!stored)
      return failure();
    return walk(stored);
  };

  auto arg = walk(operand);
  if (failed(arg))
    return failure();
  return HostArg{*arg, memberIndex};
}

} // namespace cir
