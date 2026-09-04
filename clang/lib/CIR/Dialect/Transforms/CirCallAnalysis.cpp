//===- CirCallAnalysis.cpp - analyze CIR for the manual path ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Transforms/CirCallAnalysis.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Transforms/CIRAnnotations.h"

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
    if (getAnnotationByName(func, "skeleton.pure"))
      pureFns.insert(func.getSymName());
  });
  return pureFns;
}

DenseMap<StringRef, StringRef> collectSkeletonOpDecls(ModuleOp module) {
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

std::string extractPreference(cir::FuncOp func) {
  if (auto ann = getAnnotationByName(func, "skeleton.region"))
    if (auto args = ann.getArgs())
      if (!args.empty())
        if (auto pref = dyn_cast<StringAttr>(args[0]))
          return pref.getValue().str();
  return "CPU";
}

FlatSymbolRefAttr extractPureFnRef(cir::CallOp callOp, unsigned argIdx,
                                   const DenseSet<StringRef> &pureFns,
                                   DominanceInfo &domInfo) {
  if (argIdx >= callOp.getNumOperands())
    return {};
  DenseSet<Value> visited;
  return tracePureFnRef(callOp.getOperand(argIdx), pureFns, domInfo, visited);
}

} // namespace cir
