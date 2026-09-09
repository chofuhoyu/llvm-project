//===- CirCallToSkeleton.cpp - calls to Skeleton ops (manual path) -*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Manual path orchestrator: recognize calls to skeleton helper functions
// (declared with the "skeleton.op" annotation) and rewrite the surrounding
// module into standard functions and Skeleton dialect ops.
//
// The pass runs in two phases: it first finds every call to a skeleton helper
// and which pure function it refers to, then lowers the annotated pure
// functions and host wrappers into func.func and skeleton ops.
//
// Semi-automatic path (TODO): annotated for-loops with inline computation.
// Lambda support (TODO): captureless lambdas as pure_fn references.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/CirCallAnalysis.h"
#include "clang/CIR/Dialect/Transforms/CirCallLowering.h"
#include "clang/CIR/Dialect/Transforms/CirSkeletonAnnotations.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"

#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_CIRCALLTOSKELETON
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

/// Drop the CIR codegen attributes clang leaves on the module (the `cir.*`
/// family and the dlti data-layout spec, whose entries mention !cir.ptr).
/// Those attributes name the CIR dialect, which the downstream MLIR tools
/// consuming a skeleton module do not register, so keeping them would make
/// the lowered module unparsable there (e.g. mlir-opt). This only runs once
/// the module has actually been rewritten into func.func + skeleton ops.
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

class CIRCallToSkeletonPass
    : public impl::CIRCallToSkeletonBase<CIRCallToSkeletonPass> {
public:
  using CIRCallToSkeletonBase::CIRCallToSkeletonBase;

  void runOnOperation() override {
    auto module = getOperation();
    auto pureFns = collectPureFunctions(module);
    auto opDecls = collectSkeletonOpDecls(module);
    if (failed(opDecls)) {
      signalPassFailure();
      return;
    }
    if (opDecls->empty())
      return;

    DominanceInfo domInfo(module);

    // Phase 1: Collect all skeleton call info from the original CIR IR.
    SmallVector<SkeletonCallInfo> worklist;
    bool invalidRegion = false;

    module.walk([&](cir::CallOp callOp) {
      auto calleeOpt = callOp.getCallee();
      if (!calleeOpt)
        return;
      StringRef callee = *calleeOpt;

      auto it = opDecls->find(callee);
      if (it == opDecls->end())
        return;

      SkeletonOpType opType = it->second;

      auto pureFnRef = extractPureFnRef(callOp, 0, pureFns, domInfo);
      if (!pureFnRef) {
        callOp.emitWarning("skeleton op call first argument must be a pure "
                           "function reference; semi-automatic path not yet "
                           "supported");
        return;
      }

      auto cirFunc = callOp->getParentOfType<cir::FuncOp>();
      if (!cirFunc)
        return;

      auto preference = getSkeletonPreference(cirFunc);
      if (failed(preference)) {
        invalidRegion = true; // the error was already reported on cirFunc
        return;
      }

      SkeletonCallInfo info{cirFunc,
                            callOp,
                            opType,
                            pureFnRef.getRootReference().getValue(),
                            *preference};
      worklist.push_back(info);
    });

    if (invalidRegion) {
      signalPassFailure();
      return;
    }
    if (worklist.empty())
      return;

    // Convert pure functions from cir.func to func.func so that the skeleton
    // ops created below can reference them via pure_fn. Fail when a pure
    // function carries a body we cannot translate; continuing would silently
    // drop it.
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
      switch (info.opType) {
      case SkeletonOpType::Map:
        result = lowerMapCall(info, newFunc, builder);
        break;
      case SkeletonOpType::Reduce:
        result = lowerReduceCall(info, newFunc, builder);
        break;
      default:
        llvm_unreachable("unknown skeleton op type");
      }

      if (result)
        func::ReturnOp::create(builder, newFunc.getLoc(), ValueRange(result));
    }

    // Erase original cir.func ops (now replaced by func.func ops).
    for (auto &info : worklist) {
      if (info.cirFunc && info.cirFunc->getParentOp())
        info.cirFunc.erase();
    }

    // Erase skeleton op declarations (external, no body). The declarations were
    // parsed into opDecls above, so membership there identifies them.
    SmallVector<cir::FuncOp> toErase;
    module.walk([&](cir::FuncOp func) {
      if (func.isExternal() &&
          opDecls->find(func.getSymName()) != opDecls->end())
        toErase.push_back(func);
    });
    for (auto func : toErase)
      func.erase();

    // After the conversions and erasures above, every cir.func this pass knows
    // how to handle is gone. A cir.func left here is ordinary C++ code around
    // the skeleton region (a caller of a host, main, ...): the host it calls
    // has been rewritten to a memref/tensor func.func and its cir.func erased,
    // so the caller now references a dangling symbol, and dropCirModuleAttrs
    // would already have taken the CIR layout attributes that remaining CIR
    // code still needs. Whole-module lowering cannot handle surrounding code
    // yet, so refuse loudly rather than emit that corrupt half-converted
    // module.
    SmallVector<cir::FuncOp> leftovers;
    module.walk([&](cir::FuncOp func) { leftovers.push_back(func); });
    if (!leftovers.empty()) {
      for (cir::FuncOp func : leftovers)
        func.emitError() << "non-skeleton function " << func.getSymName()
                         << " is left in CIR: cir-call-to-skeleton requires the "
                            "whole module to be skeleton code (pure functions, "
                            "skeleton op declarations, and their hosts), "
                            "surrounding ordinary C++ code is not supported yet";
      signalPassFailure();
      return;
    }

    dropCirModuleAttrs(module);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createCIRCallToSkeletonPass() {
  return std::make_unique<CIRCallToSkeletonPass>();
}
