//===- CirStripCIRAttrs.cpp - drop CIR metadata from a lowered module -*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// After cir-to-llvm the module is LLVM dialect, but individual operations keep
// CIR-source attributes: the `cir.*` string attributes (target CPU features
// and similar) and CIR dialect attributes such as `func_info =
// #cir.cxx_ctor<...>`. Those mention the CIR dialect and its types, so a tool
// that does not register CIR (the skeleton driver, mlir-opt) cannot parse the
// module at all. This pass strips them: it is the last step of the front end
// that hands a module to a non-CIR consumer.
//
// Only attributes are dropped; nothing about the program's semantics depends
// on them once every operation has been lowered.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_CIRSTRIPCIRATTRS
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

/// Whether \p attr mentions the CIR dialect anywhere: it is a CIR dialect
/// attribute itself, it wraps a CIR type, or it holds one nested inside an
/// array or dictionary.
static bool mentionsCIR(Attribute attr, Dialect *cirDialect) {
  if (!attr)
    return false;
  if (&attr.getDialect() == cirDialect)
    return true;
  if (auto typeAttr = dyn_cast<TypeAttr>(attr))
    return &typeAttr.getValue().getDialect() == cirDialect;
  if (auto typed = dyn_cast<TypedAttr>(attr))
    return &typed.getType().getDialect() == cirDialect;
  if (auto array = dyn_cast<ArrayAttr>(attr)) {
    for (Attribute element : array)
      if (mentionsCIR(element, cirDialect))
        return true;
    return false;
  }
  if (auto dict = dyn_cast<DictionaryAttr>(attr)) {
    for (NamedAttribute entry : dict)
      if (mentionsCIR(entry.getValue(), cirDialect))
        return true;
  }
  return false;
}

class CIRStripCIRAttrsPass
    : public impl::CIRStripCIRAttrsBase<CIRStripCIRAttrsPass> {
public:
  using CIRStripCIRAttrsBase::CIRStripCIRAttrsBase;

  void runOnOperation() override {
    Dialect *cirDialect = getContext().getLoadedDialect<cir::CIRDialect>();
    getOperation()->walk([&](Operation *op) {
      SmallVector<StringAttr> toDrop;
      for (NamedAttribute attr : op->getAttrs()) {
        // The `cir.` prefix covers the string attributes whose value is an
        // ordinary builtin attribute and whose name alone marks them.
        if (attr.getName().strref().starts_with("cir.") ||
            mentionsCIR(attr.getValue(), cirDialect))
          toDrop.push_back(attr.getName());
      }
      for (StringAttr name : toDrop)
        op->removeAttr(name);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createCIRStripCIRAttrsPass() {
  return std::make_unique<CIRStripCIRAttrsPass>();
}
