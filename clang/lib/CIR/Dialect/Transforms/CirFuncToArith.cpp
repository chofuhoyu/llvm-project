//===- CirFuncToArith.cpp - translate CIR function bodies to arith -*- C++ -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pure function is the scalar computation an operator applies. It is not part
// of the host program: only the operator unit references it, and the unit is
// compiled on its own, so the body has to become a block the linalg lowering can
// inline -- arith, math and a terminal return.
//
// The supported shape is one block of straight-line scalar code. Values staged
// through allocas are tracked as slots: in straight-line code the last store to
// a slot is what a later load reads. Comparisons, selects (CIR's ternary) and
// the math builtins are part of the scalar vocabulary; a ternary is translated
// by evaluating both arms and selecting between them, which requires the arms to
// be free of side effects.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Transforms/CirFuncToArith.h"

#include "clang/CIR/Dialect/Transforms/CirScalarTypeConverter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringSwitch.h"
#include <functional>

#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace mlir;
using namespace cir;

namespace cir {

namespace {

/// The math dialect operation a CIR builtin maps to, or an empty string. CIR
/// models the C library's floating-point builtins as operations of its own, and
/// each has a math dialect counterpart with the same meaning; the unit lowering
/// then takes math to LLVM intrinsics.
StringRef mathDialectOp(Operation *op) {
  return StringSwitch<StringRef>(op->getName().getStringRef())
      .Case("cir.sqrt", "math.sqrt")
      .Case("cir.exp", "math.exp")
      .Case("cir.exp2", "math.exp2")
      .Case("cir.log", "math.log")
      .Case("cir.log2", "math.log2")
      .Case("cir.log10", "math.log10")
      .Case("cir.sin", "math.sin")
      .Case("cir.cos", "math.cos")
      .Case("cir.tan", "math.tan")
      .Case("cir.tanh", "math.tanh")
      .Case("cir.fabs", "math.absf")
      .Case("cir.floor", "math.floor")
      .Case("cir.ceil", "math.ceil")
      .Case("cir.round", "math.round")
      .Case("cir.trunc", "math.trunc")
      .Default("");
}

} // namespace

func::FuncOp createFuncFromCirBody(cir::FuncOp cirFunc, StringRef newName,
                                   OpBuilder &builder) {
  auto *ctx = builder.getContext();
  CirScalarTypeConverter converter;

  SmallVector<Type> inputs, results;
  for (Type t : cirFunc.getArgumentTypes()) {
    Type mapped = converter.convertType(t);
    if (!mapped) {
      cirFunc.emitError() << "unsupported CIR type in pure function: " << t;
      return nullptr;
    }
    inputs.push_back(mapped);
  }
  for (Type t : cirFunc.getResultTypes()) {
    Type mapped = converter.convertType(t);
    if (!mapped) {
      cirFunc.emitError() << "unsupported CIR type in pure function: " << t;
      return nullptr;
    }
    results.push_back(mapped);
  }

  auto funcType = FunctionType::get(ctx, inputs, results);
  auto newFunc =
      func::FuncOp::create(builder, cirFunc.getLoc(), newName, funcType);
  newFunc.setSymVisibility(cirFunc.getSymVisibility());

  // A pure function that carries a body needs it translated into an arith body
  // (populateArithFuncBody) so SkeletonToLinalg can clone it. On failure
  // populateArithFuncBody has already emitted a diagnostic; roll the partial
  // func.func back and report the failure to the caller.
  if (!cirFunc.isExternal()) {
    if (failed(populateArithFuncBody(cirFunc, newFunc, builder))) {
      newFunc.erase();
      return nullptr;
    }
  }

  return newFunc;
}

LogicalResult populateArithFuncBody(cir::FuncOp cirFunc,
                                    func::FuncOp targetFunc,
                                    OpBuilder &builder) {
  if (!cirFunc.getBody().hasOneBlock())
    return cirFunc.emitError()
           << "cannot translate a pure function with a multi-block body to "
              "func.func; only single-block straight-line bodies are "
              "supported";

  CirScalarTypeConverter converter;
  auto *cirEntry = &cirFunc.getBody().front();
  auto *stdEntry = targetFunc.addEntryBlock();

  // CIR parameters map positionally to the func.func entry block arguments.
  DenseMap<Value, Value> valueMap;
  for (auto [i, arg] : llvm::enumerate(cirEntry->getArguments()))
    valueMap[arg] = stdEntry->getArgument(i);

  // Alloca slots we have seen; the slot value is the latest value stored into
  // that alloca (straight-line body: stores are ordered, so "latest" is the
  // value a later load reads).
  DenseSet<Value> allocas;
  DenseMap<Value, Value> slotValues;

  // Reject loads/stores with semantics a plain value read/write cannot carry
  // (volatile, atomic, non-temporal).
  auto rejectUnsupportedMemoryAccess = [](Operation *op, StringRef kind) {
    return op->emitError() << "cannot translate a " << kind
                           << " in a skeleton pure function body";
  };

  // Translate one straight-line scalar block into \p b and return the value its
  // terminator yields. A ternary's arms recurse here: they share the slot state,
  // which is sound because a region arm may not write to a slot, so evaluating
  // both arms and selecting between their results is the same computation.
  std::function<FailureOr<Value>(Block &, OpBuilder &, bool)> translateBlock =
      [&](Block &block, OpBuilder &b, bool inRegionArm) -> FailureOr<Value> {
    auto lookupValue = [&](Value v) -> FailureOr<Value> {
      auto it = valueMap.find(v);
      if (it == valueMap.end()) {
        v.getDefiningOp()->emitError()
            << "cannot resolve an operand that was not produced by a "
               "supported scalar operation";
        return failure();
      }
      return it->second;
    };

    for (Operation &op : block.without_terminator()) {
      // A local variable whose address does not escape: `cir.alloca`.
      if (auto alloca = dyn_cast<cir::AllocaOp>(op)) {
        Type pointee = alloca.getAllocaType();
        if (!converter.convertType(pointee))
          return alloca.emitError()
                 << "unsupported alloca pointee type in pure function: "
                 << pointee;
        if (alloca.isDynamic())
          return alloca.emitError()
                 << "cannot translate a dynamically-sized alloca in a pure "
                    "function";

        // The address must only ever be used as the destination of stores or
        // the source of loads. Any other use (passed somewhere, cast, ...)
        // means the value escapes and the straight-line slot model is unsound.
        Value addr = alloca.getAddr();
        for (OpOperand &use : addr.getUses()) {
          Operation *user = use.getOwner();
          bool validUse =
              (isa<cir::StoreOp>(user) &&
               use.getOperandNumber() == 1) || // value, addr
              (isa<cir::LoadOp>(user) && use.getOperandNumber() == 0); // addr
          if (!validUse)
            return alloca.emitError()
                   << "cannot translate an alloca whose address escapes in a "
                      "pure function";
        }
        allocas.insert(addr);
        slotValues.erase(addr); // not yet initialized
        continue;
      }

      // A value read: resolve to the current slot contents.
      if (auto load = dyn_cast<cir::LoadOp>(op)) {
        if (load.getIsVolatile() || load.getIsNontemporal() ||
            load.getMemOrder().has_value())
          return rejectUnsupportedMemoryAccess(&op, "volatile/atomic load");
        Value addr = load.getAddr();
        if (!allocas.contains(addr))
          return load.emitError()
                 << "cannot translate a load from an address that is not a "
                    "local alloca in a pure function";
        auto it = slotValues.find(addr);
        if (it == slotValues.end())
          return load.emitError()
                 << "read of an uninitialized value in a pure function";
        valueMap[load.getResult()] = it->second;
        continue;
      }

      // A value write: record it as the slot's current value. A region arm is
      // evaluated whether or not it is selected, so a write there would happen
      // unconditionally.
      if (auto store = dyn_cast<cir::StoreOp>(op)) {
        if (inRegionArm)
          return store.emitError()
                 << "cannot translate a store inside a ternary arm in a pure "
                    "function";
        if (store.getIsVolatile() || store.getIsNontemporal() ||
            store.getMemOrder().has_value())
          return rejectUnsupportedMemoryAccess(&op, "volatile/atomic store");
        Value addr = store.getAddr();
        if (!allocas.contains(addr))
          return store.emitError()
                 << "cannot translate a store to an address that is not a local "
                    "alloca in a pure function";
        auto storedIt = valueMap.find(store.getValue());
        if (storedIt == valueMap.end())
          return store.emitError()
                 << "cannot resolve the value stored by this operation";
        slotValues[addr] = storedIt->second;
        continue;
      }

      // Constants.
      if (auto cst = dyn_cast<cir::ConstantOp>(op)) {
        Attribute value = cst.getValue();
        TypedAttr typedValue;
        if (auto intAttr = dyn_cast<cir::IntAttr>(value)) {
          Type convertedTy = converter.convertType(intAttr.getType());
          if (!convertedTy)
            return cst.emitError()
                   << "unsupported integer constant type: " << intAttr.getType();
          typedValue = IntegerAttr::get(convertedTy, intAttr.getValue());
        } else if (auto fpAttr = dyn_cast<cir::FPAttr>(value)) {
          Type convertedTy = converter.convertType(fpAttr.getType());
          if (!convertedTy)
            return cst.emitError() << "unsupported floating-point constant "
                                      "type: "
                                   << fpAttr.getType();
          typedValue = FloatAttr::get(convertedTy, fpAttr.getValue());
        } else if (auto boolAttr = dyn_cast<cir::BoolAttr>(value)) {
          typedValue = BoolAttr::get(b.getContext(), boolAttr.getValue());
        } else {
          return cst.emitError()
                 << "unsupported constant in pure function: " << value;
        }
        auto constOp =
            arith::ConstantOp::create(b, cst.getLoc(), typedValue);
        valueMap[cst.getResult()] = constOp.getResult();
        continue;
      }

      // Integer binary arithmetic: `cir.add/sub/mul`.
      if (auto bin = dyn_cast<cir::AddOp>(op)) {
        auto l = lookupValue(bin.getLhs());
        auto r = lookupValue(bin.getRhs());
        if (failed(l) || failed(r))
          return failure();
        valueMap[bin.getResult()] =
            arith::AddIOp::create(b, bin.getLoc(), *l, *r).getResult();
        continue;
      }
      if (auto bin = dyn_cast<cir::SubOp>(op)) {
        auto l = lookupValue(bin.getLhs());
        auto r = lookupValue(bin.getRhs());
        if (failed(l) || failed(r))
          return failure();
        valueMap[bin.getResult()] =
            arith::SubIOp::create(b, bin.getLoc(), *l, *r).getResult();
        continue;
      }
      if (auto bin = dyn_cast<cir::MulOp>(op)) {
        auto l = lookupValue(bin.getLhs());
        auto r = lookupValue(bin.getRhs());
        if (failed(l) || failed(r))
          return failure();
        valueMap[bin.getResult()] =
            arith::MulIOp::create(b, bin.getLoc(), *l, *r).getResult();
        continue;
      }

      // Integer division/remainder choose the signed or unsigned arith op from
      // the CIR operand's signedness.
      if (auto bin = dyn_cast<cir::DivOp>(op)) {
        auto l = lookupValue(bin.getLhs());
        auto r = lookupValue(bin.getRhs());
        if (failed(l) || failed(r))
          return failure();
        bool isSigned = cast<cir::IntType>(bin.getLhs().getType()).isSigned();
        Value created =
            isSigned ? arith::DivSIOp::create(b, bin.getLoc(), *l, *r).getResult()
                     : arith::DivUIOp::create(b, bin.getLoc(), *l, *r).getResult();
        valueMap[bin.getResult()] = created;
        continue;
      }
      if (auto bin = dyn_cast<cir::RemOp>(op)) {
        auto l = lookupValue(bin.getLhs());
        auto r = lookupValue(bin.getRhs());
        if (failed(l) || failed(r))
          return failure();
        bool isSigned = cast<cir::IntType>(bin.getLhs().getType()).isSigned();
        Value created =
            isSigned ? arith::RemSIOp::create(b, bin.getLoc(), *l, *r).getResult()
                     : arith::RemUIOp::create(b, bin.getLoc(), *l, *r).getResult();
        valueMap[bin.getResult()] = created;
        continue;
      }

      // Floating-point binary arithmetic.
      if (auto bin = dyn_cast<cir::FAddOp>(op)) {
        auto l = lookupValue(bin.getLhs());
        auto r = lookupValue(bin.getRhs());
        if (failed(l) || failed(r))
          return failure();
        valueMap[bin.getResult()] =
            arith::AddFOp::create(b, bin.getLoc(), *l, *r).getResult();
        continue;
      }
      if (auto bin = dyn_cast<cir::FSubOp>(op)) {
        auto l = lookupValue(bin.getLhs());
        auto r = lookupValue(bin.getRhs());
        if (failed(l) || failed(r))
          return failure();
        valueMap[bin.getResult()] =
            arith::SubFOp::create(b, bin.getLoc(), *l, *r).getResult();
        continue;
      }
      if (auto bin = dyn_cast<cir::FMulOp>(op)) {
        auto l = lookupValue(bin.getLhs());
        auto r = lookupValue(bin.getRhs());
        if (failed(l) || failed(r))
          return failure();
        valueMap[bin.getResult()] =
            arith::MulFOp::create(b, bin.getLoc(), *l, *r).getResult();
        continue;
      }
      if (auto bin = dyn_cast<cir::FDivOp>(op)) {
        auto l = lookupValue(bin.getLhs());
        auto r = lookupValue(bin.getRhs());
        if (failed(l) || failed(r))
          return failure();
        valueMap[bin.getResult()] =
            arith::DivFOp::create(b, bin.getLoc(), *l, *r).getResult();
        continue;
      }

      // Fused multiply-add: lower to an unfused mul + add sequence.
      // `cir.fmuladd` explicitly permits either fused or unfused lowering, so
      // this is a sound choice that avoids pulling the math dialect in.
      if (auto fma = dyn_cast<cir::FMulAddOp>(op)) {
        auto a = lookupValue(fma.getA());
        auto bb = lookupValue(fma.getB());
        auto c = lookupValue(fma.getC());
        if (failed(a) || failed(bb) || failed(c))
          return failure();
        auto mul = arith::MulFOp::create(b, fma.getLoc(), *a, *bb);
        valueMap[fma.getResult()] =
            arith::AddFOp::create(b, fma.getLoc(), mul.getResult(), *c)
                .getResult();
        continue;
      }

      // Unary negation.
      if (auto neg = dyn_cast<cir::FNegOp>(op)) {
        auto v = lookupValue(neg.getInput());
        if (failed(v))
          return failure();
        valueMap[neg.getResult()] =
            arith::NegFOp::create(b, neg.getLoc(), *v).getResult();
        continue;
      }
      if (auto minus = dyn_cast<cir::MinusOp>(op)) {
        auto v = lookupValue(minus.getInput());
        if (failed(v))
          return failure();
        Type convertedTy = converter.convertType(minus.getInput().getType());
        if (!convertedTy)
          return minus.emitError() << "unsupported type for integer negation: "
                                   << minus.getInput().getType();
        auto zero = arith::ConstantOp::create(
            b, minus.getLoc(), IntegerAttr::get(convertedTy, /*value=*/0));
        valueMap[minus.getResult()] =
            arith::SubIOp::create(b, minus.getLoc(), zero.getResult(), *v)
                .getResult();
        continue;
      }

      // Comparison: CIR's predicate names the C semantics, which arith spells
      // out per type (an ordered predicate for floats, signed or unsigned
      // integer ordering from the operand's signedness).
      if (auto cmp = dyn_cast<cir::CmpOp>(op)) {
        auto l = lookupValue(cmp.getLhs());
        auto r = lookupValue(cmp.getRhs());
        if (failed(l) || failed(r))
          return failure();
        Value created;
        if (isa<FloatType>(converter.convertType(cmp.getLhs().getType()))) {
          auto predicate = [&]() -> std::optional<arith::CmpFPredicate> {
            switch (cmp.getKind()) {
            case cir::CmpOpKind::lt:
              return arith::CmpFPredicate::OLT;
            case cir::CmpOpKind::le:
              return arith::CmpFPredicate::OLE;
            case cir::CmpOpKind::gt:
              return arith::CmpFPredicate::OGT;
            case cir::CmpOpKind::ge:
              return arith::CmpFPredicate::OGE;
            case cir::CmpOpKind::eq:
              return arith::CmpFPredicate::OEQ;
            case cir::CmpOpKind::ne:
              return arith::CmpFPredicate::UNE;
            case cir::CmpOpKind::one:
              return arith::CmpFPredicate::ONE;
            case cir::CmpOpKind::uno:
              return arith::CmpFPredicate::UNO;
            }
            return std::nullopt;
          }();
          if (!predicate)
            return cmp.emitError() << "unsupported floating-point comparison";
          created = arith::CmpFOp::create(b, cmp.getLoc(), *predicate, *l, *r);
        } else {
          bool isSigned =
              isa<cir::IntType>(cmp.getLhs().getType()) &&
              cast<cir::IntType>(cmp.getLhs().getType()).isSigned();
          auto predicate = [&]() -> std::optional<arith::CmpIPredicate> {
            switch (cmp.getKind()) {
            case cir::CmpOpKind::lt:
              return isSigned ? arith::CmpIPredicate::slt
                              : arith::CmpIPredicate::ult;
            case cir::CmpOpKind::le:
              return isSigned ? arith::CmpIPredicate::sle
                              : arith::CmpIPredicate::ule;
            case cir::CmpOpKind::gt:
              return isSigned ? arith::CmpIPredicate::sgt
                              : arith::CmpIPredicate::ugt;
            case cir::CmpOpKind::ge:
              return isSigned ? arith::CmpIPredicate::sge
                              : arith::CmpIPredicate::uge;
            case cir::CmpOpKind::eq:
              return arith::CmpIPredicate::eq;
            case cir::CmpOpKind::ne:
              return arith::CmpIPredicate::ne;
            default:
              return std::nullopt;
            }
          }();
          if (!predicate)
            return cmp.emitError() << "unsupported integer comparison";
          created = arith::CmpIOp::create(b, cmp.getLoc(), *predicate, *l, *r);
        }
        valueMap[cmp.getResult()] = created;
        continue;
      }

      // `cond ? a : b`. Both arms are straight-line scalar code, so they are
      // translated next to each other and the condition selects between their
      // results.
      if (auto ternary = dyn_cast<cir::TernaryOp>(op)) {
        auto cond = lookupValue(ternary.getCond());
        if (failed(cond))
          return failure();
        if (!ternary.getTrueRegion().hasOneBlock() ||
            !ternary.getFalseRegion().hasOneBlock())
          return ternary.emitError()
                 << "cannot translate a ternary with a multi-block arm in a "
                    "pure function";
        auto trueValue =
            translateBlock(ternary.getTrueRegion().front(), b, /*inRegionArm=*/true);
        if (failed(trueValue))
          return failure();
        auto falseValue = translateBlock(ternary.getFalseRegion().front(), b,
                                         /*inRegionArm=*/true);
        if (failed(falseValue))
          return failure();
        valueMap[ternary.getResult()] =
            arith::SelectOp::create(b, ternary.getLoc(), *cond, *trueValue,
                                    *falseValue)
                .getResult();
        continue;
      }

      // Scalar casts between arithmetic types. Pointer casts are not part of a
      // pure function's vocabulary.
      if (auto castOp = dyn_cast<cir::CastOp>(op)) {
        auto src = lookupValue(castOp.getSrc());
        if (failed(src))
          return failure();
        Type dstTy = converter.convertType(castOp.getType());
        Type srcTy = converter.convertType(castOp.getSrc().getType());
        if (!dstTy || !srcTy)
          return castOp.emitError() << "unsupported cast in pure function";
        Value created;
        auto srcInt = dyn_cast<IntegerType>(srcTy);
        auto dstInt = dyn_cast<IntegerType>(dstTy);
        auto srcFloat = dyn_cast<FloatType>(srcTy);
        auto dstFloat = dyn_cast<FloatType>(dstTy);
        switch (castOp.getKind()) {
        case cir::CastKind::int_to_float:
          created = cast<cir::IntType>(castOp.getSrc().getType()).isSigned()
                        ? arith::SIToFPOp::create(b, castOp.getLoc(), dstTy, *src)
                              .getResult()
                        : arith::UIToFPOp::create(b, castOp.getLoc(), dstTy, *src)
                              .getResult();
          break;
        case cir::CastKind::float_to_int:
          created = cast<cir::IntType>(castOp.getType()).isSigned()
                        ? arith::FPToSIOp::create(b, castOp.getLoc(), dstTy, *src)
                              .getResult()
                        : arith::FPToUIOp::create(b, castOp.getLoc(), dstTy, *src)
                              .getResult();
          break;
        case cir::CastKind::floating:
          if (!srcFloat || !dstFloat)
            return castOp.emitError() << "unsupported float cast in pure function";
          created = srcFloat.getWidth() < dstFloat.getWidth()
                        ? arith::ExtFOp::create(b, castOp.getLoc(), dstTy, *src)
                              .getResult()
                        : arith::TruncFOp::create(b, castOp.getLoc(), dstTy, *src)
                              .getResult();
          break;
        case cir::CastKind::integral:
          if (!srcInt || !dstInt)
            return castOp.emitError()
                   << "unsupported integer cast in pure function";
          if (srcInt.getWidth() < dstInt.getWidth())
            created =
                cast<cir::IntType>(castOp.getSrc().getType()).isSigned()
                    ? arith::ExtSIOp::create(b, castOp.getLoc(), dstTy, *src)
                          .getResult()
                    : arith::ExtUIOp::create(b, castOp.getLoc(), dstTy, *src)
                          .getResult();
          else
            created = arith::TruncIOp::create(b, castOp.getLoc(), dstTy, *src)
                          .getResult();
          break;
        default:
          return castOp.emitError()
                 << "unsupported cast in pure function: a pure function may "
                    "only cast between arithmetic types";
        }
        valueMap[castOp.getResult()] = created;
        continue;
      }

      // A math builtin: the unit lowering turns the math dialect op into the
      // matching LLVM intrinsic.
      if (StringRef mathName = mathDialectOp(&op); !mathName.empty()) {
        auto operand = lookupValue(op.getOperand(0));
        if (failed(operand))
          return failure();
        Type resultTy = converter.convertType(op.getResult(0).getType());
        if (!resultTy)
          return op.emitError() << "unsupported result type in pure function";
        OperationState state(op.getLoc(), mathName);
        state.addOperands(*operand);
        state.addTypes(resultTy);
        valueMap[op.getResult(0)] = b.create(state)->getResult(0);
        continue;
      }

      // Anything else is out of scope: report it rather than silently dropping
      // part of the computation.
      return op.emitError() << "unsupported operation in pure function body: "
                            << op.getName();
    }

    // The block's result: a pure function returns it, a ternary arm yields it.
    Operation *terminator = block.getTerminator();
    if (auto ret = dyn_cast<cir::ReturnOp>(terminator)) {
      if (ret.getNumOperands() != 1)
        return ret.emitError()
               << "a skeleton pure function must return exactly one value";
      auto v = lookupValue(ret.getOperand(0));
      if (failed(v))
        return failure();
      func::ReturnOp::create(b, ret.getLoc(), ValueRange(*v));
      return *v;
    }
    if (auto yield = dyn_cast<cir::YieldOp>(terminator)) {
      auto v = lookupValue(yield.getValue());
      if (failed(v))
        return failure();
      return *v;
    }
    return terminator->emitError()
           << "cannot translate this block terminator in a pure function";
  };

  builder.setInsertionPointToEnd(stdEntry);
  return success(translateBlock(*cirEntry, builder, /*inRegionArm=*/false)
                     .succeeded());
}

} // namespace cir
