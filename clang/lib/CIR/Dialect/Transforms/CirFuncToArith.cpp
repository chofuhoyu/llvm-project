//===- CirFuncToArith.cpp - translate CIR function bodies to arith -*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Transforms/CirFuncToArith.h"
#include "clang/CIR/Dialect/Transforms/CirScalarTypeConverter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace mlir;
using namespace cir;

namespace cir {

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
  builder.setInsertionPointToStart(stdEntry);

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

  for (Operation &op : cirEntry->getOperations()) {
    builder.setInsertionPointToEnd(stdEntry);

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

    // A value write: record it as the slot's current value.
    if (auto store = dyn_cast<cir::StoreOp>(op)) {
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

    // Scalars never reach here unresolved.
    auto lookupValue = [&](Value v) -> FailureOr<Value> {
      auto it = valueMap.find(v);
      if (it == valueMap.end()) {
        op.emitError() << "cannot resolve an operand that was not produced by "
                          "a supported scalar operation";
        return failure();
      }
      return it->second;
    };

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
          return cst.emitError() << "unsupported floating-point constant type: "
                                 << fpAttr.getType();
        typedValue = FloatAttr::get(convertedTy, fpAttr.getValue());
      } else {
        return cst.emitError()
               << "unsupported constant in pure function: " << value;
      }
      auto constOp =
          arith::ConstantOp::create(builder, cst.getLoc(), typedValue);
      valueMap[cst.getResult()] = constOp.getResult();
      continue;
    }

    // Integer binary arithmetic: `cir.add/sub/mul`.
    if (auto bin = dyn_cast<cir::AddOp>(op)) {
      auto l = lookupValue(bin.getLhs());
      auto r = lookupValue(bin.getRhs());
      if (failed(l) || failed(r))
        return failure();
      auto created = arith::AddIOp::create(builder, bin.getLoc(), *l, *r);
      valueMap[bin.getResult()] = created.getResult();
      continue;
    }
    if (auto bin = dyn_cast<cir::SubOp>(op)) {
      auto l = lookupValue(bin.getLhs());
      auto r = lookupValue(bin.getRhs());
      if (failed(l) || failed(r))
        return failure();
      auto created = arith::SubIOp::create(builder, bin.getLoc(), *l, *r);
      valueMap[bin.getResult()] = created.getResult();
      continue;
    }
    if (auto bin = dyn_cast<cir::MulOp>(op)) {
      auto l = lookupValue(bin.getLhs());
      auto r = lookupValue(bin.getRhs());
      if (failed(l) || failed(r))
        return failure();
      auto created = arith::MulIOp::create(builder, bin.getLoc(), *l, *r);
      valueMap[bin.getResult()] = created.getResult();
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
          isSigned ? arith::DivSIOp::create(builder, bin.getLoc(), *l, *r)
                         .getResult()
                   : arith::DivUIOp::create(builder, bin.getLoc(), *l, *r)
                         .getResult();
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
          isSigned ? arith::RemSIOp::create(builder, bin.getLoc(), *l, *r)
                         .getResult()
                   : arith::RemUIOp::create(builder, bin.getLoc(), *l, *r)
                         .getResult();
      valueMap[bin.getResult()] = created;
      continue;
    }

    // Floating-point binary arithmetic.
    if (auto bin = dyn_cast<cir::FAddOp>(op)) {
      auto l = lookupValue(bin.getLhs());
      auto r = lookupValue(bin.getRhs());
      if (failed(l) || failed(r))
        return failure();
      auto created = arith::AddFOp::create(builder, bin.getLoc(), *l, *r);
      valueMap[bin.getResult()] = created.getResult();
      continue;
    }
    if (auto bin = dyn_cast<cir::FSubOp>(op)) {
      auto l = lookupValue(bin.getLhs());
      auto r = lookupValue(bin.getRhs());
      if (failed(l) || failed(r))
        return failure();
      auto created = arith::SubFOp::create(builder, bin.getLoc(), *l, *r);
      valueMap[bin.getResult()] = created.getResult();
      continue;
    }
    if (auto bin = dyn_cast<cir::FMulOp>(op)) {
      auto l = lookupValue(bin.getLhs());
      auto r = lookupValue(bin.getRhs());
      if (failed(l) || failed(r))
        return failure();
      auto created = arith::MulFOp::create(builder, bin.getLoc(), *l, *r);
      valueMap[bin.getResult()] = created.getResult();
      continue;
    }
    if (auto bin = dyn_cast<cir::FDivOp>(op)) {
      auto l = lookupValue(bin.getLhs());
      auto r = lookupValue(bin.getRhs());
      if (failed(l) || failed(r))
        return failure();
      auto created = arith::DivFOp::create(builder, bin.getLoc(), *l, *r);
      valueMap[bin.getResult()] = created.getResult();
      continue;
    }

    // Fused multiply-add: lower to an unfused mul + add sequence. `cir.fmuladd`
    // explicitly permits either fused or unfused lowering, so this is a sound
    // choice that avoids pulling the math dialect in.
    if (auto fma = dyn_cast<cir::FMulAddOp>(op)) {
      auto a = lookupValue(fma.getA());
      auto b = lookupValue(fma.getB());
      auto c = lookupValue(fma.getC());
      if (failed(a) || failed(b) || failed(c))
        return failure();
      auto mul = arith::MulFOp::create(builder, fma.getLoc(), *a, *b);
      auto add =
          arith::AddFOp::create(builder, fma.getLoc(), mul.getResult(), *c);
      valueMap[fma.getResult()] = add.getResult();
      continue;
    }

    // Unary negation.
    if (auto neg = dyn_cast<cir::FNegOp>(op)) {
      auto v = lookupValue(neg.getInput());
      if (failed(v))
        return failure();
      auto created = arith::NegFOp::create(builder, neg.getLoc(), *v);
      valueMap[neg.getResult()] = created.getResult();
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
          builder, minus.getLoc(), IntegerAttr::get(convertedTy, /*value=*/0));
      auto created =
          arith::SubIOp::create(builder, minus.getLoc(), zero.getResult(), *v);
      valueMap[minus.getResult()] = created.getResult();
      continue;
    }

    // Return: a single value (skeleton pure functions must return one result).
    if (auto ret = dyn_cast<cir::ReturnOp>(op)) {
      if (ret.getNumOperands() != 1)
        return ret.emitError()
               << "a skeleton pure function must return exactly one value";
      auto v = lookupValue(ret.getOperand(0));
      if (failed(v))
        return failure();
      func::ReturnOp::create(builder, ret.getLoc(), ValueRange(*v));
      continue; // terminator: nothing follows in this block
    }

    // Anything else is out of scope: report it rather than silently dropping
    // part of the computation.
    return op.emitError() << "unsupported operation in pure function body: "
                          << op.getName();
  }

  return success();
}

} // namespace cir
