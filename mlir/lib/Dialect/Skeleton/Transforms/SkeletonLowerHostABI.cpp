//===- SkeletonLowerHostABI.cpp - build the host/unit ABI thunks ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `cir-skeleton-split` leaves one declaration per operator next to the host:
//
//   __skeleton_op_<id>_shim(ptr, i64..., <scalar>...) -> ()
//
// carrying `skeleton.host_abi`, which names the unit to call and the memref
// each pointer/length pair denotes. This pass fills the bodies in, after the
// host has been lowered to the LLVM dialect.
//
// The body rebuilds what the memref type describes: for every data argument the
// pointer becomes both the allocated and the aligned pointer, the offset is
// zero, a static dimension contributes a constant size, a dynamic one takes the
// length the caller passed, and the strides are the row-major products. The unit
// is then called with the descriptor fields flattened, which is exactly the
// signature `convert-func-to-llvm` gives a function that takes memrefs -- the
// pass checks that against the declaration the module still carries, so a
// mismatch is a compile-time error rather than a corrupted call.
//
// Buffers are contiguous by construction: the host passes the address of an
// array or of a span, and a span describes one contiguous run of elements.
//
// The generated call to the unit is a plain direct call: `skeleton-lower-dispatch`
// runs next and turns it into a runtime resolve plus an indirect call, so the
// placement decision stays in one place.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir {
namespace skeleton {

#define GEN_PASS_DEF_SKELETONLOWERHOSTABI
#include "mlir/Dialect/Skeleton/Transforms/Passes.h.inc"

namespace {

constexpr StringLiteral kHostABIAttr = "skeleton.host_abi";

/// The flattened LLVM types `convert-func-to-llvm` gives one memref argument:
/// the allocated and aligned pointers, the offset, then one size and one stride
/// per dimension.
SmallVector<Type> memrefDescriptorTypes(MemRefType type, MLIRContext *ctx) {
  SmallVector<Type> types;
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i64Ty = IntegerType::get(ctx, 64);
  types.push_back(ptrTy);
  types.push_back(ptrTy);
  types.push_back(i64Ty);
  for (unsigned i = 0; i < type.getRank(); ++i) {
    types.push_back(i64Ty);
    types.push_back(i64Ty);
  }
  return types;
}

struct SkeletonLowerHostABIPass
    : public impl::SkeletonLowerHostABIBase<SkeletonLowerHostABIPass> {
  using SkeletonLowerHostABIBase::SkeletonLowerHostABIBase;

  void runOnOperation() override;

private:
  /// Fill in one thunk's body. \p thunk is the lowered declaration and \p abi
  /// its description.
  LogicalResult lowerThunk(LLVM::LLVMFuncOp thunk, DictionaryAttr abi);
};

LogicalResult SkeletonLowerHostABIPass::lowerThunk(LLVM::LLVMFuncOp thunk,
                                                   DictionaryAttr abi) {
  MLIRContext *ctx = &getContext();
  Location loc = thunk.getLoc();
  OpBuilder builder(ctx);

  auto opIdAttr = abi.getAs<IntegerAttr>("op_id");
  auto unitAttr = abi.getAs<StringAttr>("unit");
  auto operandsAttr = abi.getAs<ArrayAttr>("operands");
  if (!opIdAttr || !unitAttr || !operandsAttr)
    return thunk.emitError()
           << "malformed skeleton.host_abi: expected op_id, unit and operands";

  SmallVector<MemRefType> dataTypes;
  for (Attribute attr : operandsAttr) {
    auto typeAttr = dyn_cast<TypeAttr>(attr);
    if (!typeAttr || !isa<MemRefType>(typeAttr.getValue()))
      return thunk.emitError()
             << "malformed skeleton.host_abi: operands must be memref types";
    dataTypes.push_back(cast<MemRefType>(typeAttr.getValue()));
  }

  auto unit = SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(
      thunk, StringAttr::get(ctx, unitAttr.getValue()));
  if (!unit)
    return thunk.emitError() << "skeleton.host_abi names unit '"
                             << unitAttr.getValue()
                             << "', which is not a function in this module";

  // The host calls the unit with the descriptor fields flattened. Check that
  // this is what the unit actually takes before generating a call that would
  // read its arguments wrong.
  SmallVector<Type> expectedTypes;
  for (MemRefType type : dataTypes)
    llvm::append_range(expectedTypes, memrefDescriptorTypes(type, ctx));

  LLVMFunctionType unitType = unit.getFunctionType();
  unsigned numMemrefArgs = expectedTypes.size();
  if (unitType.getNumParams() < numMemrefArgs)
    return thunk.emitError() << "unit '" << unitAttr.getValue() << "' takes "
                             << unitType.getNumParams()
                             << " arguments, fewer than the "
                             << numMemrefArgs << " its memrefs need";
  for (unsigned i = 0; i < numMemrefArgs; ++i) {
    if (unitType.getParamType(i) != expectedTypes[i])
      return thunk.emitError()
             << "unit '" << unitAttr.getValue() << "' argument " << i
             << " has type " << unitType.getParamType(i) << " but the host "
             << "passes " << expectedTypes[i]
             << "; the operator's memref signature and the host ABI disagree";
  }

  Block *entry = thunk.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i64Ty = IntegerType::get(ctx, 64);
  auto constant = [&](int64_t value) -> Value {
    return LLVM::ConstantOp::create(builder, loc, i64Ty,
                                    builder.getI64IntegerAttr(value));
  };

  SmallVector<Value> callArgs;
  unsigned argIdx = 0;
  for (MemRefType type : dataTypes) {
    Value base = entry->getArgument(argIdx++);
    // The pointer the caller handed us is the only pointer there is: it is both
    // the allocation and the aligned address, and the view starts at offset 0.
    callArgs.push_back(base);
    callArgs.push_back(base);
    callArgs.push_back(constant(0));

    // The host only ever describes contiguous buffers: a span is one dynamic
    // dimension, an array reference is fully static. Row-major strides are
    // therefore constants unless a dynamic dimension has dimensions after it,
    // whose product would only be known at run time.
    for (int64_t dim = 0; dim + 1 < type.getRank(); ++dim)
      if (ShapedType::isDynamic(type.getShape()[dim]))
        return thunk.emitError()
               << "cannot describe the layout of " << type
               << ": a dynamic dimension must be the last one";

    SmallVector<Value> sizes;
    for (int64_t dim = 0; dim < type.getRank(); ++dim) {
      int64_t size = type.getShape()[dim];
      if (ShapedType::isDynamic(size))
        sizes.push_back(entry->getArgument(argIdx++));
      else
        sizes.push_back(constant(size));
    }
    llvm::append_range(callArgs, sizes);

    int64_t stride = 1;
    SmallVector<Value> strides(type.getRank());
    for (int64_t dim = type.getRank() - 1; dim >= 0; --dim) {
      strides[dim] = constant(stride);
      if (!ShapedType::isDynamic(type.getShape()[dim]))
        stride *= type.getShape()[dim];
    }
    llvm::append_range(callArgs, strides);
  }
  for (; argIdx < entry->getNumArguments(); ++argIdx)
    callArgs.push_back(entry->getArgument(argIdx));

  LLVM::CallOp::create(builder, loc, TypeRange{},
                       FlatSymbolRefAttr::get(ctx, unit.getSymName()),
                       ValueRange{callArgs});
  LLVM::ReturnOp::create(builder, loc, ValueRange{});
  return success();
}

void SkeletonLowerHostABIPass::runOnOperation() {
  ModuleOp module = getOperation();

  SmallVector<LLVM::LLVMFuncOp> thunks;
  for (LLVM::LLVMFuncOp func : module.getOps<LLVM::LLVMFuncOp>())
    if (func->hasAttr(kHostABIAttr))
      thunks.push_back(func);

  for (LLVM::LLVMFuncOp thunk : thunks) {
    if (!thunk.isExternal())
      continue; // already lowered, e.g. by an earlier artifact build
    auto abi = dyn_cast<DictionaryAttr>(thunk->getAttr(kHostABIAttr));
    if (!abi) {
      thunk.emitError() << "expected a dictionary " << kHostABIAttr
                        << " attribute";
      signalPassFailure();
      return;
    }
    if (failed(lowerThunk(thunk, abi))) {
      signalPassFailure();
      return;
    }
  }
}

} // namespace

std::unique_ptr<Pass> createSkeletonLowerHostABIPass() {
  return std::make_unique<SkeletonLowerHostABIPass>();
}

} // namespace skeleton
} // namespace mlir
