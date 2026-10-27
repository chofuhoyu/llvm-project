//===----------------------------------------------------------------------===//
//
// Rewrite calls to outlined op units into runtime resolves plus indirect calls.
//
// After the module is lowered to the LLVM dialect, a call to
//
//   llvm.call @__skeleton_op_0(%a, %b, %c)
//
// becomes
//
//   %impl = llvm.call @skeleton_rt_resolve(%id) : (i64) -> !llvm.ptr
//   llvm.call %impl(%a, %b, %c) : !llvm.ptr, (<args>) -> (<results>)
//
// The indirect call keeps the operator's real signature, so CPU and GPU
// implementations are interchangeable without the runtime having to interpret
// arguments. The op unit declaration is erased when nothing else refers to it
// (in the driver it lives in a separate module; in an artifact the start-up
// registration still takes its address).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir {
namespace skeleton {

#define GEN_PASS_DEF_SKELETONLOWERDISPATCH
#include "mlir/Dialect/Skeleton/Transforms/Passes.h.inc"

namespace {

constexpr StringLiteral kOpUnitPrefix = "__skeleton_op_";
constexpr StringLiteral kResolveName = "skeleton_rt_resolve";

struct SkeletonLowerDispatchPass
    : public impl::SkeletonLowerDispatchBase<SkeletonLowerDispatchPass> {
  using SkeletonLowerDispatchBase::SkeletonLowerDispatchBase;

  void runOnOperation() override {
    ModuleOp module = cast<ModuleOp>(getOperation());
    MLIRContext *ctx = &getContext();

    LLVM::LLVMFuncOp resolveDecl =
        module.lookupSymbol<LLVM::LLVMFuncOp>(kResolveName);
    if (!resolveDecl) {
      OpBuilder builder(ctx);
      builder.setInsertionPointToStart(module.getBody());
      auto funcType = LLVM::LLVMFunctionType::get(
          LLVM::LLVMPointerType::get(ctx), builder.getI64Type());
      resolveDecl = LLVM::LLVMFuncOp::create(builder, module.getLoc(),
                                             kResolveName, funcType);
    }

    SmallVector<LLVM::CallOp> calls;
    module.walk([&](LLVM::CallOp call) {
      std::optional<StringRef> callee = call.getCallee();
      if (!callee)
        return;
      if (!callee->starts_with(kOpUnitPrefix))
        return;
      calls.push_back(call);
    });

    for (LLVM::CallOp call : calls) {
      StringRef calleeName = *call.getCallee();
      StringRef idText = calleeName.drop_front(kOpUnitPrefix.size());
      int64_t opId = 0;
      if (idText.getAsInteger(10, opId)) {
        call.emitError("malformed op unit name '") << calleeName << "'";
        signalPassFailure();
        return;
      }
      if (call.getNumResults() > 1) {
        call.emitError("op units with more than one result are not supported");
        signalPassFailure();
        return;
      }

      OpBuilder builder(call);
      Location loc = call.getLoc();

      Value idValue =
          LLVM::ConstantOp::create(builder, loc, builder.getI64Type(),
                                   builder.getI64IntegerAttr(opId));
      Value impl = LLVM::CallOp::create(
                       builder, loc,
                       TypeRange{LLVM::LLVMPointerType::get(ctx)},
                       FlatSymbolRefAttr::get(ctx, kResolveName),
                       ValueRange{idValue})
                       .getResult();

      SmallVector<Value> operands;
      operands.push_back(impl);
      llvm::append_range(operands, call.getArgOperands());

      auto indirect = LLVM::CallOp::create(
          builder, loc, call.getResultTypes(), operands,
          call->getDiscardableAttrDictionary().getValue());
      indirect.getProperties().operandSegmentSizes = {
          static_cast<int32_t>(operands.size()), 0};
      indirect.getProperties().op_bundle_sizes =
          builder.getDenseI32ArrayAttr({});

      call->replaceAllUsesWith(indirect->getResults());
      call.erase();
    }

    // Drop op unit declarations that nothing refers to any more.
    SmallVector<LLVM::LLVMFuncOp> deadUnits;
    module.walk([&](LLVM::LLVMFuncOp func) {
      if (!func.getSymName().starts_with(kOpUnitPrefix))
        return;
      if (!func.isExternal())
        return;
      auto uses = SymbolTable::getSymbolUses(func, module);
      if (uses && uses->empty())
        deadUnits.push_back(func);
    });
    for (LLVM::LLVMFuncOp func : deadUnits)
      func.erase();
  }
};

} // namespace

std::unique_ptr<Pass> createSkeletonLowerDispatchPass() {
  return std::make_unique<SkeletonLowerDispatchPass>();
}

} // namespace skeleton
} // namespace mlir
