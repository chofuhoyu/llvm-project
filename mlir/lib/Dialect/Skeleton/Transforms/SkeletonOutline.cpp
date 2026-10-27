//===----------------------------------------------------------------------===//
//
// Outline skeleton operators into separately compilable op units.
//
// Each op unit becomes
//
//   func.func private @__skeleton_op_<id>(<tensor operands>) -> <results>
//       attributes {skeleton.op_id = <id>, skeleton.preference = ...}
//
// and the original position becomes a call to it. The function keeps the
// operator in its skeleton form, so the driver can later re-enter compilation
// from skeleton IR for that operator alone.
//
// Ids are assigned in module walk order. The same module is outlined the same
// way in the artifact build and when the driver re-reads the embedded IR, and
// the ids are stored on the function so nothing depends on reproducing the
// walk.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonAttrs.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonDialect.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir {
namespace skeleton {

#define GEN_PASS_DEF_SKELETONOUTLINE
#include "mlir/Dialect/Skeleton/Transforms/Passes.h.inc"

namespace {

/// Number of bytes an operator moves, when every shaped operand has a static
/// shape. Returns 0 when the size is not known at compile time.
int64_t computeOpBytes(Operation *op) {
  int64_t total = 0;
  for (Value operand : op->getOperands()) {
    auto shaped = dyn_cast<ShapedType>(operand.getType());
    if (!shaped)
      continue;
    if (!shaped.hasStaticShape())
      return 0;
    Type elementType = shaped.getElementType();
    if (!elementType.isIntOrFloat())
      return 0;
    total += (elementType.getIntOrFloatBitWidth() / 8) * shaped.getNumElements();
  }
  return total;
}

bool isOpUnit(Operation *op) {
  if (isa<MapOp, ReduceOp, VectorAddOp>(op))
    return true;
  if (isa<linalg::LinalgOp>(op) && op->hasAttr("skeleton.preference"))
    return true;
  return false;
}

struct SkeletonOutlinePass
    : public impl::SkeletonOutlineBase<SkeletonOutlinePass> {
  using SkeletonOutlineBase::SkeletonOutlineBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    // Collect in walk order; the id is the position in this list.
    SmallVector<Operation *> units;
    module.walk([&](Operation *op) {
      if (op->hasAttr("skeleton.op_id"))
        return;
      if (isOpUnit(op))
        units.push_back(op);
    });
    if (units.empty())
      return;

    for (auto [index, op] : llvm::enumerate(units)) {
      int64_t opId = static_cast<int64_t>(index);
      Location loc = op->getLoc();

      SmallVector<Value> operands = llvm::to_vector(op->getOperands());
      SmallVector<Type> operandTypes = llvm::to_vector(op->getOperandTypes());
      SmallVector<Type> resultTypes = llvm::to_vector(op->getResultTypes());

      auto funcType = FunctionType::get(ctx, operandTypes, resultTypes);
      std::string name = "__skeleton_op_" + Twine(opId).str();

      builder.setInsertionPointToEnd(module.getBody());
      auto outlined = func::FuncOp::create(builder, loc, name, funcType);
      outlined.setPrivate();
      outlined->setAttr("skeleton.op_id", builder.getI64IntegerAttr(opId));
      outlined->setAttr("skeleton.bytes",
                        builder.getI64IntegerAttr(computeOpBytes(op)));
      if (Attribute preference = op->getDiscardableAttr("skeleton.preference"))
        outlined->setDiscardableAttr("skeleton.preference", preference);

      Block *entryBlock = outlined.addEntryBlock();
      builder.setInsertionPointToStart(entryBlock);

      IRMapping mapper;
      for (unsigned i = 0; i < operands.size(); ++i)
        mapper.map(operands[i], entryBlock->getArgument(i));

      Operation *cloned = builder.clone(*op, mapper);

      builder.setInsertionPointAfter(cloned);
      func::ReturnOp::create(builder, loc, cloned->getResults());

      // Replace the original operator with a call to the op unit.
      builder.setInsertionPoint(op);
      auto call =
          func::CallOp::create(builder, loc, name, resultTypes, operands);
      op->replaceAllUsesWith(call.getResults());
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createSkeletonOutlinePass() {
  return std::make_unique<SkeletonOutlinePass>();
}

} // namespace skeleton
} // namespace mlir
