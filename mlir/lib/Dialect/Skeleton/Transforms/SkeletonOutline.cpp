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
#include "llvm/ADT/STLExtras.h"

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
  SmallVector<Value> seen;
  for (Value operand : op->getOperands()) {
    // Repeated operands (e.g. the same tensor used as two inputs) describe one
    // allocation, so count each value once. The result is an advisory lower
    // bound, not the operator's peak working set.
    if (llvm::is_contained(seen, operand))
      continue;
    seen.push_back(operand);
    auto shaped = dyn_cast<ShapedType>(operand.getType());
    if (!shaped)
      continue;
    if (!shaped.hasStaticShape())
      return 0;
    Type elementType = shaped.getElementType();
    if (!elementType.isIntOrFloat())
      return 0;
    total += ((elementType.getIntOrFloatBitWidth() + 7) / 8) *
             shaped.getNumElements();
  }
  return total;
}

/// The placement preference of an operator. Skeleton operators carry it as a
/// registered attribute; a linalg operator carries it as a discardable one.
Attribute getPreference(Operation *op) {
  if (auto map = dyn_cast<MapOp>(op))
    return map.getPreferenceAttr();
  if (auto reduce = dyn_cast<ReduceOp>(op))
    return reduce.getPreferenceAttr();
  if (auto add = dyn_cast<VectorAddOp>(op))
    return add.getPreferenceAttr();
  return op->getDiscardableAttr("skeleton.preference");
}

bool isOpUnit(Operation *op) {
  // Already-outlined modules (the IR embedded in an artifact) keep their
  // operators inside functions carrying skeleton.op_id; outlining them again
  // would nest units inside units.
  if (auto parent = op->getParentOfType<func::FuncOp>())
    if (parent->hasAttr("skeleton.op_id"))
      return false;
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
    ModuleOp module = cast<ModuleOp>(getOperation());
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

    // The host dispatch rewrites one result at a time and the GPU bridge has a
    // single output memref; reject multi-result units before creating calls
    // that the rest of the pipeline cannot handle.
    for (Operation *op : units) {
      if (op->getNumResults() > 1) {
        op->emitError(
            "operator units with more than one result are not supported");
        signalPassFailure();
        return;
      }
    }

    for (auto [index, op] : llvm::enumerate(units)) {
      int64_t opId = static_cast<int64_t>(index);
      Location loc = op->getLoc();

      SmallVector<Value> operands = llvm::to_vector(op->getOperands());
      SmallVector<Type> operandTypes = llvm::to_vector(op->getOperandTypes());
      SmallVector<Type> resultTypes = llvm::to_vector(op->getResultTypes());

      auto funcType = FunctionType::get(ctx, operandTypes, resultTypes);
      std::string name = "__skeleton_op_" + Twine(opId).str();
      if (SymbolTable::lookupSymbolIn(module, name)) {
        op->emitError("cannot outline operator: function '")
            << name << "' already exists";
        signalPassFailure();
        return;
      }

      builder.setInsertionPointToEnd(module.getBody());
      auto outlined = func::FuncOp::create(builder, loc, name, funcType);
      outlined.setPrivate();
      outlined->setAttr("skeleton.op_id", builder.getI64IntegerAttr(opId));
      outlined->setAttr("skeleton.bytes",
                        builder.getI64IntegerAttr(computeOpBytes(op)));
      if (Attribute preference = getPreference(op))
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
