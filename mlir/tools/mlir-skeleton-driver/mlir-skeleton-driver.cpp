//===- mlir-skeleton-driver.cpp - runtime operator placement driver -------===//
//
// Entry point of the Skeleton runtime. It owns the compilation pipeline from
// skeleton IR down, places every operator at run time and executes the
// program in this process.
//
// Pipeline:
//   1. outline each operator unit into its own function (skeleton-outline);
//   2. split those functions into per-operator modules, leaving declarations
//      behind in the host module;
//   3. lower the host to the LLVM dialect and rewrite operator calls into
//      runtime resolves followed by indirect calls;
//   4. JIT the host and run the entry function;
//   5. when the runtime resolves an operator it has no implementation for, it
//      calls back into this driver, which compiles the operator from its
//      skeleton IR for the chosen device and hands the pointer back.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonAttrs.h"
#include "mlir/Dialect/Skeleton/Runtime/SkeletonRuntime.h"
#include "mlir/Dialect/Skeleton/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace mlir;

//===----------------------------------------------------------------------===//
// Options
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    entryPoint("entry", llvm::cl::desc("Entry function to invoke"),
               llvm::cl::init("main"));

static llvm::cl::list<std::string>
    sharedLibs("shared-libs",
               llvm::cl::desc("Shared libraries to resolve symbols from"),
               llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated);

static llvm::cl::opt<bool> traceDecisions(
    "trace", llvm::cl::desc("Trace placement decisions and compiles"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> dumpHostIR(
    "dump-host-ir",
    llvm::cl::desc("Print the lowered host module before execution"),
    llvm::cl::init(false));

//===----------------------------------------------------------------------===//
// Operator units
//===----------------------------------------------------------------------===//

namespace {

/// One operator unit: its skeleton IR plus whatever we compiled for it.
struct UnitInfo {
  int64_t id = 0;
  int32_t preference = SKELETON_DEVICE_CPU;
  int64_t bytes = 0;
  std::string symbol;
  /// Skeleton-level module holding the operator function. Kept as text so a
  /// compile always starts from the original IR.
  std::string ir;
  std::unique_ptr<ExecutionEngine> engine;
  llvm::DenseMap<int32_t, void *> impls;
};

struct DriverState {
  MLIRContext *context = nullptr;
  std::vector<std::unique_ptr<UnitInfo>> units;
  llvm::DenseMap<int64_t, UnitInfo *> byId;
  std::unique_ptr<ExecutionEngine> hostEngine;

  /// Backing storage for ExecutionEngineOptions::sharedLibPaths, which keeps
  /// an ArrayRef into this vector.
  SmallVector<StringRef> libs;
};

DriverState *driverState = nullptr;

/// The device a preference attribute asks for.
int32_t deviceFromPreference(Operation *op) {
  Attribute raw = op->getDiscardableAttr("skeleton.preference");
  auto preference = dyn_cast_or_null<skeleton::PreferenceAttr>(raw);
  if (!preference)
    return SKELETON_DEVICE_CPU;
  if (preference.getValue() == "GPU")
    return SKELETON_DEVICE_GPU;
  return SKELETON_DEVICE_CPU;
}

/// Collect every flat symbol referenced from `op` or anything nested in it.
void collectSymbolRefs(Operation *op, SmallVectorImpl<FlatSymbolRefAttr> &refs) {
  op->walk([&](Operation *inner) {
    for (NamedAttribute attr : inner->getAttrs())
      attr.getValue().walk(
          [&](Attribute value) {
            if (auto ref = dyn_cast<FlatSymbolRefAttr>(value))
              refs.push_back(ref);
          });
  });
}

//===----------------------------------------------------------------------===//
// Lowering
//===----------------------------------------------------------------------===//

/// Lower a module to the LLVM dialect. Host modules additionally get their
/// operator calls rewritten into runtime resolves.
LogicalResult lowerToLLVM(ModuleOp module, MLIRContext *context, bool isHost) {
  PassManager pm(context);

  pm.addPass(skeleton::createSkeletonToLinalgPass());

  bufferization::OneShotBufferizePassOptions bufferizeOptions;
  bufferizeOptions.bufferizeFunctionBoundaries = true;
  bufferizeOptions.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOptions));

  if (failed(parsePassPipeline(
          "convert-linalg-to-loops,convert-scf-to-cf,expand-strided-metadata,"
          "convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,"
          "convert-cf-to-llvm,reconcile-unrealized-casts",
          pm)))
    return failure();

  if (isHost)
    pm.addPass(skeleton::createSkeletonLowerDispatchPass());

  return pm.run(module);
}

//===----------------------------------------------------------------------===//
// JIT
//===----------------------------------------------------------------------===//

ExecutionEngineOptions engineOptions(DriverState &state, bool withSharedLibs) {
  ExecutionEngineOptions options;
  options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;
  if (withSharedLibs)
    options.sharedLibPaths = state.libs;
  return options;
}

/// Make the runtime entry points resolvable from JIT-compiled host code.
void registerRuntimeSymbols(ExecutionEngine &engine) {
  engine.registerSymbols([](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap map;
    auto add = [&](StringRef name, void *pointer) {
      map[interner(name)] = {llvm::orc::ExecutorAddr::fromPtr(pointer),
                             llvm::JITSymbolFlags::Exported};
    };
    add("skeleton_rt_declare", reinterpret_cast<void *>(&skeleton_rt_declare));
    add("skeleton_rt_register",
        reinterpret_cast<void *>(&skeleton_rt_register));
    add("skeleton_rt_resolve", reinterpret_cast<void *>(&skeleton_rt_resolve));
    add("skeleton_rt_probe", reinterpret_cast<void *>(&skeleton_rt_probe));
    add("skeleton_rt_set_device_available",
        reinterpret_cast<void *>(&skeleton_rt_set_device_available));
    add("skeleton_rt_set_provider",
        reinterpret_cast<void *>(&skeleton_rt_set_provider));
    add("skeleton_rt_set_trace",
        reinterpret_cast<void *>(&skeleton_rt_set_trace));
    add("skeleton_rt_dump_stats",
        reinterpret_cast<void *>(&skeleton_rt_dump_stats));
    return map;
  });
}

/// Compile one operator unit for one device, starting from its skeleton IR.
void *compileUnit(UnitInfo &unit, int32_t device) {
  DriverState &state = *driverState;

  if (void *existing = unit.impls.lookup(device))
    return existing;

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(unit.ir, state.context);
  if (!module) {
    llvm::errs() << "[skeleton-driver] op=" << unit.id
                 << ": cannot re-parse its skeleton IR\n";
    return nullptr;
  }

  // v1 only lowers operator units for the CPU. The GPU pipeline is added once
  // the per-operator device bridge is in place.
  if (device != SKELETON_DEVICE_CPU) {
    llvm::errs() << "[skeleton-driver] op=" << unit.id
                 << ": device " << device << " is not supported yet\n";
    return nullptr;
  }

  if (failed(lowerToLLVM(*module, state.context, /*isHost=*/false)))
    return nullptr;

  auto engineOrErr = ExecutionEngine::create(*module, engineOptions(state, false));
  if (!engineOrErr) {
    llvm::errs() << "[skeleton-driver] op=" << unit.id
                 << ": JIT failed: " << llvm::toString(engineOrErr.takeError())
                 << "\n";
    return nullptr;
  }
  std::unique_ptr<ExecutionEngine> engine = std::move(*engineOrErr);

  auto symbol = engine->lookup(unit.symbol);
  if (!symbol) {
    llvm::errs() << "[skeleton-driver] op=" << unit.id
                 << ": lookup failed: " << llvm::toString(symbol.takeError())
                 << "\n";
    return nullptr;
  }

  unit.engine = std::move(engine);
  unit.impls[device] = *symbol;
  return *symbol;
}

/// Provider installed in the runtime: called on a cache miss.
void *provideImplementation(int64_t opId, int32_t device) {
  auto it = driverState->byId.find(opId);
  if (it == driverState->byId.end())
    return nullptr;
  return compileUnit(*it->second, device);
}

//===----------------------------------------------------------------------===//
// Splitting
//===----------------------------------------------------------------------===//

/// Move every outlined operator function into its own module and leave a
/// declaration in the host module. References from the operator (pure
/// functions) are cloned along with it so the unit module is self-contained.
LogicalResult splitUnits(ModuleOp module, DriverState &state) {
  SmallVector<func::FuncOp> unitFunctions;
  for (func::FuncOp func : module.getOps<func::FuncOp>())
    if (func->hasAttr("skeleton.op_id"))
      unitFunctions.push_back(func);

  if (unitFunctions.empty()) {
    llvm::errs() << "[skeleton-driver] no operator units found\n";
    return failure();
  }

  MLIRContext *context = module.getContext();
  for (func::FuncOp unitFunc : unitFunctions) {
    auto info = std::make_unique<UnitInfo>();
    info->id = unitFunc->getAttrOfType<IntegerAttr>("skeleton.op_id").getInt();
    info->preference = deviceFromPreference(unitFunc);
    if (auto bytes = unitFunc->getAttrOfType<IntegerAttr>("skeleton.bytes"))
      info->bytes = bytes.getInt();
    info->symbol = unitFunc.getSymName().str();

    SmallVector<FlatSymbolRefAttr> refs;
    collectSymbolRefs(unitFunc, refs);

    ModuleOp unitModule = ModuleOp::create(unitFunc.getLoc());
    {
      OpBuilder builder(context);
      builder.setInsertionPointToStart(unitModule.getBody());
      builder.clone(*unitFunc);

      llvm::DenseSet<StringRef> cloned;
      std::function<void(FlatSymbolRefAttr)> cloneReference =
          [&](FlatSymbolRefAttr ref) {
            if (!cloned.insert(ref.getValue()).second)
              return;
            auto callee = module.lookupSymbol<func::FuncOp>(ref);
            if (!callee)
              return;
            SmallVector<FlatSymbolRefAttr> nested;
            collectSymbolRefs(callee, nested);
            builder.clone(*callee);
            for (FlatSymbolRefAttr inner : nested)
              cloneReference(inner);
          };
      for (FlatSymbolRefAttr ref : refs)
        cloneReference(ref);
    }

    std::string text;
    llvm::raw_string_ostream stream(text);
    unitModule.print(stream);
    stream.flush();
    unitModule.getOperation()->destroy();
    info->ir = std::move(text);

    // Leave a declaration in the host; the dispatch lowering removes it once
    // the call has been rewritten.
    unitFunc.getBody().dropAllReferences();
    unitFunc.getBody().getBlocks().clear();

    state.byId[info->id] = info.get();
    state.units.push_back(std::move(info));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Execution
//===----------------------------------------------------------------------===//

FailureOr<int> runDriver(ModuleOp module, DriverState &state) {
  {
    PassManager pm(state.context);
    pm.addPass(skeleton::createSkeletonOutlinePass());
    if (failed(pm.run(module)))
      return failure();
  }

  if (failed(splitUnits(module, state)))
    return failure();

  if (failed(lowerToLLVM(module, state.context, /*isHost=*/true)))
    return failure();

  if (dumpHostIR) {
    module.print(llvm::errs());
    llvm::errs() << "\n";
  }

  auto engineOrErr =
      ExecutionEngine::create(module, engineOptions(state, true));
  if (!engineOrErr) {
    llvm::errs() << "host JIT failed: "
                 << llvm::toString(engineOrErr.takeError()) << "\n";
    return failure();
  }
  state.hostEngine = std::move(*engineOrErr);
  registerRuntimeSymbols(*state.hostEngine);
  state.hostEngine->initialize();

  skeleton_rt_set_trace(traceDecisions ? 1 : 0);
  skeleton_rt_set_provider(&provideImplementation);
  for (auto &unit : state.units)
    skeleton_rt_declare(unit->id, unit->preference, unit->bytes);

  void *empty = nullptr;
  if (llvm::Error error = state.hostEngine->invokePacked(entryPoint, {&empty})) {
    llvm::errs() << "execution failed: " << llvm::toString(std::move(error))
                 << "\n";
    return failure();
  }

  if (traceDecisions)
    skeleton_rt_dump_stats();
  return 0;
}

} // namespace

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Skeleton runtime placement driver\n");

  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllToLLVMIRTranslations(registry);
  registerAllPasses();

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(inputFilename, &context);
  if (!module) {
    llvm::errs() << "cannot parse " << inputFilename << "\n";
    return 1;
  }

  DriverState state;
  state.context = &context;
  for (const std::string &lib : sharedLibs)
    state.libs.push_back(lib);
  driverState = &state;

  FailureOr<int> result = runDriver(*module, state);
  if (failed(result))
    return 1;
  return *result;
}
