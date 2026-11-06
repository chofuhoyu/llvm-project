// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonAttrs.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonOps.h"
#include "mlir/Dialect/Skeleton/Runtime/SkeletonRuntime.h"
#include "mlir/Dialect/Skeleton/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace mlir;

// Options

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

static llvm::cl::opt<bool>
    traceDecisions("trace",
                   llvm::cl::desc("Trace placement decisions and compiles"),
                   llvm::cl::init(false));

static llvm::cl::opt<std::string>
    emitPath("emit",
             llvm::cl::desc("Emit a standalone program artifact at this path"),
             llvm::cl::init(""));

static llvm::cl::opt<std::string> runtimeLib(
    "runtime-lib",
    llvm::cl::desc("Static runtime archive linked into an emitted artifact"),
    llvm::cl::init(""));

static llvm::cl::list<std::string> linkArgs(
    "link-arg",
    llvm::cl::desc("Extra argument passed to the linker when emitting"),
    llvm::cl::ZeroOrMore);

static llvm::cl::opt<std::string>
    linkerPath("linker",
               llvm::cl::desc("Linker driver used to produce an artifact"),
               llvm::cl::init("clang"));

static llvm::cl::opt<bool>
    dumpHostIR("dump-host-ir",
               llvm::cl::desc("Print the lowered host module before execution"),
               llvm::cl::init(false));

static llvm::cl::opt<bool>
    dumpUnitIR("dump-unit-ir",
               llvm::cl::desc("Print every operator unit after lowering it for "
                              "a device, before it is JIT'ed"),
               llvm::cl::init(false));

// Operator units

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
  /// One JIT engine per device. Engines are never replaced: a function pointer
  /// handed to the runtime must stay valid for the life of the unit, even
  /// after the same unit is compiled for another device.
  llvm::DenseMap<int32_t, std::unique_ptr<ExecutionEngine>> engines;
  llvm::DenseMap<int32_t, void *> impls;
  /// Devices whose compilation already failed. Retrying on every operator
  /// invocation would repeat a full parse/lower/JIT cycle and spam errors.
  llvm::DenseSet<int32_t> failed;
};

struct DriverState {
  MLIRContext *context = nullptr;
  std::vector<std::unique_ptr<UnitInfo>> units;
  llvm::DenseMap<int64_t, UnitInfo *> byId;
  std::unique_ptr<ExecutionEngine> hostEngine;
  /// Lowered signatures of the host's op unit declarations, captured before
  /// skeleton-lower-dispatch erases them. A lazily compiled unit must match
  /// the signature the host already calls it with.
  llvm::StringMap<LLVM::LLVMFunctionType> hostSignatures;

  /// Backing storage for ExecutionEngineOptions::sharedLibPaths, which keeps
  /// an ArrayRef into this vector.
  SmallVector<StringRef> libs;

  std::string runtimeArchive;
  std::string linker;
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
void collectSymbolRefs(Operation *op,
                       SmallVectorImpl<FlatSymbolRefAttr> &refs) {
  op->walk([&](Operation *inner) {
    for (NamedAttribute attr : inner->getAttrs())
      attr.getValue().walk([&](Attribute value) {
        if (auto ref = dyn_cast<FlatSymbolRefAttr>(value))
          refs.push_back(ref);
      });
  });
}

// Lowering

/// Lower a module to the LLVM dialect. Host modules additionally get their
/// operator calls rewritten into runtime resolves. When \p hostSignatures is
/// provided, the lowered type of every outlined operator declaration is
/// recorded before the dispatch pass erases the declarations, so a lazily
/// compiled unit can be checked against the ABI the host already calls it with.
/// Whether the module still has tensor-level work for skeleton-to-linalg and
/// bufferization. A host that the C++ front end already lowered to the LLVM
/// dialect has none, and running those passes over it would only give the
/// bufferizer a module it has nothing to do with.
bool hasTensorLevelOps(ModuleOp module) {
  bool found = false;
  module.walk([&](Operation *op) {
    if (found)
      return WalkResult::interrupt();
    if (isa<skeleton::MapOp, skeleton::ReduceOp, skeleton::VectorAddOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    for (Type type : op->getOperandTypes())
      if (isa<TensorType>(type)) {
        found = true;
        return WalkResult::interrupt();
      }
    for (Type type : op->getResultTypes())
      if (isa<TensorType>(type)) {
        found = true;
        return WalkResult::interrupt();
      }
    return WalkResult::advance();
  });
  return found;
}

LogicalResult
lowerToLLVM(ModuleOp module, MLIRContext *context, bool isHost,
            llvm::StringMap<LLVM::LLVMFunctionType> *hostSignatures = nullptr) {
  PassManager pm(context);

  bool tensorLevel = hasTensorLevelOps(module);
  if (tensorLevel) {
    pm.addPass(skeleton::createSkeletonToLinalgPass());
    // An operator unit computes into a fresh tensor and materializes it in the
    // destination buffer. Empty tensor elimination turns that into computing
    // directly in the destination, which is what makes the operator's write
    // reachable from a device kernel.
    pm.addPass(bufferization::createEmptyTensorEliminationPass());

    bufferization::OneShotBufferizePassOptions bufferizeOptions;
    bufferizeOptions.bufferizeFunctionBoundaries = true;
    bufferizeOptions.functionBoundaryTypeConversion =
        bufferization::LayoutMapOption::IdentityLayoutMap;
    pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOptions));
  }

  if (failed(parsePassPipeline(
          "convert-linalg-to-loops,convert-scf-to-cf,expand-strided-metadata,"
          "convert-arith-to-llvm,convert-math-to-llvm,finalize-memref-to-llvm,"
          "convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts",
          pm)))
    return failure();

  if (failed(pm.run(module)))
    return failure();

  // clang emits an implicit template instantiation as `available_externally`: a
  // normal build either inlines it or emits it into the object file, and a
  // linker resolves it from another translation unit. This module is compiled on
  // its own -- JIT'ed, or emitted and linked against libraries that do not
  // export those inline bodies -- so the definition has to stay. `linkonce_odr`
  // keeps it, and still allows several units to carry the same one.
  for (LLVM::LLVMFuncOp func : module.getOps<LLVM::LLVMFuncOp>())
    if (!func.isExternal() &&
        func.getLinkage() == LLVM::Linkage::AvailableExternally)
      func.setLinkage(LLVM::Linkage::LinkonceODR);

  if (!isHost)
    return success();

  if (hostSignatures) {
    for (LLVM::LLVMFuncOp func : module.getOps<LLVM::LLVMFuncOp>())
      if (func.getSymName().starts_with("__skeleton_op_"))
        (*hostSignatures)[func.getSymName()] = func.getFunctionType();
  }

  // The ABI thunks the C++ front end declared are filled in once the host is
  // LLVM; they call the operator unit directly, so the dispatch lowering below
  // still turns every operator call into one resolve plus one indirect call.
  PassManager dispatchPm(context);
  dispatchPm.addPass(skeleton::createSkeletonLowerHostABIPass());
  dispatchPm.addPass(skeleton::createSkeletonLowerDispatchPass());
  return dispatchPm.run(module);
}

// GPU operator units

/// Turn the op unit module into the shape skeleton-target-lower expects: an
/// implementation marked as a GPU target plus a wrapper with the original
/// name. The wrapper is what the host calls; target-lower puts the UVM bridge
/// around its call, so the host stays device-agnostic.
LogicalResult prepareGpuUnit(ModuleOp module, const UnitInfo &unit) {
  MLIRContext *context = module.getContext();
  auto impl = module.lookupSymbol<func::FuncOp>(unit.symbol);
  if (!impl) {
    llvm::errs() << "op unit '" << unit.symbol << "' not found\n";
    return failure();
  }

  FunctionType signature = impl.getFunctionType();
  impl.setName(unit.symbol + "_impl");
  impl->setAttr(
      "skeleton.target",
      skeleton::TargetAttr::get(context, StringAttr::get(context, "GPU")));

  OpBuilder builder(context);
  builder.setInsertionPoint(impl);
  auto wrapper =
      func::FuncOp::create(builder, impl.getLoc(), unit.symbol, signature);
  wrapper.setPrivate();
  Block *block = wrapper.addEntryBlock();
  builder.setInsertionPointToStart(block);
  auto call =
      func::CallOp::create(builder, impl.getLoc(), impl.getSymName(),
                           signature.getResults(), block->getArguments());
  func::ReturnOp::create(builder, impl.getLoc(), call.getResults());
  return success();
}

/// Lower a GPU op unit: skeleton form, bufferization and UVM bridge through
/// target-lower, then the NVVM pipeline that produces the cubin and the host
/// side launch code.
LogicalResult lowerGpuUnit(ModuleOp module, const UnitInfo &unit) {
  MLIRContext *context = module.getContext();
  if (failed(prepareGpuUnit(module, unit)))
    return failure();

  PassManager pm(context);
  pm.addPass(skeleton::createSkeletonToLinalgPass());
  // Same reason as on the CPU: the operator computes into a fresh tensor and
  // materializes the result in its destination buffer. A device kernel cannot
  // reach a host temporary, so the computation has to be rewritten to write the
  // destination directly before the GPU bridge runs bufferization.
  pm.addPass(bufferization::createEmptyTensorEliminationPass());
  pm.addPass(skeleton::createSkeletonTargetLowerPass());
  if (failed(parsePassPipeline(
          "func.func(gpu-map-parallel-loops,convert-parallel-loops-to-gpu),"
          "gpu-lower-to-nvvm-pipeline{cubin-format=fatbin}",
          pm)))
    return failure();
  return pm.run(module);
}

// JIT

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
    add("skeleton_rt_sync", reinterpret_cast<void *>(&skeleton_rt_sync));
    add("skeleton_rt_alloc", reinterpret_cast<void *>(&skeleton_rt_alloc));
    add("skeleton_rt_free", reinterpret_cast<void *>(&skeleton_rt_free));
    add("skeleton_rt_executor_id",
        reinterpret_cast<void *>(&skeleton_rt_executor_id));
    add("skeleton_rt_executor_count",
        reinterpret_cast<void *>(&skeleton_rt_executor_count));
    add("skeleton_rt_set_device_available",
        reinterpret_cast<void *>(&skeleton_rt_set_device_available));
    add("skeleton_rt_set_free_bytes",
        reinterpret_cast<void *>(&skeleton_rt_set_free_bytes));
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

  // A failed compilation is not retried: the runtime resolves this operator on
  // every call, so retrying would repeat a full parse/lower/JIT cycle and spam
  // diagnostics. The runtime falls back to another device when one exists.
  if (unit.failed.contains(device))
    return nullptr;

  auto fail = [&](StringRef message) -> void * {
    llvm::errs() << "[skeleton-driver] op=" << unit.id << ": " << message
                 << "\n";
    unit.failed.insert(device);
    return nullptr;
  };

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(unit.ir, state.context);
  if (!module)
    return fail("cannot re-parse its skeleton IR");

  if (device == SKELETON_DEVICE_GPU) {
    if (failed(lowerGpuUnit(*module, unit)))
      return fail("cannot lower the unit for the GPU");
  } else if (device == SKELETON_DEVICE_CPU) {
    if (failed(lowerToLLVM(*module, state.context, /*isHost=*/false)))
      return fail("cannot lower the unit for the CPU");
  } else {
    return fail(("unknown device " + Twine(device)).str());
  }

  // The host already calls this symbol with a fixed ABI. Bufferization and
  // lowering happen separately for the host and the unit, so verify that the
  // lazily compiled function has exactly the signature the host expects.
  auto unitFunc = module->lookupSymbol<LLVM::LLVMFuncOp>(unit.symbol);
  if (!unitFunc)
    return fail("the lowered module has no unit function");
  auto hostSignature = state.hostSignatures.find(unit.symbol);
  if (hostSignature == state.hostSignatures.end())
    return fail("the host has no signature for this unit");
  if (unitFunc.getFunctionType() != hostSignature->second) {
    llvm::errs() << "[skeleton-driver] op=" << unit.id << ": ABI mismatch for '"
                 << unit.symbol << "'\n  host: ";
    llvm::errs() << hostSignature->second
                 << "\n  unit: " << unitFunc.getFunctionType() << "\n";
    unit.failed.insert(device);
    return nullptr;
  }

  if (dumpUnitIR) {
    llvm::errs() << "[skeleton-driver] op=" << unit.id << " device="
                 << (device == SKELETON_DEVICE_GPU ? "GPU" : "CPU")
                 << " lowered unit IR:\n";
    module->print(llvm::errs());
    llvm::errs() << "\n";
  }

  auto engineOrErr = ExecutionEngine::create(
      *module, engineOptions(state, /*withSharedLibs=*/true));
  if (!engineOrErr) {
    std::string message =
        "JIT failed: " + llvm::toString(engineOrErr.takeError());
    return fail(message);
  }
  std::unique_ptr<ExecutionEngine> engine = std::move(*engineOrErr);
  // A GPU unit calls back into the runtime to synchronize the device before the
  // host reads a buffer the kernel wrote, so the unit's engine needs the same
  // entry points the host engine has.
  registerRuntimeSymbols(*engine);
  engine->initialize();

  auto symbol = engine->lookup(unit.symbol);
  if (!symbol) {
    std::string message =
        "lookup failed: " + llvm::toString(symbol.takeError());
    return fail(message);
  }

  // Keep every device's engine alive: the runtime may still hold a pointer
  // into an engine compiled for another device.
  unit.engines[device] = std::move(engine);
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

// Splitting

/// Move every outlined operator function into its own module and leave a
/// declaration in the host module. References from the operator (pure
/// functions and globals) are cloned along with it so the unit module is
/// self-contained.
LogicalResult splitUnits(ModuleOp module, DriverState &state) {
  SmallVector<func::FuncOp> unitFunctions;
  for (func::FuncOp func : module.getOps<func::FuncOp>())
    if (func->hasAttr("skeleton.op_id"))
      unitFunctions.push_back(func);

  // A program without operator units is still a valid host program. Leave the
  // unit table empty and let the host run on its own.
  if (unitFunctions.empty())
    return success();

  MLIRContext *context = module.getContext();
  for (func::FuncOp unitFunc : unitFunctions) {
    auto idAttr = unitFunc->getAttrOfType<IntegerAttr>("skeleton.op_id");
    if (!idAttr) {
      llvm::errs() << "[skeleton-driver] op unit '" << unitFunc.getSymName()
                   << "' has no integer skeleton.op_id\n";
      return failure();
    }
    auto info = std::make_unique<UnitInfo>();
    info->id = idAttr.getInt();
    info->preference = deviceFromPreference(unitFunc);
    if (auto bytes = unitFunc->getAttrOfType<IntegerAttr>("skeleton.bytes"))
      info->bytes = bytes.getInt();
    info->symbol = unitFunc.getSymName().str();

    SmallVector<FlatSymbolRefAttr> refs;
    collectSymbolRefs(unitFunc, refs);

    ModuleOp unitModule = ModuleOp::create(unitFunc.getLoc());
    // Module-level attributes (data layout specifications and similar) affect
    // bufferization and lowering; the unit must see the same ones as the host.
    unitModule->setAttrs(module->getAttrs());
    {
      OpBuilder builder(context);
      builder.setInsertionPointToStart(unitModule.getBody());
      builder.clone(*unitFunc);

      llvm::DenseSet<StringRef> cloned;
      std::function<LogicalResult(FlatSymbolRefAttr)> cloneReference =
          [&](FlatSymbolRefAttr ref) -> LogicalResult {
        if (!cloned.insert(ref.getValue()).second)
          return success();
        Operation *refOp = SymbolTable::lookupSymbolIn(module, ref.getValue());
        if (!refOp)
          return success();
        if (auto callee = dyn_cast<func::FuncOp>(refOp)) {
          // The callee's body lives in that unit's own module, so cloning it
          // here would copy an empty declaration and the unit would fail to
          // link at JIT time.
          if (callee->hasAttr("skeleton.op_id"))
            return unitFunc.emitError()
                   << "operator unit '" << unitFunc.getSymName()
                   << "' calls operator unit '" << callee.getSymName() << "'";
          SmallVector<FlatSymbolRefAttr> nested;
          collectSymbolRefs(callee, nested);
          builder.clone(*callee);
          for (FlatSymbolRefAttr inner : nested)
            if (failed(cloneReference(inner)))
              return failure();
        } else {
          // Globals (memref.global, llvm.mlir.global, ...) are referenced
          // by name from the unit, so they must travel with it.
          builder.clone(*refOp);
        }
        return success();
      };
      for (FlatSymbolRefAttr ref : refs)
        if (failed(cloneReference(ref)))
          return failure();
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

    if (state.byId.count(info->id)) {
      unitFunc.emitError() << "operator unit '" << unitFunc.getSymName()
                           << "' has a duplicate skeleton.op_id (" << info->id
                           << ")";
      return failure();
    }
    state.byId[info->id] = info.get();
    state.units.push_back(std::move(info));
  }
  return success();
}

// Execution

FailureOr<int> runDriver(ModuleOp module, DriverState &state) {
  // The runtime only supports a C-style entry point: no arguments. Its result,
  // if it has one, is the process exit code and is reported by the artifact; in
  // process it is ignored. A C++ `main` reaches here as an llvm.func, a
  // hand-written skeleton program as a func.func.
  Operation *entry = module.lookupSymbol(entryPoint);
  if (!entry) {
    llvm::errs() << "entry function '" << entryPoint << "' not found\n";
    return failure();
  }
  if (auto funcEntry = dyn_cast<func::FuncOp>(entry)) {
    if (funcEntry.getNumArguments() != 0) {
      llvm::errs() << "entry function '" << entryPoint
                   << "' must take no arguments\n";
      return failure();
    }
  } else if (auto llvmEntry = dyn_cast<LLVM::LLVMFuncOp>(entry)) {
    if (llvmEntry.getFunctionType().getNumParams() != 0) {
      llvm::errs() << "entry function '" << entryPoint
                   << "' must take no arguments\n";
      return failure();
    }
  } else {
    llvm::errs() << "entry function '" << entryPoint
                 << "' is neither a func.func nor an llvm.func\n";
    return failure();
  }

  {
    PassManager pm(state.context);
    pm.addPass(skeleton::createSkeletonOutlinePass());
    if (failed(pm.run(module)))
      return failure();
  }

  if (failed(splitUnits(module, state)))
    return failure();

  if (failed(lowerToLLVM(module, state.context, /*isHost=*/true,
                         &state.hostSignatures)))
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

  // Only force tracing on when the flag is given. Calling set_trace(0) here
  // would mark the environment variable as already read and hide
  // SKELETON_TRACE=1 in the driver.
  if (traceDecisions)
    skeleton_rt_set_trace(1);
  skeleton_rt_set_provider(&provideImplementation);
  for (auto &unit : state.units)
    skeleton_rt_declare(unit->id, unit->preference, unit->bytes);

  // `invokePacked` calls a packed wrapper that reads one pointer per argument
  // out of the array and writes the function's result to the slot right after
  // them. A C++ `main` returns its exit code, so that slot has to exist even
  // though the driver ignores the value; without it the wrapper stores through
  // a pointer read past the end of the array.
  uint64_t entryResult = 0;
  SmallVector<void *> entryArgs;
  if (auto entry = cast<LLVM::LLVMFuncOp>(module.lookupSymbol(entryPoint));
      !isa<LLVM::LLVMVoidType>(entry.getFunctionType().getReturnType()))
    entryArgs.push_back(&entryResult);

  if (llvm::Error error =
          state.hostEngine->invokePacked(entryPoint, entryArgs)) {
    llvm::errs() << "execution failed: " << llvm::toString(std::move(error))
                 << "\n";
    return failure();
  }

  if (traceDecisions)
    skeleton_rt_dump_stats();
  return static_cast<int>(entryResult);
}

} // namespace

// Artifact emission

namespace {

/// Metadata captured from an op unit before lowering drops the attributes.
struct UnitMeta {
  int64_t id = 0;
  int32_t preference = SKELETON_DEVICE_CPU;
  int64_t bytes = 0;
  std::string symbol;
};

std::optional<std::string> captureSkeletonIR(ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  if (failed(writeBytecodeToFile(module, stream)))
    return std::nullopt;
  stream.flush();
  if (text.empty())
    return std::nullopt;
  return text;
}

void addRuntimeDeclarations(OpBuilder &builder, ModuleOp module, Location loc) {
  MLIRContext *ctx = builder.getContext();
  auto i64Ty = builder.getI64Type();
  auto i32Ty = builder.getI32Type();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto voidTy = LLVM::LLVMVoidType::get(ctx);

  builder.setInsertionPointToEnd(module.getBody());
  LLVM::LLVMFuncOp::create(
      builder, loc, "skeleton_rt_declare",
      LLVM::LLVMFunctionType::get(voidTy, {i64Ty, i32Ty, i64Ty}));
  LLVM::LLVMFuncOp::create(
      builder, loc, "skeleton_rt_register",
      LLVM::LLVMFunctionType::get(voidTy, {i64Ty, i32Ty, ptrTy}));
}

/// Generate skeleton_program_init: declare every op unit and register the CPU
/// implementation that was linked into the artifact.
LogicalResult addProgramInit(OpBuilder &builder, ModuleOp module,
                             ArrayRef<UnitMeta> units, Location loc) {
  MLIRContext *ctx = builder.getContext();
  auto i64Ty = builder.getI64Type();
  auto i32Ty = builder.getI32Type();
  auto voidTy = LLVM::LLVMVoidType::get(ctx);

  builder.setInsertionPointToEnd(module.getBody());
  auto init = LLVM::LLVMFuncOp::create(builder, loc, "skeleton_program_init",
                                       LLVM::LLVMFunctionType::get(voidTy, {}));
  Block *block = init.addEntryBlock(builder);
  builder.setInsertionPointToStart(block);

  for (const UnitMeta &unit : units) {
    Value id = LLVM::ConstantOp::create(builder, loc, i64Ty,
                                        builder.getI64IntegerAttr(unit.id));
    Value preference = LLVM::ConstantOp::create(
        builder, loc, i32Ty, builder.getI32IntegerAttr(unit.preference));
    Value bytes = LLVM::ConstantOp::create(
        builder, loc, i64Ty, builder.getI64IntegerAttr(unit.bytes));
    LLVM::CallOp::create(builder, loc, TypeRange{},
                         FlatSymbolRefAttr::get(ctx, "skeleton_rt_declare"),
                         ValueRange{id, preference, bytes});

    auto impl = module.lookupSymbol<LLVM::LLVMFuncOp>(unit.symbol);
    if (!impl) {
      llvm::errs() << "op unit '" << unit.symbol
                   << "' disappeared before emission\n";
      return failure();
    }
    Value cpu = LLVM::ConstantOp::create(
        builder, loc, i32Ty, builder.getI32IntegerAttr(SKELETON_DEVICE_CPU));
    Value address = LLVM::AddressOfOp::create(builder, loc, impl);
    LLVM::CallOp::create(builder, loc, TypeRange{},
                         FlatSymbolRefAttr::get(ctx, "skeleton_rt_register"),
                         ValueRange{id, cpu, address});
  }
  LLVM::ReturnOp::create(builder, loc, ValueRange{});
  return success();
}

/// Generate the C entry point: run the registration, then the program. An
/// entry that returns a value -- a C++ `main` returns an int -- gives the
/// process its exit code.
void addMain(OpBuilder &builder, ModuleOp module, StringRef entryName,
             Location loc) {
  MLIRContext *ctx = builder.getContext();
  auto i32Ty = builder.getI32Type();

  builder.setInsertionPointToEnd(module.getBody());
  auto mainFn = LLVM::LLVMFuncOp::create(
      builder, loc, "main", LLVM::LLVMFunctionType::get(i32Ty, {}));
  Block *block = mainFn.addEntryBlock(builder);
  builder.setInsertionPointToStart(block);

  LLVM::CallOp::create(builder, loc, TypeRange{},
                       FlatSymbolRefAttr::get(ctx, "skeleton_program_init"),
                       ValueRange{});

  Value exitCode;
  auto entry = module.lookupSymbol<LLVM::LLVMFuncOp>(entryName);
  Type entryResultTy = entry ? entry.getFunctionType().getReturnType()
                             : LLVM::LLVMVoidType::get(ctx);
  if (isa<LLVM::LLVMVoidType>(entryResultTy)) {
    LLVM::CallOp::create(builder, loc, TypeRange{},
                         FlatSymbolRefAttr::get(ctx, entryName), ValueRange{});
  } else {
    Value result = LLVM::CallOp::create(builder, loc, TypeRange{entryResultTy},
                                        FlatSymbolRefAttr::get(ctx, entryName),
                                        ValueRange{})
                       .getResult();
    if (result.getType() == i32Ty)
      exitCode = result;
    else if (isa<IntegerType>(result.getType()))
      exitCode = LLVM::TruncOp::create(builder, loc, i32Ty, result);
  }
  if (!exitCode)
    exitCode = LLVM::ConstantOp::create(builder, loc, i32Ty,
                                        builder.getI32IntegerAttr(0));
  LLVM::ReturnOp::create(builder, loc, ValueRange{exitCode});
}

/// Embed the skeleton IR in its own section so the driver can re-read it from
/// the artifact.
void addSkeletonIR(OpBuilder &builder, ModuleOp module, StringRef ir,
                   Location loc) {
  MLIRContext *ctx = builder.getContext();
  auto i8Ty = builder.getI8Type();
  auto arrayTy = LLVM::LLVMArrayType::get(i8Ty, ir.size());

  SmallVector<NamedAttribute> attrs{NamedAttribute(
      StringAttr::get(ctx, "section"), StringAttr::get(ctx, ".skeleton.ir"))};

  builder.setInsertionPointToEnd(module.getBody());
  // External linkage keeps the section alive through the optimization
  // pipeline; an internal, unreferenced global would be removed by GlobalDCE.
  LLVM::GlobalOp::create(builder, loc, arrayTy, /*isConstant=*/true,
                         LLVM::Linkage::External, "skeleton.ir",
                         builder.getStringAttr(ir), /*alignment=*/1,
                         /*addrSpace=*/0, /*dsoLocal=*/false,
                         LLVM::ThreadLocalMode::NotThreadLocal,
                         /*comdat=*/SymbolRefAttr{}, attrs);
}

/// Emit `module` as an object file and link it with the runtime archive.
LogicalResult writeAndLink(ModuleOp module, StringRef output,
                           StringRef runtimeArchive, StringRef linker) {
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "failed to translate the artifact to LLVM IR\n";
    return failure();
  }

  std::string triple = llvm::sys::getDefaultTargetTriple();
  llvm::Triple targetTriple(triple);
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, error);
  if (!target) {
    llvm::errs() << "no target for " << triple << ": " << error << "\n";
    return failure();
  }

  llvm::TargetOptions targetOptions;
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetTriple, /*CPU=*/"generic", /*Features=*/"", targetOptions,
      std::nullopt));
  if (!machine) {
    llvm::errs() << "cannot create a target machine\n";
    return failure();
  }
  llvmModule->setDataLayout(machine->createDataLayout());
  llvmModule->setTargetTriple(targetTriple);

  // The JIT host compiles at the default optimization level; run the same
  // standard pipeline here so an artifact's AOT code is not left unoptimized.
  {
    llvm::LoopAnalysisManager loopAnalyses;
    llvm::FunctionAnalysisManager functionAnalyses;
    llvm::CGSCCAnalysisManager cgsccAnalyses;
    llvm::ModuleAnalysisManager moduleAnalyses;
    llvm::PassBuilder passBuilder(machine.get());
    passBuilder.registerModuleAnalyses(moduleAnalyses);
    passBuilder.registerCGSCCAnalyses(cgsccAnalyses);
    passBuilder.registerFunctionAnalyses(functionAnalyses);
    passBuilder.registerLoopAnalyses(loopAnalyses);
    passBuilder.crossRegisterProxies(loopAnalyses, functionAnalyses,
                                     cgsccAnalyses, moduleAnalyses);
    llvm::ModulePassManager modulePipeline =
        passBuilder.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2);
    modulePipeline.run(*llvmModule, moduleAnalyses);
  }

  llvm::SmallVector<char, 0> object;
  llvm::raw_svector_ostream objectStream(object);
  llvm::legacy::PassManager passManager;
  if (machine->addPassesToEmitFile(passManager, objectStream, nullptr,
                                   llvm::CodeGenFileType::ObjectFile)) {
    llvm::errs() << "the target cannot emit object files\n";
    return failure();
  }
  passManager.run(*llvmModule);

  llvm::SmallString<128> objectPath;
  int fd = -1;
  if (std::error_code ec = llvm::sys::fs::createTemporaryFile(
          "skeleton-artifact", "o", fd, objectPath)) {
    llvm::errs() << "cannot create a temporary object file: " << ec.message()
                 << "\n";
    return failure();
  }
  {
    llvm::raw_fd_ostream file(fd, /*shouldClose=*/true);
    file.write(object.data(), object.size());
  }

  llvm::ErrorOr<std::string> program = llvm::sys::findProgramByName(linker);
  if (!program) {
    llvm::errs() << "cannot find linker '" << linker
                 << "': " << program.getError().message() << "\n";
    llvm::sys::fs::remove(objectPath);
    return failure();
  }
  SmallVector<StringRef, 8> args{*program, objectPath, runtimeArchive,
                                 "-o",     output,     "-no-pie",
                                 "-ldl",   "-lstdc++", "-lm"};
  for (const std::string &extra : linkArgs)
    args.push_back(extra);
  int status = llvm::sys::ExecuteAndWait(*program, args);
  llvm::sys::fs::remove(objectPath);
  if (status != 0) {
    llvm::errs() << "linking the artifact failed (status " << status << ")\n";
    return failure();
  }
  return success();
}

} // namespace

/// Emit a standalone artifact: CPU implementations, the embedded skeleton IR
/// and the runtime. The artifact has no GPU code and no compiler, so running
/// it directly is CPU-only.
LogicalResult emitArtifact(ModuleOp module, DriverState &state,
                           StringRef output) {
  MLIRContext *context = state.context;

  // Fail here rather than inside the linker, where a wrong path shows up as a
  // missing symbol.
  if (!llvm::sys::fs::exists(state.runtimeArchive)) {
    llvm::errs() << "--runtime-lib '" << state.runtimeArchive
                 << "' does not exist\n";
    return failure();
  }

  {
    PassManager pm(context);
    pm.addPass(skeleton::createSkeletonOutlinePass());
    if (failed(pm.run(module)))
      return failure();
  }

  SmallVector<UnitMeta> units;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (!func->hasAttr("skeleton.op_id"))
      continue;
    auto idAttr = func->getAttrOfType<IntegerAttr>("skeleton.op_id");
    if (!idAttr) {
      llvm::errs() << "op unit '" << func.getSymName()
                   << "' has no integer skeleton.op_id\n";
      return failure();
    }
    UnitMeta unit;
    unit.id = idAttr.getInt();
    unit.preference = deviceFromPreference(func);
    if (auto bytes = func->getAttrOfType<IntegerAttr>("skeleton.bytes"))
      unit.bytes = bytes.getInt();
    unit.symbol = func.getSymName().str();
    units.push_back(std::move(unit));
  }

  std::optional<std::string> skeletonIR = captureSkeletonIR(module);
  if (!skeletonIR)
    return failure();

  Operation *entry = module.lookupSymbol(entryPoint);
  if (!entry) {
    llvm::errs() << "entry function '" << entryPoint << "' not found\n";
    return failure();
  }
  if (auto funcEntry = dyn_cast<func::FuncOp>(entry)) {
    if (funcEntry.getNumArguments() != 0 || funcEntry.getNumResults() != 0) {
      llvm::errs() << "artifact entry must take no arguments and return void\n";
      return failure();
    }
  } else if (auto llvmEntry = dyn_cast<LLVM::LLVMFuncOp>(entry)) {
    if (llvmEntry.getFunctionType().getNumParams() != 0) {
      llvm::errs() << "artifact entry must take no arguments\n";
      return failure();
    }
  } else {
    llvm::errs() << "artifact entry '" << entryPoint
                 << "' is neither a func.func nor an llvm.func\n";
    return failure();
  }
  if (SymbolTable::lookupSymbolIn(module, "__skeleton_entry")) {
    llvm::errs() << "artifact already has a '__skeleton_entry' symbol\n";
    return failure();
  }
  if (failed(SymbolTable::replaceAllSymbolUses(
          entry, StringAttr::get(context, "__skeleton_entry"), module))) {
    llvm::errs() << "cannot rewrite uses of entry function '" << entryPoint
                 << "'\n";
    return failure();
  }
  SymbolTable::setSymbolName(entry, "__skeleton_entry");

  if (SymbolTable::lookupSymbolIn(module, "main")) {
    llvm::errs() << "artifact already has a 'main' symbol; choose a "
                    "different entry function\n";
    return failure();
  }

  if (failed(lowerToLLVM(module, context, /*isHost=*/true)))
    return failure();

  Location loc = module.getLoc();
  OpBuilder builder(context);
  addRuntimeDeclarations(builder, module, loc);
  if (failed(addProgramInit(builder, module, units, loc)))
    return failure();
  addMain(builder, module, "__skeleton_entry", loc);
  addSkeletonIR(builder, module, *skeletonIR, loc);

  return writeAndLink(module, output, state.runtimeArchive, state.linker);
}

// Main

/// Load either a skeleton-level MLIR file or an emitted artifact. An artifact
/// carries its skeleton IR in the .skeleton.ir section.
static OwningOpRef<ModuleOp> loadProgram(StringRef path, MLIRContext *context) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrErr) {
    llvm::errs() << "cannot read " << path << "\n";
    return {};
  }
  StringRef data = (*bufferOrErr)->getBuffer();
  if (!data.starts_with("\x7f"
                        "ELF"))
    return parseSourceFile<ModuleOp>(path, context);

  auto objectOrErr = llvm::object::ObjectFile::createObjectFile(path);
  if (!objectOrErr) {
    llvm::errs() << "cannot read the artifact: "
                 << llvm::toString(objectOrErr.takeError()) << "\n";
    return {};
  }
  llvm::object::ObjectFile *object = objectOrErr->getBinary();
  for (const llvm::object::SectionRef &section : object->sections()) {
    auto nameOrErr = section.getName();
    if (!nameOrErr || *nameOrErr != ".skeleton.ir")
      continue;
    auto contentsOrErr = section.getContents();
    if (!contentsOrErr) {
      llvm::errs() << "cannot read .skeleton.ir: "
                   << llvm::toString(contentsOrErr.takeError()) << "\n";
      return {};
    }
    return parseSourceString<ModuleOp>(*contentsOrErr, context);
  }
  llvm::errs() << "artifact has no .skeleton.ir section\n";
  return {};
}

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Skeleton runtime placement driver\n");

  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  registerAllToLLVMIRTranslations(registry);
  registerAllPasses();

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  OwningOpRef<ModuleOp> module = loadProgram(inputFilename, &context);
  if (!module) {
    llvm::errs() << "cannot parse " << inputFilename << "\n";
    return 1;
  }

  DriverState state;
  state.context = &context;
  for (const std::string &lib : sharedLibs)
    state.libs.push_back(lib);
  state.runtimeArchive = runtimeLib;
  state.linker = linkerPath;
  driverState = &state;

  if (!emitPath.empty()) {
    if (state.runtimeArchive.empty()) {
      llvm::errs() << "--emit requires --runtime-lib\n";
      return 1;
    }
    return failed(emitArtifact(*module, state, emitPath)) ? 1 : 0;
  }

  FailureOr<int> result = runDriver(*module, state);
  if (failed(result))
    return 1;
  return *result;
}
