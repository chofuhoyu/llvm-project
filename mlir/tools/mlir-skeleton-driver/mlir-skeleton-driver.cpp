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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
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

static llvm::cl::opt<std::string> emitPath(
    "emit", llvm::cl::desc("Emit a standalone program artifact at this path"),
    llvm::cl::init(""));

static llvm::cl::opt<std::string> runtimeLib(
    "runtime-lib",
    llvm::cl::desc("Static runtime archive linked into an emitted artifact"),
    llvm::cl::init(""));

static llvm::cl::list<std::string> linkArgs(
    "link-arg",
    llvm::cl::desc("Extra argument passed to the linker when emitting"),
    llvm::cl::ZeroOrMore);

static llvm::cl::opt<std::string> linkerPath(
    "linker", llvm::cl::desc("Linker driver used to produce an artifact"),
    llvm::cl::init("clang"));

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
  void *entryArgs[] = {&empty};
  if (llvm::Error error =
          state.hostEngine->invokePacked(entryPoint, entryArgs)) {
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
// Artifact emission
//===----------------------------------------------------------------------===//

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
  module.print(stream);
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
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto voidTy = LLVM::LLVMVoidType::get(ctx);

  builder.setInsertionPointToEnd(module.getBody());
  auto init = LLVM::LLVMFuncOp::create(
      builder, loc, "skeleton_program_init",
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
    Value cpu = LLVM::ConstantOp::create(builder, loc, i32Ty,
                                         builder.getI32IntegerAttr(
                                             SKELETON_DEVICE_CPU));
    Value address = LLVM::AddressOfOp::create(builder, loc, impl);
    LLVM::CallOp::create(builder, loc, TypeRange{},
                         FlatSymbolRefAttr::get(ctx, "skeleton_rt_register"),
                         ValueRange{id, cpu, address});
  }
  LLVM::ReturnOp::create(builder, loc, ValueRange{});
  return success();
}

/// Generate the C entry point: run the registration, then the program.
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
  LLVM::CallOp::create(builder, loc, TypeRange{},
                       FlatSymbolRefAttr::get(ctx, entryName), ValueRange{});
  Value zero = LLVM::ConstantOp::create(builder, loc, i32Ty,
                                        builder.getI32IntegerAttr(0));
  LLVM::ReturnOp::create(builder, loc, ValueRange{zero});
}

/// Embed the skeleton IR in its own section so the driver can re-read it from
/// the artifact.
void addSkeletonIR(OpBuilder &builder, ModuleOp module, StringRef ir,
                   Location loc) {
  MLIRContext *ctx = builder.getContext();
  auto i8Ty = builder.getI8Type();
  auto arrayTy = LLVM::LLVMArrayType::get(i8Ty, ir.size());

  SmallVector<NamedAttribute> attrs{
      NamedAttribute(StringAttr::get(ctx, "section"),
                     StringAttr::get(ctx, ".skeleton.ir"))};

  builder.setInsertionPointToEnd(module.getBody());
  LLVM::GlobalOp::create(builder, loc, arrayTy, /*isConstant=*/true,
                         LLVM::Linkage::Internal, "skeleton.ir",
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
  if (std::error_code ec =
          llvm::sys::fs::createTemporaryFile("skeleton-artifact", "o", fd,
                                             objectPath)) {
    llvm::errs() << "cannot create a temporary object file: " << ec.message()
                 << "\n";
    return failure();
  }
  {
    llvm::raw_fd_ostream file(fd, /*shouldClose=*/true);
    file.write(object.data(), object.size());
  }

  SmallVector<StringRef, 8> args{linker, objectPath, runtimeArchive, "-o",
                                 output, "-ldl", "-lstdc++", "-lm"};
  for (const std::string &extra : linkArgs)
    args.push_back(extra);
  std::string program = linker.str();
  int status = llvm::sys::ExecuteAndWait(program, args);
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
    UnitMeta unit;
    unit.id = func->getAttrOfType<IntegerAttr>("skeleton.op_id").getInt();
    unit.preference = deviceFromPreference(func);
    if (auto bytes = func->getAttrOfType<IntegerAttr>("skeleton.bytes"))
      unit.bytes = bytes.getInt();
    unit.symbol = func.getSymName().str();
    units.push_back(std::move(unit));
  }
  if (units.empty()) {
    llvm::errs() << "no operator units to emit\n";
    return failure();
  }

  std::optional<std::string> skeletonIR = captureSkeletonIR(module);
  if (!skeletonIR)
    return failure();

  auto entry = module.lookupSymbol<func::FuncOp>(entryPoint);
  if (!entry) {
    llvm::errs() << "entry function '" << entryPoint << "' not found\n";
    return failure();
  }
  FunctionType entryType = entry.getFunctionType();
  if (entryType.getNumInputs() != 0 || entryType.getNumResults() != 0) {
    llvm::errs() << "artifact entry must take no arguments and return void\n";
    return failure();
  }
  entry.setName("__skeleton_entry");

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

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

/// Load either a skeleton-level MLIR file or an emitted artifact. An artifact
/// carries its skeleton IR in the .skeleton.ir section.
static OwningOpRef<ModuleOp> loadProgram(StringRef path, MLIRContext *context) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrErr) {
    llvm::errs() << "cannot read " << path << "\n";
    return {};
  }
  StringRef data = (*bufferOrErr)->getBuffer();
  if (!data.starts_with("\x7f" "ELF"))
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
