# 1.1.1 目录结构详解

LLVM本体项目位于仓库的 `llvm/` 目录下。本节将详细解析该目录下的主要文件和文件夹的组织方式及其用途。

## 顶层目录结构

LLVM本体的顶层目录位于 [llvm/](/llvm/)，其主要内容如下：

```
llvm/
├── include/llvm/       # 公共头文件
├── lib/                # 库实现代码
├── tools/              # 独立工具
├── utils/              # 工具脚本和TableGen
├── cmake/              # CMake模块
├── docs/               # 文档
├── examples/           # 示例代码
├── test/               # 测试用例
├── unittests/          # 单元测试
├── benchmarks/         # 基准测试
├── projects/           # 旧的项目目录（已弃用）
├── runtimes/           # 运行时构建
├── bindings/           # 语言绑定
├── resources/          # 资源文件
├── CMakeLists.txt      # 主CMake构建文件
└── README.txt          # 项目说明
```

以下是对各个主要目录的详细说明：

## include/llvm/ - 公共头文件

[llvm/include/llvm/](/llvm/include/llvm/) 目录包含了LLVM的所有公共头文件。这些头文件定义了LLVM的API接口，供外部代码和LLVM内部其他模块使用。

该目录按功能模块组织成多个子目录：

| 子目录 | 主要功能 | 代表性文件 |
|-------|---------|-----------|
| [ADT/](/llvm/include/llvm/ADT/) | 代数数据类型（Algebraic Data Types）- LLVM自定义的容器和数据结构 | [ArrayRef.h](/llvm/include/llvm/ADT/ArrayRef.h), [DenseMap.h](/llvm/include/llvm/ADT/DenseMap.h), [SmallVector.h](/llvm/include/llvm/ADT/SmallVector.h) |
| [IR/](/llvm/include/llvm/IR/) | 中间表示核心 - LLVM IR的核心类定义 | [Module.h](/llvm/include/llvm/IR/Module.h), [Function.h](/llvm/include/llvm/IR/Function.h), [Instruction.h](/llvm/include/llvm/IR/Instruction.h), [Type.h](/llvm/include/llvm/IR/Type.h) |
| [Analysis/](/llvm/include/llvm/Analysis/) | 分析框架 - 各种程序分析Pass | [LoopInfo.h](/llvm/include/llvm/Analysis/LoopInfo.h), [Dominators.h](/llvm/include/llvm/Analysis/Dominators.h) |
| [CodeGen/](/llvm/include/llvm/CodeGen/) | 代码生成基础设施 - 从IR到机器码的转换 | [MachineFunction.h](/llvm/include/llvm/CodeGen/MachineFunction.h), [SelectionDAG.h](/llvm/include/llvm/CodeGen/SelectionDAG.h) |
| [MC/](/llvm/include/llvm/MC/) | 机器码层 - 汇编器、反汇编器支持 | [MCInst.h](/llvm/include/llvm/MC/MCInst.h), [MCStreamer.h](/llvm/include/llvm/MC/MCStreamer.h) |
| [Support/](/llvm/include/llvm/Support/) | 支持库 - 平台抽象、工具函数 | [raw_ostream.h](/llvm/include/llvm/Support/raw_ostream.h), [ErrorHandling.h](/llvm/include/llvm/Support/ErrorHandling.h) |
| [Target/](/llvm/include/llvm/Target/) | 目标架构抽象 - 目标无关的后端接口 | [TargetMachine.h](/llvm/include/llvm/Target/TargetMachine.h) |
| [Transforms/](/llvm/include/llvm/Transforms/) | 优化变换 - IR优化Pass | [PassManager.h](/llvm/include/llvm/Transforms/PassManager.h) |
| [TableGen/](/llvm/include/llvm/TableGen/) | TableGen核心 - TableGen解析器和记录 | [Record.h](/llvm/include/llvm/TableGen/Record.h) |

## lib/ - 库实现代码

[llvm/lib/](/llvm/lib/) 目录包含了LLVM所有库的实现代码。该目录的组织结构与 `include/llvm/` 基本对应，每个头文件目录都有一个对应的实现目录。

主要子目录包括：

- `lib/ADT/` - ADT库的实现
- `lib/IR/` - IR核心的实现
- `lib/Analysis/` - 分析Pass的实现
- `lib/CodeGen/` - 代码生成的实现
- `lib/MC/` - 机器码层的实现
- `lib/Support/` - 支持库的实现
- `lib/Target/` - 各目标架构的后端实现
  - `lib/Target/X86/` - X86后端
  - `lib/Target/ARM/` - ARM后端
  - `lib/Target/AArch64/` - AArch64后端
  - `lib/Target/RISCV/` - RISC-V后端
  - 等等...
- `lib/Transforms/` - 优化变换的实现
  - `lib/Transforms/Scalar/` - 标量优化
  - `lib/Transforms/IPO/` - 过程间优化
  - `lib/Transforms/InstCombine/` - 指令组合
  - 等等...
- `lib/TableGen/` - TableGen的实现
- `lib/Bitcode/` - 位码读写
- `lib/BinaryFormat/` - 二进制格式支持
- `lib/DebugInfo/` - 调试信息支持
- `lib/ExecutionEngine/` - 执行引擎
- `lib/Linker/` - 链接器
- `lib/LTO/` - 链接时优化
- `lib/Object/` - 对象文件处理
- `lib/ProfileData/` - 性能分析数据
- `lib/Remarks/` - 优化remark支持
- `lib/TextAPI/` - 文本API支持
- `lib/WindowsManifest/` - Windows清单支持
- `lib/XRay/` - XRay跟踪支持

每个目标后端目录（如 `lib/Target/X86/`）通常包含：
- 目标描述的TableGen文件（`.td`）
- 指令选择、寄存器分配、调度等的实现
- 汇编器、反汇编器支持

## tools/ - 独立工具

[llvm/tools/](/llvm/tools/) 目录包含了多个独立的可执行工具，这些工具是LLVM功能的命令行接口。主要工具包括：

| 工具 | 功能 | 说明 |
|-----|------|------|
| [llc](/llvm/tools/llc/) | LLVM静态编译器 | 将LLVM IR编译为目标机器码或汇编 |
| [opt](/llvm/tools/opt/) | LLVM优化器 | 对LLVM IR运行优化Pass |
| [llvm-dis](/llvm/tools/llvm-dis/) | LLVM反汇编器 | 将位码反汇编为人类可读的IR |
| [llvm-as](/llvm/tools/llvm-as/) | LLVM汇编器 | 将人类可读的IR汇编为位码 |
| [llvm-link](/llvm/tools/llvm-link/) | LLVM链接器 | 链接多个LLVM模块 |
| [lli](/llvm/tools/lli/) | LLVM解释器 | 直接执行LLVM IR |
| [llvm-mc](/llvm/tools/llvm-mc/) | 机器码工具 | 汇编、反汇编机器码 |
| [llvm-nm](/llvm/tools/llvm-nm/) | 符号表工具 | 列出目标文件中的符号 |
| [llvm-objdump](/llvm/tools/llvm-objdump/) | 目标文件反汇编 | 反汇编目标文件 |
| [llvm-readobj](/llvm/tools/llvm-readobj/) | 目标文件信息 | 显示目标文件的详细信息 |
| [llvm-dwarfdump](/llvm/tools/llvm-dwarfdump/) | DWARF调试信息 | 转储DWARF调试信息 |
| [clang](/clang/) | C/C++/Objective-C编译器 | 独立的项目，不在llvm/tools/中 |

每个工具通常有自己的子目录，包含该工具的源代码和CMakeLists.txt。

## utils/ - 工具脚本和TableGen

[llvm/utils/](/llvm/utils/) 目录包含各种实用脚本和工具，其中最重要的是TableGen相关的代码。

主要子目录：

- [utils/TableGen/](/llvm/utils/TableGen/) - TableGen工具实现
  - [llvm-tblgen.cpp](/llvm/utils/TableGen/llvm-tblgen.cpp) - TableGen主程序入口
  - 各种Emitter实现（如 `RegisterInfoEmitter.cpp`, `InstrInfoEmitter.cpp` 等）
- `utils/bugpoint/` - Bugpoint工具（自动测试用例缩减）
- `utils/lit/` - LLVM集成测试工具
- `utils/FileCheck/` - FileCheck测试工具
- `utils/not/` - NOT工具（用于测试）
- `utils/count/` - COUNT工具（用于测试）
- `utils/PerfectShuffle/` - 完美洗牌生成
- `utils/Release/` - 发布脚本
- `utils/benchmark/` - 基准测试工具
- `utils/git/` - Git相关脚本
- `utils/codegen-diff/` - 代码生成差异比较
- `utils/llvm-build/` - LLVM构建工具
- `utils/convertyaml/` - YAML转换工具
- `utils/abtest/` - A/B测试工具

TableGen是LLVM中非常重要的工具，它使用特定的领域特定语言（DSL）来描述目标架构、指令集、寄存器等信息，然后生成相应的C++代码。详见[1.4节 TableGen详解](../section-4/README.md)。

## cmake/ - CMake模块

[llvm/cmake/](/llvm/cmake/) 目录包含了LLVM构建系统所需的CMake模块文件。这些模块定义了LLVM特有的CMake函数、宏和变量。

主要模块文件位于 `cmake/modules/` 目录：

| 模块 | 功能 | 详见 |
|-----|------|------|
| [AddLLVM.cmake](/llvm/cmake/modules/AddLLVM.cmake) | 添加LLVM库和可执行文件 | [1.3节](../section-3/README.md) |
| [LLVM-Build.cmake](/llvm/cmake/modules/LLVM-Build.cmake) | LLVM构建配置 | [1.3节](../section-3/README.md) |
| [TableGen.cmake](/llvm/cmake/modules/TableGen.cmake) | TableGen代码生成支持 | [1.3节](../section-3/README.md) |
| [HandleLLVMOptions.cmake](/llvm/cmake/modules/HandleLLVMOptions.cmake) | 编译选项处理 | [1.3节](../section-3/README.md) |
| [LLVM-Config.cmake](/llvm/cmake/modules/LLVM-Config.cmake) | LLVM配置工具 | [1.3节](../section-3/README.md) |

详见[1.3节 CMake构建系统详解](../section-3/README.md)。

## test/ - 测试用例

[llvm/test/](/llvm/test/) 目录包含了LLVM的回归测试套件。这些测试使用LLVM的lit测试工具运行，通常配合FileCheck使用。

测试按功能组织：

- `test/Analysis/` - 分析Pass测试
- `test/Assembler/` - 汇编器测试
- `test/Bitcode/` - 位码读写测试
- `test/CodeGen/` - 代码生成测试
  - `test/CodeGen/X86/` - X86代码生成测试
  - `test/CodeGen/ARM/` - ARM代码生成测试
  - 等等...
- `test/ExecutionEngine/` - 执行引擎测试
- `test/Feature/` - 特性测试
- `test/Instrumentation/` - 插装测试
- `test/Linker/` - 链接器测试
- `test/MC/` - 机器码层测试
- `test/Other/` - 其他测试
- `test/Support/` - 支持库测试
- `test/TableGen/` - TableGen测试
- `test/Transforms/` - 变换Pass测试
  - `test/Transforms/Scalar/` - 标量优化测试
  - `test/Transforms/IPO/` - 过程间优化测试
  - 等等...
- `test/Verifier/` - IR验证器测试
- `test/tools/` - 工具测试

## unittests/ - 单元测试

[llvm/unittests/](/llvm/unittests/) 目录包含了LLVM的单元测试，使用Google Test框架编写。这些测试针对LLVM的各个组件进行更细粒度的测试。

主要单元测试目录：

- `unittests/ADT/` - ADT库单元测试
- `unittests/IR/` - IR单元测试
- `unittests/Support/` - 支持库单元测试
- `unittests/CodeGen/` - 代码生成单元测试
- `unittests/MC/` - 机器码层单元测试
- `unittests/Transforms/` - 变换单元测试
- `unittests/TableGen/` - TableGen单元测试
- `unittests/ProfileData/` - 性能分析数据单元测试
- `unittests/DebugInfo/` - 调试信息单元测试
- 等等...

## docs/ - 文档

[llvm/docs/](/llvm/docs/) 目录包含了LLVM的官方文档源文件，主要是reStructuredText格式，使用Sphinx构建。

## examples/ - 示例代码

[llvm/examples/](/llvm/examples/) 目录包含了一些示例代码，展示如何使用LLVM的API：

- `examples/HowToUseJIT/` - 如何使用LLVM JIT
- `examples/ModuleMaker/` - 如何创建LLVM模块
- `examples/Kaleidoscope/` - Kaleidoscope语言教程（分章实现）
  - `examples/Kaleidoscope/Chapter2/` - 第二章：实现解析器和AST
  - `examples/Kaleidoscope/Chapter3/` - 第三章：代码生成
  - 等等...

## bindings/ - 语言绑定

[llvm/bindings/](/llvm/bindings/) 目录包含了LLVM的其他语言绑定，主要是：

- `bindings/go/` - Go语言绑定
- `bindings/ocaml/` - OCaml语言绑定
- `bindings/python/` - Python语言绑定（已弃用，推荐使用llvmlite）

## 目录组织原则

LLVM的目录组织遵循以下几个重要原则：

1. **头文件与实现分离** - 公共头文件在 `include/llvm/`，实现在 `lib/` 对应目录
2. **按功能模块组织** - 每个主要功能模块有独立的目录
3. **目标后端独立** - 每个目标架构在 `lib/Target/` 下有独立目录
4. **工具与库分离** - 库代码在 `lib/`，工具在 `tools/`
5. **测试独立** - 测试代码与生产代码分离

理解这些目录结构原则有助于在修改LLVM时找到正确的位置添加或修改代码。
