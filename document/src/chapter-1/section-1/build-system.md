# 1.1.2 构建系统概览

LLVM使用CMake作为其构建系统。CMake是一个跨平台的构建生成工具，它本身不直接构建项目，而是生成适合目标平台的构建工具文件（如GNU Make、Ninja、Visual Studio解决方案等）。

本节将介绍LLVM构建系统的基本概念和推荐的构建流程。

## CMake版本要求

LLVM对CMake版本有最低要求。从[llvm/CMakeLists.txt:2](/llvm/CMakeLists.txt#L2)可以看到：

```cmake
cmake_minimum_required(VERSION 3.20.0)
```

这意味着构建LLVM至少需要CMake 3.20.0版本。

## 源外构建 (Out-of-source build)

LLVM**强烈推荐**使用源外构建（out-of-source build）方式。这意味着构建目录应该与源码目录分离。这种做法有以下优点：

1. **保持源码目录干净** - 构建产生的临时文件不会混入源码
2. **支持多种构建配置** - 可以同时维护Debug、Release等多种构建
3. **易于清理** - 只需删除构建目录即可清理所有构建产物
4. **避免版本控制干扰** - 构建文件不会被误提交到Git

## 基本构建流程

### 1. 创建构建目录

首先创建一个独立的构建目录，建议放在LLVM源码树之外：

```bash
# 在与llvm-project同级的目录创建build
mkdir build
cd build
```

也可以在llvm-project内部创建（但不在llvm/目录内）：

```bash
cd llvm-project
mkdir build
cd build
```

### 2. 运行CMake配置

使用CMake配置项目，指定源码路径和构建选项：

```bash
# 基本配置，使用默认选项
cmake -S ../llvm -B .

# 或者指定生成器（如Ninja）
cmake -G Ninja -S ../llvm -B .

# 带常用选项的配置示例
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_TARGETS_TO_BUILD="X86;RISCV" \
  -DCMAKE_INSTALL_PREFIX=/path/to/install \
  -S ../llvm \
  -B .
```

参数说明：
- `-G <generator>`: 指定生成器（如 "Ninja", "Unix Makefiles", "Visual Studio 17 2022" 等）
- `-S <source>`: 指定源码目录（包含顶层CMakeLists.txt的目录）
- `-B <build>`: 指定构建目录
- `-D<var>=<value>`: 设置CMake变量

### 3. 执行构建

CMake配置完成后，使用生成的构建工具进行编译：

```bash
# 使用CMake的跨平台构建命令（推荐）
cmake --build .

# 或者直接调用生成的工具
ninja          # 如果使用Ninja生成器
make -j$(nproc) # 如果使用Unix Makefiles生成器

# 并行编译加速
cmake --build . -j$(nproc)  # 使用所有CPU核心
cmake --build . -j8          # 指定使用8个并行任务
```

### 4. 安装（可选）

如果需要安装LLVM到指定目录：

```bash
cmake --build . --target install

# 或者直接调用
ninja install
make install
```

## C++标准要求

从[llvm/CMakeLists.txt:62](/llvm/CMakeLists.txt#L62)可以看到，LLVM要求使用C++17：

```cmake
set(LLVM_REQUIRED_CXX_STANDARD 17)
```

CMake会自动配置编译器使用C++17标准，用户也可以选择更新的标准（如C++20），但不能使用低于C++17的标准。

## 推荐的构建配置

以下是一些常用的构建配置示例：

### 开发构建（Debug模式）

```bash
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DLLVM_ENABLE_PROJECTS="clang" \
  -S ../llvm \
  -B .
```

特点：
- 无优化，便于调试
- 启用断言
- 只构建X86目标以加快编译
- 包含Clang

### 发布构建（Release模式）

```bash
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_TARGETS_TO_BUILD="all" \
  -DLLVM_ENABLE_PROJECTS="clang;lld;mlir" \
  -DCMAKE_INSTALL_PREFIX=/opt/llvm \
  -S ../llvm \
  -B .
```

特点：
- 最高优化级别
- 禁用断言
- 构建所有目标
- 包含多个子项目
- 指定安装路径

### 带调试信息的发布构建（RelWithDebInfo）

```bash
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;RISCV;ARM" \
  -S ../llvm \
  -B .
```

特点：
- 优化编译但保留调试信息
- 启用断言
- 适合性能分析和调试

## 构建类型说明

| 构建类型 | 优化级别 | 调试信息 | 断言 | 适用场景 |
|---------|---------|---------|------|---------|
| Debug | 无 | 有 | 默认启用 | 开发、调试 |
| Release | 最高 | 无 | 默认禁用 | 生产使用 |
| RelWithDebInfo | 较高 | 有 | 默认禁用 | 性能分析、调试Release问题 |
| MinSizeRel | 最小尺寸 | 无 | 默认禁用 | 空间受限环境 |

## 重要注意事项

1. **不要在源码目录内构建** - LLVM不支持源码内构建
2. **选择合适的生成器** - Ninja通常比Make更快，推荐使用
3. **并行编译** - 使用 `-j` 参数充分利用多核CPU
4. **目标选择** - 通过 `LLVM_TARGETS_TO_BUILD` 只构建需要的目标可显著加快编译
5. **内存需求** - 完整构建LLVM可能需要大量内存（建议16GB以上）

## 增量构建

修改代码后，只需重新运行构建命令即可：

```bash
cmake --build .
```

CMake会自动检测变化的文件并只重新编译必要的部分。如果修改了CMakeLists.txt或构建配置，可能需要重新运行CMake配置：

```bash
# 重新运行CMake配置（会保留之前的设置）
cmake .

# 或者完全重新配置
rm -rf *
cmake -G Ninja -S ../llvm -B .
```

## 常见问题

**Q: CMake配置时提示找不到编译器？**
A: 确保编译器在PATH中，或通过 `CC` 和 `CXX` 环境变量指定：
```bash
CC=clang CXX=clang++ cmake -G Ninja -S ../llvm -B .
```

**Q: 如何只构建特定的工具？**
A: 使用目标指定：
```bash
cmake --build . --target llc opt
```

**Q: 构建过程中内存不足怎么办？**
A: 减少并行任务数，或只构建部分目标：
```bash
cmake --build . -j2
```

关于CMake变量和宏的详细说明，请参见[1.3节 CMake构建系统详解](../section-3/README.md)。
