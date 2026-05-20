# Skeleton 方言示例实现设计决策

## 1. 方言命名

- **Dialect name**: `skeleton`
- **C++ namespace**: `mlir::skeleton`
- 方言名不带 "Ops" 后缀，符合 MLIR 命名惯例

## 2. 目录结构

```
mlir/include/mlir/Dialect/Skeleton/IR/        - TableGen 定义 + C++ 头文件
mlir/include/mlir/Dialect/Skeleton/Transforms/ - Pass 定义
mlir/lib/Dialect/Skeleton/IR/                 - C++ 实现
mlir/lib/Dialect/Skeleton/Transforms/         - Pass 实现
mlir/test/Dialect/Skeleton/                   - 测试
```

遵循 CreatingADialect.md 的标准布局。include 和 lib 两边都需要 CMakeLists.txt 接入。

## 3. 示例操作：skeleton.custom_matmul

- 三个输入操作数：lhs (M×K)、rhs (K×N)、output (M×N)
- 一个输出结果（与 output 同类型）
- 可选的 `preference` 属性（`#skeleton.preference<"GPU">`）
- 包含形状验证器，检查 M/K/N 维度兼容性
- custom assembly format：`ins(...) outs(...) -> type`
- 使用 `AnyRankedTensor` 而非 `MemRef` 作为操作数类型，保持函数式风格

## 4. 示例属性：skeleton.preference

- 参数类型选择 `std::string` 而非 `StringRef`：因为 MLIR 的 `FieldParser` 没有 `StringRef` 特化，使用 `std::string` 避免了编译错误
- 使用 `assemblyFormat = "<" $value ">"` 自动生成 parse/print
- 方言前缀：`skeleton`（满足 discardable attribute 要求）
- 打印格式：`#skeleton.preference<GPU>`（注意 std::string 打印时不带引号，但解析时支持 `"GPU"` 的形式）

## 5. 示例 Pass：skeleton-to-linalg

- Pass 名称：`skeleton-to-linalg`
- 功能：将 `skeleton.custom_matmul` 无条件转换为 `linalg.matmul`
- 如果 skeleton.custom_matmul 携带 preference 属性，作为 discardable attribute (`skeleton.preference`) 保留在 linalg.matmul 上
- 使用 `OpRewritePattern` + `applyPatternsGreedily` 而非 `applyPartialConversion`，简化逻辑
- 依赖：MLIRSkeleton + MLIRLinalgDialect + MLIRFuncDialect

## 6. CMake 集成

- 方言库：`add_mlir_dialect_library(MLIRSkeleton ...)` → 自动加入 `MLIR_DIALECT_LIBS`
- 变换库：`add_mlir_dialect_library(MLIRSkeletonTransforms ...)` → 同上
- 需要同时修改 `mlir/include/mlir/Dialect/CMakeLists.txt` 和 `mlir/lib/Dialect/CMakeLists.txt`
- 父 include CMakeLists.txt 确保 `add_mlir_dialect()` 的 TableGen 目标可被 lib 依赖
- Pass 注册：添加到 `RegisterAllPasses.cpp`，显式调用 `skeleton::registerSkeletonPasses()`
- 方言注册：添加到 `RegisterAllDialects.cpp`

## 7. 方言注册方式

选择修改 `RegisterAllDialects.cpp` 和 `RegisterAllPasses.cpp` 而非独立工具，因为：
- 使 `mlir-opt` 可以直接加载 Skeleton 方言
- 方便 lit 测试（利用现有 `mlir-opt` 基础设施）
- 符合 MLIR 方言集成惯例

## 8. TableGen 生成的文件命名

- `add_mlir_dialect(SkeletonOps skeleton)` 生成的 dialect 文件名为 `SkeletonOpsDialect.h.inc` 和 `SkeletonOpsDialect.cpp.inc`（而非 `SkeletonDialect.*.inc`）
- Op 实现文件 `SkeletonOps.cpp.inc` 有 GET_OP_LIST 和 GET_OP_CLASSES 两个区段，后者包含 print/parse/verify 实现

## 9. 构建体验

- 原始 `build/` 有预存的 Hexagon intrinsics 循环依赖问题，无法增量构建
- 新建 `build-skeleton/` 从头构建解决问题
- 需复制 `FileCheck`、`count`、`not` 等 lit 工具到新 build 目录

## 10. 测试

- 单文件 `skeleton-to-linalg.mlir`，使用 `-split-input-file` 分隔 3 个 test case
- 测试无 preference 转换、有 preference 转换（属性保留）、多 op 转换
- 使用 `llvm-lit` 运行，`PASS (100.00%)`
