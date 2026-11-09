# 20261109：C++ 前端与算子边界（cpp-codegen-operators）

审查范围：`git log --oneline caelum/skeleton-dialect..caelum/cpp-codegen-operators`，共 33 个提交
（2026-11-05，从 `03a59fad4e49` 的 C++ 端到端分析到 `280621fe799d` 的三层架构总结）。
`git diff --stat` 是 73 个文件、+3835/−2905。删掉的行里有很大一部分来自半自动路径
（`CirLoopToSkeleton.cpp` 677 行、`CirCallToSkeleton.cpp` 224 行、三组测试）和四个两段式
e2e（`.driver` 拼接 + 手写 `main`），其余是 `CirCallLowering` / `CirFuncToArith` 的改写。

姊妹审查：[`20261109-finish-roadmap-approach.md`](20261109-finish-roadmap-approach.md) 审的是
这条分支之上的 49 个提交（`caelum/cpp-codegen-operators..caelum/finish-roadmap-approach`）。
两份文件的分界就是 `280621fe799d`：那里两条分支同一个树，之后才分叉。

## 合并视角：哪些已经在 finish 上修好了（先读）

这条分支是 `caelum/finish-roadmap-approach` 的祖先，finish 已经把它整体包含进去。所以“审完
把代码合进去”在 git 上只有一次合并（合 finish），本文的 16 条问题里有 9 条在 finish 上已经
不存在，剩下 7 条才是合并后要动的清单。每条标题带状态标签（`［已在 finish 修复］` /
`［仍在 finish 上］`），这里是汇总：

| # | 问题 | finish 上 | 由谁解决 |
|---|---|---|---|
| 代码 1 | reduce 的 destination 是只读视图 | 已修复 | `6fb433db2c1d`、`0a02ede7c566`、`361c7074a36b` |
| 代码 2 | thin wrapper 的 `cir.store` 被删 | 已修复 | `f50e3d9214ed` |
| 代码 3 | map 的标量输入在 C++ 边界不可达 | 已修复 | `2948881f9e59` |
| 代码 4 | driver 入口的 `cast<LLVMFuncOp>` | **仍在** | —— |
| 代码 5 | ABI 的 stride 对末尾动态维算错 | **仍在**（注释已补，`0b03bc2ac78f`） | —— |
| 代码 6 | `skeleton-lower-host-abi` 没有测试 | **仍在** | —— |
| 代码 7 | dispatch 静默跳过解析失败的单元名 | **仍在** | —— |
| 代码 8 | span 长度固定成 i64 | **仍在** | —— |
| 代码 9 | `skeleton++` 的 `exec`/trap 与 `--link-arg` | **仍在** | —— |
| 代码 10 | `createUnit` 里两个连续的 `if (Map)` | 已修复 | `0a02ede7c566` |
| 文档 1 | roadmap 仍是 v6 | 已修复（v9） | `f50e3d9214ed` |
| 文档 2 | `cpp_codegen.md` §2 没标成历史 | 已修复 | `8a85c9b10cff` |
| 文档 3 | decisions §1.3 说“标量按值传” | **仍在** | —— |
| 文档 4 | `SkeletonOps.h` 的标量广播说法 | 已修复 | `4edb0d1a6fc9` |
| 文档 5 | `CirCallLowering.h` 的 destination 注释 | 已修复 | `6fb433db2c1d` |
| 文档 6 | `operators.md` §5 的 XFAIL 理由 | 已修复 | `8a85c9b10cff` |

读法：标“已修复”的 9 条可以直接跳过（它们只解释现在代码为什么长这样）；标“仍在”的 7 条
是合并前后要处理的，逐条细节在 §4 与 §5。

一处过程记录：`f50e3d9214ed` 把 roadmap v9 的 191 行重写和 `CirSkeletonSplit` 的 15 行修改
放在同一个提交里，按“一条改动一个提交”本应分开。不影响内容，只影响将来 bisect 与回溯。

## 0. 方法与边界

读过的：

- 计划与记录：`caelum/plan/cpp_codegen.md`、`caelum/doc/cpp-codegen-operators-decisions.md`、
  `caelum/plan/operators.md`、`caelum/plan/roadmap.md`、`caelum/report/20261107.md`。
- 按区域读 diff 与最终文件：CIR 侧（`CirSkeletonSplit`、`CirCallLowering`、`CirCallAnalysis`、
  `CirFuncToArith`、`CirStripCIRAttrs`、`CirSkeletonAnnotations`、`CIRGenStmt` 的注解形参删除）、
  方言与 lowering（`SkeletonOps.h`、`SkeletonOps.cpp`、`SkeletonToLinalg`、`SkeletonLowerDispatch`、
  新增 `SkeletonLowerHostABI`）、driver（入口、混合模块、产物 `main`、退出码）、`skeleton++`
  与 lit 接入，以及 `mlir/test/Dialect/Skeleton/`、`mlir/test/Integration/Dialect/Skeleton/{CPU,GPU}`、
  `clang/test/CIR/Transforms/skeleton-split*.cpp` 的相关用例。
- `.profile/`（独立仓库）没有读：本轮没有需要它的判断。

没做、也做不到的：

- 本机不编译、不测试，也没有跑任何远端构建。所有“测试通过”都来自测试文件与文档本身，不是实测；
  GPU 行为只能在代码层面推理。
- 远端 build 服务器本轮没有访问：这条分支的验收证据全在树里，没有需要读远端日志才能判断的问题。
- 行号约定：§1–§5 正文里的 `path:line` 取自**这条分支自己的树**（等价于 `git show
  caelum/cpp-codegen-operators:<path>`）。标了“状态”的行在描述 finish 的现状时，行号取自
  finish 并在句中写明（如“finish 的 `CirSkeletonSplit.cpp:363`”）；不要把两类行号混着对
  工作区。

## 1. 改动摘要

一句话：C++ 从“算子描述语言”变成“程序语言”。`skeleton++ app.cpp` 一条命令把 `main`、控制流、
标准库调用和算子一起编出来；算子在函数级被切走，宿主保留原符号与 C++ 签名，边界是
destination-passing。骨架成立、负例齐、真实算子（softmax）与 benchmark 的入口都在。

但有两处需要先说在前面：归约单元的 destination 仍然是只读视图，所以“归约上不了设备”在这条
分支上是真的，而记录给的理由不是真的原因（问题 1）；thin-wrapper 的语句删除规则会静默丢掉
wrapper 里的写操作（问题 2）。

### 1.1 边界：整模块替换 → 函数级切分

`CirSkeletonSplit` 取代 `CirCallToSkeleton`。带 `skeleton.region` 的 host：

- 保留符号和 C++ 签名，函数体改写成对 `__skeleton_op_<id>_shim(...)` 的调用；
- 内部唯一的骨架调用变成一个 `func.func private @__skeleton_op_<id>(memref..., scalar...) -> ()`，
  带 `skeleton.op_id` / `skeleton.bytes` / `skeleton.preference`；
- thunk 是 `cir.func private` 声明，带 `skeleton.host_abi = {op_id, unit, operands}`，body 由
  `skeleton-lower-host-abi` 在 host 降到 LLVM 之后填。

`main`、普通函数、纯函数都留给 `cir-to-llvm`。v1 的硬限制（每个 host 恰好一次骨架调用、
必须是 thin wrapper、必须返回 void、纯函数不能被 host 调用）都在切分时报错。

### 1.2 边界 ABI

- 数据参数三种形式：`T (&)[N]` → 静态维、`skeleton_detail::span<T> &` → 一维动态、`T &` →
  单元素且只允许出现在 destination；裸 `T *` 拒绝。
- 形状不来自算子语义，直接来自 destination 参数的类型。
- 单元签名是 `(N 个 memref, 标量...) -> ()`，无返回值；结果通过
  `bufferization.materialize_in_destination ... writable` 写回目的地。
- `skeleton-lower-host-abi` 把 `(ptr, i64...)` 拼成 memref 描述符（allocated=aligned=ptr、
  offset=0、静态维是常量、动态维取长度、行主序 stride），并用单元 lower 后的
  `LLVMFunctionType` 逐参数校验，不一致就编译期报错。

### 1.3 driver 接受混合模块

- `hasTensorLevelOps`：host 已经是 LLVM 方言时跳过 `skeleton-to-linalg` + bufferize；
- 入口放宽到 `llvm.func`（C++ 的 `main` 是 `llvm.func @main() -> i32`）；
- `invokePacked` 为有返回值的入口留结果槽；产物里 `main` 返回入口的退出码；
- `available_externally` 且带 body 的定义改成 `linkonce_odr`（JIT 要自己解析隐式模板实例化）；
- 入口之前跑 `skeleton-lower-host-abi`，之后才是 `skeleton-lower-dispatch`。

### 1.4 一条命令

`mlir/tools/skeleton++/skeleton++.sh.in` 由 CMake `configure_file` 生成，串起
clang++ → cir-opt（`--cir-skeleton-split --cir-to-llvm=allow-mixed-module --cir-strip-cir-attrs`）
→ driver。clang driver 的 `-fskeleton` 没做（决策 §4.3 明确说等 API 稳定）。

### 1.5 纯函数词汇

`CirFuncToArith` 从 367 行扩到 705 行：alloca/load/store 槽位跟踪、整数/浮点四则、
`div`/`rem` 按符号选 op、比较、select（ternary 两臂都求值）、算术 cast、`cir.fmuladd` 展开、
`get_global` 当死值、以及 C 库 math 函数名 → `math.*`（`expf`/`log`/`sqrt` 等）。其余调用、
多 block、volatile/atomic 一律带诊断拒绝。

### 1.6 方言与 lowering

- `skeleton.map` 允许低秩输入按前导维广播（`SkeletonToLinalg` 的 `broadcastMap`）；
- `skeleton.reduce` 的 `init` 可以是 0..inputRank 秩，降的是 init 没有的尾部维；
  `ReduceOp` 校验 init 是 input 的前缀维；
- `SkeletonLowerDispatch` 跳过 `__skeleton_op_<id>_shim`（同一个前缀但不是单元）。

### 1.7 删除

半自动路径的 pass、注册、`CIRGenStmt.cpp` 的 `attachLoopAnnotations`、三组测试；
四个两段式 e2e 改成单 `.cpp`；`skeleton-peel-module.sh` 删除。

## 2. 重点文件与逻辑

| 文件 | 这一版做了什么 | 先看哪段 |
|---|---|---|
| `clang/lib/CIR/Dialect/Transforms/CirSkeletonSplit.cpp`（新，530 行） | 切分本身：检查、建单元、建 thunk、改写 host | `runOnOperation`（检查与工作列表）→ `createUnit`（map/reduce 的单元形状与写回）→ `rewriteHost`/`createThunk` |
| `clang/lib/CIR/Dialect/Transforms/CirCallLowering.{h,cpp}` | 形状与 ABI 的来源 | `hostParamToMemRef`（类型 → memref）、`resolveDataOperands`（谁是数据、谁是标量） |
| `clang/lib/CIR/Dialect/Transforms/CirCallAnalysis.cpp` | 反向解析调用实参、纯函数引用 | `resolveToHostArg`、`tracePureFnRef` |
| `clang/lib/CIR/Dialect/Transforms/CirFuncToArith.cpp` | 纯函数的标量词汇 | 文件头（支持范围）、`populateArithFuncBody` 的 op 分派、math 名字表 |
| `mlir/lib/Dialect/Skeleton/Transforms/SkeletonLowerHostABI.cpp`（新，227 行） | thunk body：描述符构造 + 签名校验 | `memrefDescriptorTypes`、`lowerThunk` 的 stride 计算与校验 |
| `mlir/lib/Dialect/Skeleton/Transforms/SkeletonToLinalg.cpp` | map 广播、reduce 的维 | `broadcastMap`、`ConvertReduceOp` 里从 init 秩算 `dimensions` |
| `mlir/lib/Dialect/Skeleton/IR/SkeletonOps.cpp` | 两个 op 的 verifier 放宽 | `MapOp::verify`、`ReduceOp::verify` |
| `mlir/include/mlir/Dialect/Skeleton/SkeletonOps.h` | 用户 API 与约定 | 文件头 40 行注释 + 三个 helper 声明 |
| `mlir/tools/mlir-skeleton-driver/mlir-skeleton-driver.cpp` | 混合模块、入口、产物 | `hasTensorLevelOps`/`lowerToLLVM`、`runDriver` 的入口检查、`emitArtifact`/`addMain` |
| `mlir/tools/skeleton++/skeleton++.sh.in` | 三段流水线 | 参数解析、`cir-opt` 那一行、`--run` 与产物两条尾巴 |
| `clang/lib/CIR/Lowering/DirectToLLVM/LowerToLLVM.cpp` | `allow-mixed-module` | option 定义与 `populateCIRToLLVMPasses` |
| 测试 | 正/负例与四条 e2e | `clang/test/CIR/Transforms/skeleton-split.cpp`（断言单元与 thunk 形状）、`skeleton-split-invalid.cpp`（八类诊断）、`CPU/cpp-softmax.cpp`（真实 DAG）、`GPU/cpp-*.mlir` |

## 3. 阅读顺序

按这个顺序读，每一步都能回答一个具体问题；不需要先读 diff。

1. `mlir/include/mlir/Dialect/Skeleton/SkeletonOps.h`（文件头注释 + 三个 helper 声明）。
   回答：用户在 C++ 里怎么写一个算子边界？形状从哪来？谁是 destination？
2. `caelum/doc/cpp-codegen-operators-decisions.md` §1–§2。
   回答：为什么一律按引用传、为什么 destination 在前且无返回值、为什么要多一层 thunk、
   元数据挂在 thunk 上的原因。这一节是后面所有代码的“为什么”。
3. `CirSkeletonSplit.cpp`：先 `runOnOperation`（哪些情况报错、worklist 怎么建），再 `createUnit`
   （单元的签名与写回），最后 `rewriteHost` / `createThunk`（host 侧怎么取指针和长度）。
   回答：一个 host 怎么变成“host stub + 单元 + thunk”？
4. `CirCallLowering.cpp`：`hostParamToMemRef` 与 `resolveDataOperands`。
   回答：`float (&)[2][4]`、`span<T> &`、`T &`、`T *` 各自变成什么？标量是怎么被认出来的？
5. `SkeletonLowerHostABI.cpp`：`memrefDescriptorTypes`、`lowerThunk` 的 stride 与校验。
   回答：C++ 的指针+长度怎么变成 `convert-func-to-llvm` 眼里的 memref 描述符？
6. driver：`lowerToLLVM` → `runDriver` → `emitArtifact`/`addMain`。
   回答：为什么 C++ 的 `main` 能直接当入口？产物为什么能把退出码带出来？
7. `skeleton++`：三段命令，以及 `--emit-mlir` 停在哪。
   回答：一条命令背后的真实管线；想手工复现时该跑什么。
8. 测试：`skeleton-split.cpp`（正例，`CHECK` 就是 ABI 的规格）→ `skeleton-split-invalid.cpp`
   （负例）→ `CPU/cpp-softmax.cpp`（四个单元的真实 DAG）→ `GPU/cpp-*.mlir`（放置）。
   回答：哪些形状被钉住了、哪些只是“跑出数值”。

## 4. 问题

### 1. ［高］［已在 finish 修复］归约单元的 destination 是只读视图，GPU 上的失败是它造成的

**状态**：`caelum/finish-roadmap-approach` 上已修复——`createToTensor` 增加 `writable` 参数，
reduce（`6fb433db2c1d`）与 map/matmul 的 destination 都改成可写视图，`GPU/cpp-softmax` 的
`XFAIL` 已删除。本节保留是因为它是那个 `XFAIL` 的真实原因、也是 D2 的入口；合 finish 时不需要
动作。

`CirSkeletonSplit.cpp:299` 给 reduce 的 destination 用的是和输入同一个 helper：

```
Value init = createToTensor(body, loc, dstMemref);
```

而 `createToTensor`（`CirCallLowering.cpp`，唯一实现）固定 `restrict=true, writable=false`。
`CirCallLowering.h:96-103` 的注释恰好写了“destination 不走这条路，它由
`materialize_in_destination` 写回”——代码违反了这条注释。

后果：one-shot bufferization 认为这个 tensor 不可写，于是给 fill 和 reduce 分配临时缓冲、
最后再拷回 destination。CPU 上只是白做功；设备上那份临时缓冲是宿主内存，kernel 访问它就是
非法地址。这正是 `GPU/cpp-softmax.mlir` 的 `XFAIL` 在树上记录的现象，但 XFAIL 与
`cpp-codegen-operators-decisions.md` §5.3 把原因写成“归约的组合器被当成独立迭代映射到线程”，
与代码里的形状不符。`clang/test/CIR/Transforms/skeleton-split.cpp` 的 reduce 段现在把这个错误
形状**钉住**了：`CHECK: bufferization.to_tensor {{.*}} restrict : memref<3xf32>`（没有
`writable`），而同一段又断言 `materialize_in_destination ... in writable`。

map 走的是另一条路（`tensor.empty` + `driver` 里的 `empty-tensor-elimination`，`:261-273`），
所以只有 reduce 中招。这是本分支最硬的一处“记录说已经达成、代码说没有”的缺口；修法是让
destination 用可写视图（`createToTensor(..., /*writable=*/true)`），并让 fill/reduce 直接吃
它的结果，而不是丢弃 fill 的返回值。姊妹审查的 D2 就是这个修法。

### 2. ［高］［已在 finish 修复］thin wrapper 放行所有 `cir.store`，改写时又按“没有结果”删掉

**状态**：已修复——`f50e3d9214ed` 把删除条件改成“有结果且无使用”（现在 `:222` 是
`if (op->getNumResults() && op->use_empty())`），store 因此留下来。合 finish 时不需要动作。

`CirSkeletonSplit.cpp:210`：

```
if (!op->getNumResults() || op->use_empty())
  op->erase();
```

任何没有结果的操作都会被删——包括 `cir.store`。而 `isThinWrapperOp` 的白名单里有
`cir::StoreOp`、`cir::GetGlobalOp`、`cir::GetMemberOp`，所以下面这种 wrapper 会**编过**：

```cpp
static int g_calls = 0;
extern "C" __attribute__((annotate("skeleton.region", "CPU")))
void f(const float (&a)[4], float (&out)[4]) {
  g_calls = 1;                       // get_global + store，白名单通过
  skeleton_map(my_add, out, a, a);
}
```

`g_calls = 1` 在 `rewriteHost` 里被静默丢掉，程序照跑，只是少了一句。对 `span` 的 `size`
成员赋值、对 `T &` 参数的赋值同样中招。这与项目“不支持就带定位报错、不要静默错编”的原则
直接冲突，而且报错文本（“this statement would be dropped”）说明作者本意是拒绝这类语句，
只是检查放宽到了“操作种类”而不是“这个操作是否只服务于这次调用的 staging”。

最小修法是只删有结果且无使用的操作（姊妹分支就是这么改的）；更严的做法是只允许 store 写进
staging alloca。

### 3. ［中］［已在 finish 修复］`skeleton_map` 的标量输入在 C++ 边界上不可达

**状态**：已修复——`2948881f9e59` 增加 `byReference` 标量（`CirCallLowering.cpp:133`），切分把
destination 之后指向标量的指针当标量值、并在调用前 load；`SkeletonOps.h` 的文件头也改成了这个
说法（文档问题 4 一并解决）。

`SkeletonOps.h:43-47`、`SkeletonOps.cpp` 的 map builder、`SkeletonToLinalg` 的 rank-0 广播都
按“map 可以吃标量输入”写（softmax 的注释也说“归约结果喂给下一步”），但边界上做不到：

- `skeleton_map` 的变参是 `const Ins &...`，所以调用点传进来的永远是**指针**；
- `resolveDataOperands`（`CirCallLowering.cpp:105/123`）只把“非指针”的操作数当标量；
- `hostParamToMemRef`（`:45`）对非 destination 的 `T &` 直接失败（`:75`）。

于是 `skeleton_map(fn, out, x, 1.0f)` 报 “argument N is not a direct parameter of the host
function”（字面量 staging 在 alloca 里），而把 `float k` 参数直接传进去报 “host parameter N of
type !cir.ptr<!cir.float>, which is not a data argument”。两条错误信息都没有说“标量输入不支持”，
读者会以为是自己写法的问题。
本分支的 softmax 是用 `float m[3]` 这种 rank-1 数组把归约结果喂给下一步的，所以没暴露。
姊妹分支的 `byReference` 标量处理就是补这一格；本文档承诺的能力在本分支上不可达，属于
“文档写了、单元 builder 写了、前端没写”。

### 4. ［中］［仍在 finish 上］driver 的入口检查与后面的 `cast<LLVM::LLVMFuncOp>` 不一致

**状态**：仍在——`caelum/finish-roadmap-approach` 的 `mlir-skeleton-driver.cpp:668` 还是无条件
`cast`。合并后要修。

`runDriver` 在 `:574-592` 明确接受两种入口，`func.func` 只检查“没有参数”，**允许有返回值**；
但在 `:638` 无条件：

```
if (auto entry = cast<LLVM::LLVMFuncOp>(module.lookupSymbol(entryPoint));
```

一个合法的输入 `func.func @main() -> i32`（手写 MLIR，甚至就是某个 C++ 程序在 `cir-to-llvm`
之前的形态被直接喂给 driver）会走到这里，`cast` 在 debug 构建里断言、release 里是 UB。
`emitArtifact`（`:953`）对同样输入是明确报错的（“must take no arguments and return void”），
两处行为不一致。改成 `dyn_cast` 后按类型决定要不要结果槽即可。

### 5. ［中］［仍在 finish 上（潜在）］host ABI 的“动态维必须是最后一维”检查没有让该形状变正确

**状态**：仍在——stride 的计算没有变；`0b03bc2ac78f` 只补了注释（说明这个形状前端产生不了、
这条检查是给手写 IR 准备的），但 `memref<3x?xf32>` 依旧会得到 `[1, 1]`。

`SkeletonLowerHostABI.cpp:160-164` 只拒绝“rank>1 且动态维不在最后”，`:177-183` 的 stride 却
只乘**静态**维：

```
strides[dim] = constant(stride);
if (!ShapedType::isDynamic(...)) stride *= type.getShape()[dim];
```

对 `memref<3x?xf32>` 会得到 `[1, 1]`，而正确值是 `[size(1), 1]`。前端只产生“全静态”或
“一维动态”，所以现状不可达；但诊断的措辞（“a dynamic dimension must be the last one”）
暗示这种形状是被支持的。要么把检查收紧成“rank>1 时不允许任何动态维”，要么用 sizes 算 stride。

### 6. ［中］［仍在 finish 上］新 pass `skeleton-lower-host-abi` 没有任何直接测试

**状态**：仍在——finish 改了这条诊断的措辞，但 `mlir/test/Dialect/Skeleton/` 下仍然没有
`skeleton-lower-host-abi*.mlir`。姊妹审查的问题 4 是同一件事。

`mlir/test/Dialect/Skeleton/` 下没有对应的 `.mlir` 用例（`skeleton-lower-dispatch.mlir` 是有的）。
这个 pass 的三条诊断——`skeleton.host_abi` 形状不对、单元签名与描述符不符、“动态维必须是
最后一维”——全部没有负例；唯一的覆盖是 `skeleton++` 的 C++ e2e，而 e2e 只证明“对的那条路能跑”。
ABI 是最容易在重构里悄悄坏掉的一层（它是 `convert-func-to-llvm` 内部约定的复制），
建议至少补三条 `expected-error` 和一条正例。

### 7. ［低］［仍在 finish 上］dispatch 对解析失败的单元名从报错改成静默跳过

**状态**：仍在（finish 的 `SkeletonLowerDispatch.cpp:76-77` 未变）。

`SkeletonLowerDispatch.cpp:76-77`：`getAsInteger` 失败就 `continue`。改动本身有理由（thunk 共享
前缀），但代价是任何拼错/漏改的 `__skeleton_op_...` 符号会留成直接调用，到 JIT 时才
“Symbols not found”。区分方式可以是“名字以 `_shim` 结尾就跳过，其余解析失败仍报错”。

### 8. ［低］［仍在 finish 上］span 的长度在 ABI 里固定成 i64，而它是 `unsigned long`

**状态**：仍在（finish 的 `CirSkeletonSplit.cpp:363` 未变）。

`createThunk` 的 `lengthTy` 硬编码 `i64`，`skeleton_detail::span::size` 是 `unsigned long`：
LP64 上一致，32 位目标上 `cir.call` 会因参数类型不符报错（响亮失败，不是静默错编，但这条
ABI 契约没有写出来）。可以改成从 size 成员的类型取宽度。

### 9. ［低］［仍在 finish 上］`skeleton++` 的两处小毛病

**状态**：仍在——finish 只加了 `--driver-arg=`，`exec` 与 `trap` 的冲突、`--link-arg` 只给产物
都没改。

- `:115` 的 `--run` 用 `exec` 替换 shell，`:85` 的 `trap 'rm -rf "$work"' EXIT` 因此不会执行，
  每跑一次泄漏一个 `/tmp/skeleton-*`；
- `--link-arg` 只传给产物路径，`--run` 那条不传（想加链接参数调试 driver 时做不到）。

### 10. ［低］［已在 finish 修复］`createUnit` 里两个连续的 `if (opType == Map)`

**状态**：已修复——随 `0a02ede7c566` 合并成一个 `if/else` 链（finish 上只剩一个
`if (info.opType == SkeletonOpType::Map)`）。

`:266` 的 `if` 只算 `dynSizes`/`tensor.empty`，`:276` 的第二个 `if` 才构造算子。行为没问题，
但读起来像两次改写的合并残留；把 `result` 的初始化并进同一个分支会更清楚。

## 5. 文档与记录一致性问题

1. **`caelum/plan/roadmap.md` 仍是 v6**，而这条分支删了半自动路径、加了整程序 C++。
   §3.1 还写“半自动路径仍在树上，是接下来要删的第一样东西”，§2.5 写 GPU“只支持静态形状”，
   §5 的待办第一条仍是“删除半自动路径”。W8 在 `cpp_codegen.md` §5 里列的是“文档与 API 收尾”，
   这一项没有收。
   ——**已在 finish 修复**：roadmap 升到 v9（`f50e3d9214ed`，与一处 pass 修改同提交）。
2. **`caelum/plan/cpp_codegen.md` §2 以现在时描述已经不存在的东西**：`cir-call-to-skeleton`、
   整模块假设、被替换的 host 函数。事实链在 §10，而 §2 没有标注自己是历史（该标注是姊妹分支
   补的）。
   ——**已在 finish 修复**：`8a85c9b10cff` 给 §2 加了“这是 2026-11-05 W2 之前的起点记录”的标注。
3. **`caelum/doc/cpp-codegen-operators-decisions.md` §1.3** 说“标量按值传，进单元后是标量参数”；
   对 reduce 的 `init` 成立，对 map 的变参不成立（问题 3）。
   ——**仍在 finish 上**：finish 改了行为与 `SkeletonOps.h`，但没有改这条决策记录。
4. **`mlir/include/mlir/Dialect/Skeleton/SkeletonOps.h:43-47`** 的“scalar input to skeleton_map
   is broadcast over the destination”在本分支不可达（问题 3）。
   ——**已在 finish 修复**：标量现在真的能传进来，这句话成立；措辞也在 `4edb0d1a6fc9` 里改过。
5. **`CirCallLowering.h:96-103`** 关于 destination 视图的注释与 `CirSkeletonSplit.cpp:299`
   的实现相反（问题 1）。
   ——**已在 finish 修复**：`createToTensor` 的注释现在写了 `writable` 与 `restrict` 的共同约束。
6. **`caelum/plan/operators.md` §5 的 XFAIL 理由**与本分支代码不符（问题 1）。XFAIL 本身是
   诚实的“尚未达成”记录，错误的是原因。
   ——**已在 finish 修复**：`operators.md` §5 与 §8 改写，理由更正为 destination 视图不可写。

## 6. 结论与建议

这条分支的主体是成立的：函数级切分、destination-passing 边界、thunk ABI、混合模块 driver、
一条命令、纯函数词汇、半自动路径清场，都能在树里找到代码、测试与决策记录；负例覆盖了
`cpp_codegen.md` §4.4 列出的前三类（裸指针、一个 host 多次骨架调用、host 调纯函数），
第四类“单元签名对不上”属于 ABI pass，没有用例（问题 6）。作为一个可继续迭代的基线，
它没有问题。

在**不合并 finish、只把这条分支当最终状态**的前提下，它需要解决的是三件：

1. **reduce 的 destination 改成可写视图**，并把 `GPU/cpp-softmax` 的 XFAIL 连同它错误的理由
   一起删掉。这是最硬的一处“文档说已经达成、代码说没有”。
2. **thin-wrapper 的删除规则收紧**：只删有结果且无使用的操作；否则 wrapper 里的一句赋值会被
   静默吃掉。这是静默错编，优先级高于任何便利性。
3. **把 `skeleton-lower-host-abi` 纳入测试视野**（正例 + 三条负例），并把 `runDriver` 的入口
   `cast` 改成按类型分派——这两处一个没有覆盖、一个在合法输入上会崩。

按“把 finish 合进来”的口径，这三件里的第 1、2 件已经在 finish 上做完了（见开头的汇总表），
第 3 件里的 ABI 测试与 `cast` 都还在。合并视角的清单是：

- **不需要动作**：代码问题 1、2、3、10 与文档问题 1、2、4、5、6 都已在 finish 上解决。它们留在
  这里只解释“现在代码为什么长这样”。
- **合并前/后要修**：代码问题 4（driver 的 `cast`）、5（ABI 的 stride）、6（ABI 没有用例）、
  7（dispatch 静默跳过）、8（span 长度宽度）、9（`skeleton++` 的临时目录与 `--link-arg`），
  以及文档问题 3（decisions §1.3 的“标量按值传”）。其中 6 与姊妹审查的问题 4 是同一件事，
  一起做。

这七条都不阻塞合并本身：它们要么是“合法输入上会崩”（4）、要么是潜在形状（5）、要么缺测试与
诊断（6/7/8）、要么是工具与记录的小账（9、文档 3）。但如果目标是“合进去的代码是可维护的
基线”，4 与 6 建议在合并时一并处理——一个是崩溃，一个是 ABI 层唯一的测试缺口。
