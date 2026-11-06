# C++ 端到端：现状、缺口与顺序

> 2026-11-05 · 本文只分析 C++ 前端这条线：现在编到哪一步、要补什么、按什么顺序补。
> 方向与运行时以 [`roadmap.md`](roadmap.md) 为准，算子这条线以 [`operators.md`](operators.md)
> 为准；与它们冲突时以前两份为准，回来改本文。运行时细节见
> [`../doc/runtime.md`](../doc/runtime.md)。

## 1. 结论

- 今天 C++ 在这套系统里的角色是**算子描述语言**，不是程序语言。`.cpp` 里只能写纯函数和
  对 `skeleton_map` / `skeleton_reduce` 的薄包装；`main`、控制流、分配、打印必须手写 MLIR，
  再用脚本剥壳、`cat` 拼成一个模块。四个 C++ e2e 全是这种两段拼接。
- 缺的不是 CIRGen 的 AST→IR——CIR 已经把整个程序（包括 `main`、`printf`、`std::vector`）带进
  了 MLIR。缺的是三样：**切分**（哪些函数变成 op 单元、其余留给 host）、**边界 ABI**
  （C++ 的指针/span ↔ skeleton 的 memref/tensor）、**driver 接受 LLVM 层的 host**。
- 因此不需要为了业务代码去改 clang 的 CodeGen。需要动的是 `clang/lib/CIR` 里的切分 pass、
  一条 host/单元 ABI 的 lowering、以及 driver 的输入与入口。只有选「整个程序都降成 MLIR
  core」那条路时，才需要新写一个 CIR→core 的编译器后端；本文不建议那条路（见 §4.1）。
- 顺序上，这份工作**不是**「operators.md 全部做完再做」，也**不是**「编译器做完再开始算子」。
  正确的位置是：先定算子的边界 API（属于 operators.md §4），前端与第一个算子的实现并行，
  benchmark 与算子扩张放在前端之后。理由见 §6.2。

## 2. 现在编到哪一步

> **这一节是 2026-11-05 W2 之前的起点记录**，保留它是为了说明当时缺什么。它描述的
> `cir-call-to-skeleton`、整模块假设、被替换的 host 函数都已经不存在；现在的事实链见 §10。

### 2.1 事实链

| 阶段 | 现在的命令 | 产物 | 谁做的 |
|---|---|---|---|
| C++ → CIR | `clang -cc1 -fclangir -emit-cir` | CIR module（含全部函数） | clang |
| CIR → skeleton | `cir-opt -cir-call-to-skeleton` | `func.func` + `skeleton.*` | `clang/lib/CIR` 的 pass |
| 剥壳 + 拼接 | `skeleton-peel-module.sh` + `cat *.driver` | 一个 implicit module | 测试脚本 |
| skeleton → 机器码 | `mlir-skeleton-driver` | JIT，或 `--emit` 产物 | driver |

证据在 [`cpp-map-add.mlir`](../../mlir/test/Integration/Dialect/Skeleton/CPU/cpp-map-add.mlir)
的 RUN 行，以及 [`cpp-map-add.cpp`](../../mlir/test/Integration/Dialect/Skeleton/CPU/cpp-map-add.cpp)
开头的注释：`@main` 手写的原因是 "a C++ main cannot yet ride the whole CIR -> Skeleton -> LLVM
chain"。

### 2.2 三个硬限制

1. **整模块假设。** `CIRCallToSkeletonPass` 在末尾检查残留的 `cir.func`，有就报错：
   "cir-call-to-skeleton requires the whole module to be skeleton code (pure functions,
   skeleton op declarations, and their hosts), surrounding ordinary C++ code is not supported
   yet"。所以 `main`、调用 host 的人、任何普通函数都编不过。见
   [`CirCallToSkeleton.cpp`](../../clang/lib/CIR/Dialect/Transforms/CirCallToSkeleton.cpp)
   的 `runOnOperation` 末尾。
2. **host 函数被整体替换。** `rewriteToStandardFunc` 丢掉原函数体、按算子语义重新发明返回类型，
   把 wrapper 改写成 memref/tensor 签名的 `func.func`。改写之后只有 MLIR 能调用它，C++ 调用方
   看到的还是原来的指针签名，两边对不上。同一处还留了一条 TODO：host 必须是"只包一次 skeleton
   调用的薄包装"，否则多余的语句会被静默丢掉。
3. **纯函数只支持单个 block 的直线标量算术。** `populateArithFuncBody` 明确拒绝多 block 函数体；
   `CirFuncToArith` 只处理常量、加减乘除、取负、`fneg`、`fmuladd` 展开，控制流、比较、cast、调用、
   逃逸的 alloca、volatile/atomic 都带诊断拒绝。lambda 作为 `pure_fn` 也还没解析。

### 2.3 所以「C++ 只能写算子」

- 被标注的 wrapper 是**边界**，但它的 C++ 签名在 pass 里被替换掉了；普通 C++ 函数无法调用它。
- 形状信息来自参数类型（数组引用给静态维、`skeleton_detail::span` 给动态维），这是对的，
  但输出张量的分配、返回值的所有权、长度的运行时校验，全部由改写后的 MLIR host 承担，
  C++ 侧没有对应的表达。
- 半自动路径（从带注解的循环反推算子）还在树上，roadmap §2.1 已经决定删除；它是另一套"在
  C++ 里写计算逻辑"的入口，不属于这条线，但容易被当成现状的一部分。

### 2.4 roadmap §5 的推进情况

| roadmap §5 | 状态 | 说明 |
|---|---|---|
| 1 删除半自动路径 | 完成（2026-11-05） | `CirLoopToSkeleton`、其注册与三组测试已删除；见 §10 |
| 2 运行时第一版 | 完成 | M1–M5 与 2026-10-31 / 11-03 的审查修复；只剩"`target-lower` 是否收进 driver"未定 |
| 3 远程构建与测试 | 完成 | `.profile/remote`，三组测试按脚本过滤规则跑 |
| 4 GPU 正确性 | 进行中 | UVM 已替换旧桥接；原 20% 失败未复现；动态形状仍被 `target-lower` 拒绝 |
| 5 算子集合与 benchmark | 第一个算子完成 | softmax 以扁平 DAG 实现；API 见 §10 与决策文档；benchmark 未做 |

C++ 前端不是 roadmap §5 里单独列出的一项，它是 operators.md §3 的主体、§5 验收的前提，因此
可以和 §5.1（删半自动）以及 §5.4 的 GPU 线（含动态形状）并行推进；其中 §5.1 建议先做，改动小
且能少一类"整模块假设"的干扰。

## 3. 目标拆解

operators.md §5 的验收是：**整个程序（算子 + `main` + 输出）用 C++ 写，一条命令编出来，
driver 与产物两条路径跑出期望数值，CPU 与 GPU 各一次。** 把它拆成六个子问题：

1. **整程序前端**：clang 已经能把整个程序变成 CIR；这一格基本是现成的。
2. **切分**：把带 `skeleton.region` 的 wrapper 及其内部的骨架算子变成 op 单元，其余函数留给 host。
3. **host 降级**：把剩下的 C++（`main`、控制流、标准库调用、打印）降到机器码。
4. **边界 ABI**：C++ 的参数/返回值与 skeleton 单元的 memref/tensor 签名怎么对上，谁分配、谁释放、
   长度从哪来、静态维和动态维怎么表示。
5. **构建与运行集成**：一条命令、driver 模式、产物模式、CPU/GPU。
6. **语义与诊断**：支持哪些 C++、哪些写法必须报错而不是静默错编。

第 2–5 条是本文的主体；第 6 条是贯穿性的验收条件。

## 4. 关键设计选择

### 4.1 host 用什么表示

| 方案 | host 表示 | 能覆盖的 C++ | 新增工作 | 风险 |
|---|---|---|---|---|
| A 统一 MLIR | memref/tensor/arith/scf | 很小：指针算术、`malloc/new`、可变参数 `printf`、`std::vector/std::string`、异常都难表示 | 新写 CIR→MLIR core 的整套 lowering | 等于再写一个后端；memref 也表达不了别名、生命周期和 C++ 对象 |
| B 分离（推荐） | clang → CIR → LLVM dialect/IR | 日常 C++（无异常/RTTI），标准库与 I/O 天然可用 | 切分 pass、ABI lowering、driver 输入与链接 | 两套 IR 在边界上要对齐；shim 的 ABI 要生成并校验 |
| C 改 codegen | 在 CIRGen 里直接产出 skeleton | 与 B 相同 | 改 CIRGen/CodeGen 的语句与表达式 | 不必要：注解信息已经在了，缺的是切分与降级 |

**推荐 B。** 理由：

- 业务代码不可避免要碰 libc、libstdc++、可变参数和指针。这些东西只以原生机器码存在，
  只有走 LLVM 才能自然地调用；把它们搬进 memref/tensor 世界，工作量不是"多写几个 pattern"，
  而是重做内存模型。
- 边界只需要跨一次（每个算子调用一次），而统一 MLIR 要让整个程序都跨过去。
- "混合 codegen"的主张仍然成立：CIR 层是同一个 module，标注部分在 CIR pass 里变成 skeleton
  单元，其余部分仍然走 CIR；到 lowering 时才分成两条后端。最终两边都落到 LLVM，运行时只看到
  机器码和 skeleton IR。区别只是 host 在 lowering 后不再保持 memref/tensor 形态。

**关于"改 clang codegen"的记忆。** 对了一半：需要动 clang，但动的是
`clang/lib/CIR/Dialect/Transforms`（切分、ABI）与 clang 的**driver/frontend action**（把
`-emit-cir`、切分、`cir-to-llvm` 串成一条），不是 AST→IR 的 CodeGen。CIRGen 已经能生成
`main`、调用、控制流、全局变量和标准库调用；`__attribute__((annotate))` 也已经随函数保留到
CIR。只有当要求"host 也必须用 memref/tensor 表示"（方案 A）时，才需要写新的 CIR→core
lowering——那仍然是一个 lowering，不是 CodeGen。真正会碰到 CodeGen 的是**语句级注解**
（半自动路径在 `CIRGenStmt.cpp` 里加过 `attachLoopAnnotations`），而那条路径正在删除。

review 20260917 说的"在 codegen 阶段同时产出两种方言"，在实现上可以落成"CIRGen 之后、lowering
之前的一个 CIR pass"：标注信息和参数类型都还在 CIR 里，不需要回到 AST。只有当以后的 API 需要
CIR 已经丢掉的源码信息（例如从函数体内部的语句推断边界、或语句级标注）时，才需要真正动
CIRGen；当前的薄包装 + 参数类型携带形状不需要。这一点决定了 W2 是一个 pass 而不是一次
codegen 重写。

### 4.2 边界放在哪

现在：被标注的 wrapper 就是边界，pass 把整个 `cir.func` 删掉、换成一个 memref 签名的
`func.func`。结果是 C++ 调用方与实现被拆散。

建议：**wrapper 保留符号和 C++ 签名，变成 host stub；它内部的骨架算子才变成 op 单元。**
这与 review 20260917「前提 3」的方向一致（"宿主改成函数级、保留符号（wrapper + 内部
skeleton op），签名/长度/返回在 host–skeleton 边界上定"）。

```
C++ 源
  ├─ 普通函数 / main ── CIR ── convert-cir-to-llvm ── LLVM ── 机器码
  └─ 带 skeleton.region 的 wrapper
        ├─ 单元：skeleton op + memref 签名 ── driver 编译 ── CPU / GPU 实现
        └─ host stub：保留 C++ 签名 ── ABI pass 填入 shim ── resolve + 间接调用
```

- 普通 C++ 调用 `vec_add(a, b, c)` 时看到的就是 `vec_add` 的原签名，链接期不需要知道单元。
- 单元只包含骨架算子本身，签名由 pass 决定（memref / 无返回值，见 §4.3）。
- **v1 限制：一个 wrapper 里只有一次骨架调用。** 多个骨架算子写在同一个函数里，需要 pass
  逐语句改写、并在 host 侧为中间结果分配缓冲；这可以放到 v2（见 §6.3 的 M5）。v1 的扁平 DAG
  由 C++ host 里的多个 wrapper 调用表达，正好对应 operators.md §2 的"大图不是大单元"。

### 4.3 边界 ABI

这是整条线上最容易出错、也最值得先定死的部分。

**单元侧。** 建议把单元函数写成**显式输出缓冲、无返回值**：

```
func.func @__skeleton_op_0(%a: memref<?xf32>, %b: memref<?xf32>, %out: memref<?xf32>)
```

单元内部把 `%out` 通过可写的 `bufferization.to_tensor` 变成 tensor，作为骨架算子的 `outs`；
目标是 bufferization 之后写入直接落在 `%out` 上，这样单元 lower 后的签名就是"N 个 memref
描述符、无返回值"，不会出现"tensor 返回值在 bufferize 时被改成额外的 out 参数"这种隐藏参数，
shim 可以按固定规则生成。**这是 W4 的第一个验证点**：如果 bufferization 仍然给单元加出参，
就说明 `to_tensor` 的可写别名没有被认出来，ABI 方案要重新看，不能假设它成立。

**host 侧。** 保留 C++ 的自然写法：数组引用给静态维、`skeleton_detail::span<T>` 给运行时维；
裸指针没有长度，**在边界上拒绝**（今天它会被映射成动态 memref，长度由调用方"碰巧"提供，
在 C++ 调用方这里不再成立）。返回值改成输出参数（destination-passing）：

```cpp
extern "C" __attribute__((annotate("skeleton.region", "CPU")))
void vec_add(skeleton_detail::span<const float> a,
             skeleton_detail::span<const float> b,
             skeleton_detail::span<float> out);
```

理由：

- 所有权明确：缓冲由调用方分配（`std::vector`、栈数组都行），单元不分配、不释放，产物与
  driver 两条路径行为一致。
- 动态长度有来源：span 的 `size` 字段；静态长度在类型里。
- 单元里的 `skeleton.map` 仍然是 tensor→tensor，`outs` 用 `%out` 的 tensor 视图，语义不变。

**shim（边界函数）。** 在 host 已经降到 LLVM dialect 之后，由一个 ABI pass 为每个 host stub
生成函数体：从 C++ 参数里取出指针和长度，构造 memref 描述符（allocated/aligned/offset/sizes/
strides），调用 `skeleton_rt_resolve(op_id)` 拿实现指针，按单元的签名做带类型的间接调用。
shim 的输入签名就是 C++ 的签名，输出是单元的 memref 签名；两边的 `LLVMFunctionType` 由 driver
现有的 ABI 校验兜底。

**元数据。** 切分 pass 要在 host stub 上留下足够的信息，ABI pass 才能生成 shim，至少包括：

- 单元 id 与单元函数符号；
- 每个 C++ 数据参数对应哪个单元操作数、是静态维还是运行时长度、静态维的数值；
- 输出参数的位置与形状；
- preference 与 `bytes`。

建议放在 host stub 的一个 discardable 属性里（例如 `skeleton.host_abi`），由 driver 在切分时
读走。不要依赖参数顺序去猜。

**对齐与布局。** memref 描述符的布局由 `finalize-memref-to-llvm` 决定，C++ 侧的 span 是
`{指针, 长度}` 的标准布局；shim 是唯一同时看到两边的地方，所以描述符的构造只在 shim 里做。
需要一条 ABI 回归测试：打印/检查描述符的字段，确保 host 与单元看到的 offset/sizes/strides
一致；data layout、target triple、指针宽度都要跟随同一个目标。

### 4.4 第一版的 C++ 范围

按这次确认的目标：**日常 C++，无异常、无 RTTI，允许 `std::vector` / `std::string` / `printf`。**

| 类别 | v1 | 说明 |
|---|---|---|
| 函数、控制流、递归 | 支持 | 走 CIR→LLVM |
| POD、数组、指针算术 | 支持 | host 侧是普通 LLVM 内存 |
| `std::vector` / `std::string`（局部对象） | 支持 | 构造/析构是普通调用；分配走 libstdc++ |
| `printf` | 支持 | 变参调用走 LLVM |
| `std::cout` / 静态对象 | 谨慎 | JIT 不会自动跑 `llvm.global_ctors`；产物路径由 crt 跑。v1 用 `printf` |
| 异常、RTTI | 关闭 | `-fno-exceptions -fno-rtti`；标准库的 `throw` 路径退化为 abort |
| 协程、部分 builtin | 按 CIR 现状 | CIR 仍在上游化，先用特性矩阵验证，缺的报错而不是静默 |

**算子纯函数**仍然只支持单个 block 的直线标量算术。这不是 host 的限制：真实的算子（softmax、
layernorm）应该写成扁平 DAG 的组合，纯函数只做逐元素/逐归约的标量计算。若第一个真实算子需要
分支或循环，再单独扩 `CirFuncToArith`，不要为了它把整个 C++ 的语义塞进纯函数。

**诊断原则**：所有"不支持"都必须在编译期带定位报错。今天的 pass 已经对纯函数、span 成员、
形状不一致做了这件事；新的切分/ABI 要延续，尤其是裸指针边界、一个 wrapper 多次骨架调用、
host 调用纯函数、单元签名对不上这四类。

## 5. 要做的具体工作

规模是相对量级（S/M/L），不是工期；依赖关系见 §6.1。

| ID | 工作 | 规模 | 主要落点 |
|---|---|---|---|
| W0 | 删除半自动路径，清场 | S | `clang/lib/CIR`、测试（roadmap §2.1） |
| W1 | 定边界 API 与 ABI 规范 | M | `SkeletonOps.h`、本文、operators.md §4/§6 |
| W2 | CIR 切分 pass：函数级、保留符号、产出单元与元数据 | L | `CirCallToSkeleton` / 新 pass |
| W3 | host 的 CIR→LLVM 集成与模块形态 | M | cir-opt / clang frontend action |
| W4 | host↔单元 ABI lowering：shim、描述符、DPS | L | driver 或新的 MLIR pass |
| W5 | driver：新输入形态、入口、切分、产物链接 | M | `mlir-skeleton-driver` |
| W6 | 一条命令的前端 | S/M | clang driver（JobAction）或包装工具 |
| W7 | 测试：单文件 e2e、C++ 特性矩阵、负例、迁移四个旧用例 | M | `mlir/test/Integration/...`、`clang/test/CIR/...` |
| W8 | 文档与 API 收尾 | S | `caelum/doc/`、`caelum/plan/` |

### W0 清场

按 roadmap §2.1 删除 `CirLoopToSkeleton`、`Passes.td` 注册、`CIRGenStmt.cpp` 里的
`attachLoopAnnotations` 及其形参、`skeleton-loop-path.cpp`、`cir-loop-to-skeleton.cir`、
`loop-map-add.*`（半自动路径的）。这是独立工作，先做，能少一类"整模块假设"的干扰。

### W1 定边界 API 与 ABI 规范

产出不是代码，而是三份决定：

1. C++ 侧怎么写算子调用（span / 数组引用 / 输出参数；是否保留返回值的写法给纯 MLIR 测试用）；
2. 单元的签名规则（memref 参数顺序、输出缓冲、返回值 void、preference 从哪来）；
3. host stub 的元数据格式（`skeleton.host_abi` 的字段）。

这三份决定必须先于 W2/W4，否则前端会建在一个会被第一个真实算子推翻的 API 上。它们也属于
operators.md §4.3（host 侧怎么调用）与 §6（未定项）。

### W2 CIR 切分 pass

- 去掉"整模块必须全是 skeleton"的检查，改成函数级：
  - 带 `skeleton.region` 的 wrapper：内部的骨架调用变成一个 `func.func @__skeleton_op_N`，
    带 `skeleton.op_id` / `preference` / `bytes`；wrapper 本身保留符号与 C++ 签名，函数体清空，
    挂上 `skeleton.host_abi` 元数据。v1 要求 wrapper 是薄包装，否则报错（不要静默丢语句）。
  - 其他函数：原样保留，交给 W3。
  - 纯函数：仍然是 `func.func`（arith body），只被单元引用；如果 host 代码调用它，报错。
- 不再在切分时删掉模块上的 `cir.*` / `dlti.*` 属性。现在的 `dropCirModuleAttrs` 是因为下游
  MLIR 不注册 CIR 方言；混合模块还要靠这些属性做 CIR→LLVM 和 data layout，必须在 lowering
  之后再处理。
- 单元里的全局/常量引用怎么随单元走：复用 `splitUnits` 现有的克隆逻辑，但要在切分阶段就
  保证单元是自包含的（纯函数、global），否则 driver 切分时才发现缺符号。
- 负例：裸指针数据参数、一个 wrapper 两次骨架调用、wrapper 里有非骨架语句、host 调用纯函数。

### W3 host 的 CIR→LLVM 集成

- 在切分之后对**剩余 CIR** 跑 `cir-to-llvm`（`populateCIRToLLVMPasses`），产出 LLVM dialect
  的 host。单元与纯函数不参与这次转换。
- 模块形态：一个 module 里同时有 `llvm.func` host、`func.func` 单元、`func.func` 纯函数。
  `convert-cir-to-llvm` 只转 CIR，其他原样保留。
- 入口：C++ 的 `int main()` 要变成一个 `void()` 的入口（例如 `__skeleton_entry` 调 `main`
  并丢掉返回值）。driver 现在只接受 `func.func` 的 `() -> ()` 入口；W5 要放宽到 `llvm.func`。
- 需要验证的 C++ 特性按 §4.4 的表逐项过一遍；CIR 缺什么就记下来，先报错，不要静默。

### W4 host↔单元 ABI lowering

- 单元签名规则落地：`(ins..., out) -> ()`，`out` 通过可写 `to_tensor` 变成骨架算子的 `outs`。
- 为每个 host stub 生成 shim：描述符构造、`skeleton_rt_resolve`、带类型的间接调用。
- 描述符字段：allocated/aligned 指针、offset=0、sizes、strides（identity layout，v1 只支持
  连续）；静态维从元数据来，动态维从 span 的 `size` 来。
- 校验：shim 看到的单元签名必须与单元 lower 后的 `LLVMFunctionType` 一致；不一致直接报错，
  不要等到运行。
- 位置：可以做成 driver 编译单元时生成，也可以做成 host 降级后的一个 MLIR pass。建议后者：
  ABI 是编译期属性，driver 只消费结果；但两者的签名推导必须共用同一个函数，避免各写一份。

### W5 driver

- 输入从"未切分的 skeleton module"扩成"已切分的混合 module"：host 是 LLVM dialect，单元是
  带 `skeleton.op_id` 的 `func.func`，host stub 带 `skeleton.host_abi`。
- `skeleton-outline` 在这些单元上已经是幂等的（父函数带 `skeleton.op_id` 时跳过），driver 只需
  跳过"没有可 outline 的算子"的情况；`splitUnits` 基本可复用。
- 入口查找支持 `LLVM::LLVMFuncOp`；`invokePacked` 本身不关心是哪种函数。
- 产物路径：host 已经是 LLVM dialect，`writeAndLink` 照常能翻译成 LLVM IR；需要把入口重命名
  逻辑从 `func::FuncOp` 扩到 `llvm.func`，并继续生成 `main` + `skeleton_program_init`。
- 嵌入的 `.skeleton.ir` 仍然是单元 IR 的 bytecode；单元元数据（id/preference/bytes）继续写在
  单元函数上，保证产物与 driver 重编号一致。

### W6 一条命令

先用包装工具验证全链路：

```
skeleton++ app.cpp -o app
# 等价于：
# clang -cc1 -fclangir -emit-cir app.cpp -o app.cir
# cir-opt app.cir --cir-skeleton-split --cir-to-llvm -o app.mlir
# mlir-skeleton-driver app.mlir --emit app --runtime-lib=...
```

等 API 和 ABI 稳定后，再把它接进 clang driver（`clang/lib/Driver/ToolChains/Clang.cpp` 的
JobAction），提供 `-fskeleton` 之类的开关。注意这一步动的是 driver，不是 CodeGen；不要一开始
就把管线塞进 `-cc1`，否则调试成本高。

### W7 测试

- **单文件 e2e**：`main` + 一个算子 + `printf`，全部 C++，CPU；然后 GPU。数值有期望，driver
  与产物各跑一次。这是 operators.md §5 的验收基础。
- **C++ 特性矩阵**：POD、循环/分支、`std::vector`、`std::string`（局部）、`printf`、全局常量、
  函数指针、模板实例化。每项一个最小程序，确认能编、能跑。
- **负例**：裸指针边界、wrapper 里多次骨架调用、host 调用纯函数、动态形状 GPU、形状不一致。
- **迁移**：`cpp-map-add*`、`cpp-reduce-sum` 四个两段式用例改成一个 `.cpp`；删掉
  `skeleton-peel-module.sh` 与 `cat` 拼接。保留纯 MLIR 用例，它们验证运行时，不依赖前端。
- **GPU**：静态形状的单元先跑通（今天 `cpp-map-add-static-gpu` 已经能从 C++ 编到 GPU）；
  动态形状（span）仍受 `skeleton-target-lower` 拒绝，属于 roadmap §5.4 的独立工作，不要混在
  这条线里。

## 6. 顺序

### 6.1 依赖

```
W0 ── 独立，先做

W1 ──> W2 ──> W4 ──> W5 ──> W6 ──> W7(单文件 e2e)
         └──> W3 ──────┘
W7(特性矩阵/负例) 贯穿 W2–W6
```

W3 与 W4 并行，W5 需要两者都完成（host 已经能降级，且 ABI 已经能生成 shim）。

关键路径是 W1 → W2 → W4 → W5 → W7。W3 与 W2 可以并行；W6 在 W5 之后；W0 不阻塞任何东西，
但越早做越少干扰。

### 6.2 与 operators.md 的关系

operators.md 的五块内容与这条线的关系：

| operators.md | 关系 | 时间 |
|---|---|---|
| §2 组合方式（扁平 DAG） | 已定；决定单元粒度与中间结果走内存 | 已完成 |
| §3 前端：程序逻辑也要能用 C++ 写 | 就是本文 | 本文 |
| §4.1 skeleton 侧怎么表示 | 决定单元里有哪些骨架算子 | W1 之前 |
| §4.2 前端入口 / annotation | 本文 W1/W2 | W1 之后 |
| §4.3 host 侧怎么调用 | 本文 W1/W4（ABI、所有权） | W1 |
| §5 验收（整程序 C++） | W7 | 前端之后 |
| §6 benchmark / 算子扩张 / 融合 | 需要前端跑真实负载 | 前端之后 |

**建议的顺序（按里程碑）：**

1. **现在**：W0 清场（roadmap §5.1）。它与前端无关，先做。
2. **M1（设计）**：operators.md §4 先走到"第一个算子选定 + skeleton 图定形 + host API/ABI 定稿"。
   同时用**手写 MLIR host** 把第一个算子实现出来，验证骨架算子够不够、CPU/GPU 能不能跑。
   这是 operators.md §5 允许的临时形态。
3. **M2（前端风险回收）**：用 W2/W3/W5/W6 跑通一个最小的单文件 C++ 程序（`main` + 一个已有
   算子 + `printf`，CPU），先把 CIR 成熟度、切分、host 降级、driver 入口这四个不确定性消掉。
   这一步不依赖第一个真实算子。
4. **M3（ABI 与迁移）**：W4 的 shim、DPS、元数据落地；四个旧 e2e 改成单文件；这时 C++ 前端
   对已有算子是完整的。
5. **M4（真实算子）**：把第一个算子的 host 与 `main` 换成纯 C++，跑通 driver + 产物、CPU + GPU，
   记录数值与耗时。这是 operators.md §5 的正式验收，也是 benchmark 的起点。
6. **M5（扩张）**：多骨架调用的 host 函数、动态形状 GPU、更多算子、融合；这些回到 operators.md
   的节奏。

**为什么不是"operators.md 全部做完再做"**：operators.md 的 benchmark、算子扩张、融合都不依赖
C++ 前端，而 §5 的验收依赖它。把前端排到整个 operators.md 之后，会把"程序真的能用 C++ 写"
这件事推到 benchmark 之后，前端的风险（CIR 成熟度、ABI）也会在最紧的时候才暴露。

**为什么不是"前端全部做完再开始算子"**：前端的 API/ABI 必须由真实算子来检验；没有第一个算子，
就只能凭空设计 span/输出参数/单元签名，等 softmax 或 layernorm 一来很可能推翻。而且
operators.md 的第一个算子用手写 MLIR host 就能验证，正是为了让两条线并行。

一句话：**API 在前，实现并行，验收在前端之后，扩张在最后。**

### 6.3 里程碑与出口条件

| 里程碑 | 内容 | 出口条件 |
|---|---|---|
| M0 | W0 清场 | 半自动路径及其测试删除，skeleton 三组测试仍全绿 |
| M1 | 边界 API/ABI 规范 + 第一个算子（手写 MLIR host） | 第一个算子 CPU 数值正确；GPU 可跑或明确记录阻塞；API 文档定稿 |
| M2 | 最小单文件 C++（已有算子 + `main` + `printf`） | 一条命令编出、driver 与产物都输出期望值 |
| M3 | ABI shim + DPS + 旧用例迁移 | 四个旧 e2e 变成单文件；负例有诊断 |
| M4 | 第一个真实算子的纯 C++ 版本 | operators.md §5 全部判据通过（CPU/GPU、driver/产物、数值、耗时） |
| M5 | 多骨架调用 host、动态形状 GPU、更多算子 | 每项独立验收，不与 M4 混在一起 |

## 7. 风险

- **CIR 的成熟度**：CIR 仍在上游化，异常、协程、部分 builtin 可能缺。缓解：M2 先跑特性矩阵；
  缺的先用 `-fno-exceptions -fno-rtti` 和 `printf` 绕开，并明确记录；不要为了一个 builtin 去改
  CIRGen。若真的出现无法绕开的 CIR 缺口，备选是 host 走 legacy CodeGen + AST 侧的切分，但那是
  另一条大得多的路，只有在前端已经证明有价值时才考虑。
- **ABI 脆弱**：memref 描述符是 MLIR 的内部 ABI，span 是 C++ 的布局。缓解：描述符只在 shim 里
  构造；host 与单元共用同一套 bufferization 配置；driver 比较 `LLVMFunctionType`；加描述符字段
  的回归测试。
- **JIT 不跑全局构造**：`std::cout`、静态对象这类依赖 `llvm.global_ctors` 的写法在 driver 模式下
  不会自动初始化。缓解：v1 用 `printf`；需要时 driver 在入口前显式跑 ctors，或这类程序只走产物。
- **两套 host 形态**：测试用 MLIR core host，C++ 用 LLVM host。缓解：driver 同时支持两种输入；
  纯 MLIR 用例继续走原路径，C++ 用例走新路径，不要急着合并。
- **一个 wrapper 一次骨架调用**：v1 的限制会限制业务代码的写法。缓解：扁平 DAG 由多个 wrapper
  表达；M5 再让 pass 处理一个函数里的多个骨架算子。
- **shim 的开销**：每次算子调用多一次 resolve + 描述符构造。缓解：resolve 已经是 O(1)；描述符是
  几条标量指令；先测 M4 的耗时，不在没有数据前优化。
- **动态形状 GPU**：span 仍是 `memref<?xf32>`，`skeleton-target-lower` 拒绝；这是 roadmap §5.4
  的独立问题，不要把它塞进前端验收。
- **clang driver 集成**：`-fskeleton` 要动 JobAction 和测试。缓解：W6 先用包装工具，driver
  集成后置。

## 8. 未定问题

- C++ 侧用项目自己的 `skeleton_detail::span`，还是接受 `std::span`（布局不保证）？建议前者，
  `std::span` 只作为可转换的便捷入口。
- 是否保留返回值的写法给纯 MLIR 用例（今天的 `skeleton.map` 返回 tensor）？建议保留，C++ 边界
  用 DPS，二者在单元内部可以统一成 `outs`。
- 单元里能不能有多个骨架算子？v1 不行；如果第一个真实算子需要，再讨论融合/中间缓冲。
- host 能不能调用纯函数？v1 不行（纯函数只被单元引用）；如果业务代码需要，走普通 C++ 函数。
- 长度不一致的运行时检查放在哪：今天的 `cf.assert` 在 host；DPS + 单元内检查更自然，但要和
  GPU 桥接的错误处理统一。
- 错误处理：运行时 `resolve` 失败是 `exit(1)`（runtime.md §16），C++ 侧要不要暴露可恢复的错误？
  这是运行时的问题，不是前端，但 C++ API 的错误约定要等它定。

## 9. 参考

- [`roadmap.md`](roadmap.md)：方向、运行时现状、接下来。
- [`operators.md`](operators.md)：组合方式、第一个真实算子、验收。
- [`../doc/runtime.md`](../doc/runtime.md)：driver、ABI、产物、单元编译。
- [`../review/20260917.md`](../review/20260917.md)：混合 codegen 的方向与"前提 3"。
- [`CirCallToSkeleton.cpp`](../../clang/lib/CIR/Dialect/Transforms/CirCallToSkeleton.cpp)：
  整模块假设与 host 替换。
- [`CirCallLowering.cpp`](../../clang/lib/CIR/Dialect/Transforms/CirCallLowering.cpp)：
  wrapper 改写、span/数组引用的形状来源。
- [`CirFuncToArith.cpp`](../../clang/lib/CIR/Dialect/Transforms/CirFuncToArith.cpp)：
  纯函数支持范围。
- [`mlir-skeleton-driver.cpp`](../../mlir/tools/mlir-skeleton-driver/mlir-skeleton-driver.cpp)：
  输入、切分、host lower、产物。

## 10. 实施记录（2026-11-05）

W0–W7 做完了，W8 是本文。每一步的取舍记在
[`../doc/cpp-codegen-operators-decisions.md`](../doc/cpp-codegen-operators-decisions.md)。

| ID | 状态 | 落点 |
|---|---|---|
| W0 删除半自动路径 | 完成 | `CirLoopToSkeleton` 及 `Passes.td`/`cir-opt`/`CIRGenStmt` 的注解形参、三组测试全部删除 |
| W1 边界 API 与 ABI | 完成 | `mlir/include/mlir/Dialect/Skeleton/SkeletonOps.h`、决策文档 §1–2 |
| W2 CIR 切分 pass | 完成 | `cir-skeleton-split`（`CirSkeletonSplit.cpp`），替代 `cir-call-to-skeleton` |
| W3 host 的 CIR→LLVM | 完成 | `cir-opt` 注册混合模块需要的方言；新增 `cir-strip-cir-attrs` 清掉 CIR 残留属性；入口放宽到 `llvm.func` |
| W4 host↔单元 ABI | 完成 | `skeleton-lower-host-abi`：thunk 体、memref 描述符、单元签名校验 |
| W5 driver | 完成 | 接受混合模块；`llvm.func` 入口；产物 `main` 返回程序退出码 |
| W6 一条命令 | 完成（包装工具） | `mlir/tools/skeleton++`，三个工具串起来；clang driver 的 `-fskeleton` 没做 |
| W7 测试 | 完成 | 单文件 e2e（map/reduce/softmax/特性矩阵）、GPU 一份、负例两组 |

**与本文当初判断不同的地方**（细节见决策文档）：

- **§4.2 的「wrapper 保留符号、内部算子变单元」照做了，但多了一层 thunk。** wrapper 的
  body 在 CIR 里，memref 描述符是 LLVM 概念；两边分开之后，C++ 参数 → (指针, 长度) 留在
  CIR，指针+长度 → 描述符留在 LLVM pass。
- **§4.3 的 ABI 示例改成了按引用传 span。** CIR 把按值的 aggregate 在函数边界上拆开，
  参数类型里就没有 span 了，形状只能从 codegen 的 coercion 序列里反推。
- **§4.3 的「W4 第一个验证点」是成立的**：单元用可写 `to_tensor` 当骨架算子的 `outs`
  之后，bufferization 仍然把单元 lower 成「N 个 memref 字段、无返回值」，没有隐藏出参。
- **归约不再从零开始**：§2.2 说纯函数只做逐元素/逐归约的标量计算，但 identity 是算子语义
  的一部分，现在由调用点传入（`skeleton_reduce(fn, out, in, init)`）。
- **§D 的「host 调用纯函数 v1 不行」被推翻了**：纯函数在切分之后就是一个 func.func，host
  调它就是普通调用，`cir-to-llvm` 能把它降到 `llvm.call`。`CPP/cpp-features` 覆盖了这条。

**还没做的**：clang driver 的 `-fskeleton`（§W6 后半）、一个 host 里多个骨架算子（§6.3 M5）、
扁平图上的融合 pass。

**2026-11-07 补的四项**（都在 §4.1/§4.4 的范围之内）：

- **动态形状上设备**：`skeleton_detail::buffer<T>` 是运行时拥有的存储，与 `span` 同一套两个
  成员，边界上是 `memref<?xT>`；单元带 `skeleton.managed` 时设备侧不做分配与拷贝。
  `GPU/cpp-managed-add` 是这条路径的 e2e。
- **标量参数**：§4.3 说标量按值传给单元，实际上 `skeleton_map` 的变参是引用，标量在调用点
  是一个指向它的指针。切分现在把 destination 之后指向标量的指针当标量值，调用前 load；
  原来这会让 `skeleton_map(fn, out, x, 1.0f)` 报“不是数据参数”。
- **matmul 入口**：`skeleton_matmul(out, a, b)` 没有纯函数，落到 `linalg.matmul`；
  destination 先清零（`linalg.matmul` 是累加语义）。
- **destination 一律可写视图**：map/reduce/matmul 都不再经过 `tensor.empty` + empty tensor
  elimination 或“写视图 + 丢弃结果”的写法。丢弃 fill 的结果会让它在 tensor 层变成死代码，
  这正是“identity 没写进去”的根因。
