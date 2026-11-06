# Caelum 计划

> v9 · 2026-11-07
>
> 这份文档记录方向、决定与接下来要做的事。没写进来的部分都还没定。
> 运行时的设计细节在 `caelum/doc/runtime.md`；完成 roadmap 过程中的取舍记录在
> `caelum/doc/finish-roadmap-decisions.md`。

## 1. 目标

重心是**运行时算子放置**（operator placement）：用户显式写出骨架算子，
放置决策在运行时做出，把每个算子分配到 CPU / GPU 上执行。

编译器不再识别算子，只做两件事：把用户写下的算子落成 IR，
把源码里的 annotation 完整地带下去。识别相关的机制（循环反推、形状猜测）退出主线。

## 2. 设计决定

### 2.1 前端只保留手动路径

- 半自动路径（从带注解的循环反推算子）**已删除**，没有过渡期、没有开关。
- 删除范围：`CirLoopToSkeleton` 及 `Passes.td` 注册、`cir-loop-to-skeleton.cir`、
  `skeleton-loop-path.cpp`、`loop-map-add.*`、`CIRGenStmt.cpp` 里只为循环注解服务的
  `attachLoopAnnotations` 与传 attr 的形参。
- 工作流全部走手动路径：用户显式给出算子，算子上的 annotation 表达放置意图。

### 2.2 annotation：只表达算子与放置目标

语法已定，三个字段：

| annotation | 位置 | 含义 |
|---|---|---|
| `__attribute__((annotate("skeleton.region", "CPU"\|"GPU")))` | 宿主函数 | 这是一个算子边界，参数里的 `"CPU"`/`"GPU"` 是放置偏好 |
| `__attribute__((annotate("skeleton.pure")))` | 纯函数 | 只被算子单元引用的逐元素/逐归约标量计算 |
| `__attribute__((annotate("skeleton.op", "map"\|"reduce"\|"matmul")))` | 头文件里的算子声明 | 哪个 skeleton 原语 |

- 职责止于「算子 + 放置目标」，不承担识别线索。
- 边界是目的地式：第一个数据参数是输出缓冲；形状来自参数类型（`T (&)[N]` 编译期、
  `skeleton_detail::span<T>`/`buffer<T>` 运行时、`T &` 单个元素）。
- 一个宿主函数里只能有一次骨架调用；扁平 DAG 由多个宿主函数表达（见 `operators.md` §2）。
- 单元额外带 `skeleton.managed`：数据参数全部是运行时缓冲时置位，设备侧不再做分配与拷贝。

### 2.3 放置决策在运行时

- 编译期 annotation 是决策依据之一，编译期不做「选设备」。
- `skeleton-preference-partition` 与 `finalize-memref-to-gpu` 已随第一版运行时删除；
  `skeleton-target-lower` 保留，只做「把一个单元降成某个目标的代码」（见 §5.2）。

### 2.4 算子集合

由 benchmark 定的第一批，见 `caelum/doc/benchmark.md`：

| 层 | 算子 | 入口 |
|---|---|---|
| 方言 | `skeleton.map`、`skeleton.reduce`、`skeleton.vector.add` | 手写 MLIR |
| C++ 原语 | `skeleton_map`、`skeleton_reduce`、`skeleton_matmul` | `mlir/Dialect/Skeleton/SkeletonOps.h` |
| 真实算子 | softmax（4 个算子）、layernorm（6 个）、matmul（1 个）、逐元素 add | 扁平 DAG，每个算子一个单元 |

- `skeleton.matmul` 没有纯函数：contraction 本身就是算子，落到 `linalg.matmul`。
- 任意带 `skeleton.preference` 的 linalg op 仍是合法单元（手写 MLIR 用例用它验证运行时），
  但 C++ 侧只暴露上面三个原语。
- 扩张规则：benchmark 需要的算子进矩阵，矩阵之外的不加。

### 2.5 运行时的形态

- **入口是独立 driver**（`mlir-skeleton-driver`），拥有从 skeleton IR 往下的整条编译管线，
  并在自己的进程里执行程序。
- **决策粒度是单个算子**。运行期每个调用点一次 resolve，由运行时决定落到哪个 executor 的
  哪个 device 上。
- **运行期可以回到编译期**：缓存里没有可用实现时，从该单元的 skeleton IR 重新跑
  skeleton → linalg → bufferize → 目标 lowering → JIT。
- **产物是独立可执行文件**：只含 host、各算子的 CPU 实现、内嵌 skeleton IR 与运行时，
  不含 GPU 代码。`./prog` 自己运行是纯 CPU；`driver prog` 读产物里的 IR、按需重编译出
  GPU 实现后执行——GPU 代码的唯一来源是重编译。
- **内存模型是 UVM，分配归运行时**：`skeleton_rt_alloc` 在有 GPU 时给 managed 内存、
  没有时给对齐的 host 内存；`skeleton_detail::buffer<T>` 是它在 C++ 侧的样子。单元的
  数据参数全是它时，该单元被标记 managed，设备侧不再分配、不再拷贝。
- **设备执行在算子边界同步**：`skeleton_rt_sync` 等 device 完成，lowering 把它插在
  GPU 单元调用之后、第一次读回之前。
- **术语**：node（机器）→ executor（执行器，放置决策的落点）→ device（设备）。
  v1 一个进程一个 executor（`skeleton_rt_executor_id` 恒为 0），trace 与 stats 带
  `exec=` 字段；多 executor、跨进程、跨机器的部署形态以后再说。

## 3. 现状

### 3.1 编译器

- 整条手动路径可用：C++ → CIR → skeleton → linalg → CPU/GPU lowering。
- **半自动路径已删除**：代码、测试、远端过滤器都不再引用它。
- **整个程序可以用 C++ 写**：`skeleton++ app.cpp` 一条命令把算子、`main`、控制流、
  标准库调用、`printf` 编成一个程序；`main` 手写 MLIR 的两段式写法已经没有用例。
- GPU 单元同样来自 C++：`cpp-map-add-static`（静态数组）与 `cpp-managed-add`（运行时缓冲）
  各一条。

### 3.2 运行时

第一版可用（`caelum/doc/runtime.md`）：driver 按 op 决策、按需从该单元的 skeleton IR
重编译、在自己的进程里执行；`--emit` 产出只带 CPU 实现与内嵌 IR 的可执行文件，driver
可读回其 IR 并为 GPU 重编译。远端验证全部通过（用例个数以 `remote.sh` 的过滤规则为准）。

2026-10-31 与 2026-11-03 的审查修复补齐了按设备保活的 JIT 引擎、resolve 失败终止、
body-region 算子、编译失败负缓存、GPU 显存探测、跨模块 ABI 校验、产物 O2 与 bytecode
内嵌 IR、算子表指针失效、嵌套图与单元间调用。

2026-11-07 补的三处（决策见 `finish-roadmap-decisions.md` D3/D4/D6）：算子边界的显式
同步、由运行时拥有的 UVM 分配、trace/stats 的 executor 标识。

### 3.3 GPU 执行

两条路径，取决于宿主怎么持有缓冲：

- **宿主缓冲（静态数组）**：`gpu.alloc host_shared`（CUDA 上即 `cuMemAllocManaged`）→
  host `memref.copy` 写入 → kernel 直接访问同一块 buffer → 同步 →
  host `memref.copy` 读回 → `gpu.dealloc`。只读的 `memref.get_global` 不写回。
- **运行时缓冲（`skeleton_detail::buffer<T>`）**：内存本来就在运行时手上，单元标记
  `skeleton.managed`，没有分配、没有进出拷贝；kernel 直接读写宿主随后要读的那块内存。

两者的共同点与限制：identity layout、单输出、静态形状或（运行时缓冲的）一维动态形状。
二维动态形状、非 identity layout 仍被拒绝并给出诊断。

归约在设备上可用：`linalg.reduce` 的迭代类型是 `[parallel, reduction]`，
`convert-linalg-to-parallel-loops` 生成「外层 `scf.parallel`、归约维 `scf.for`」，
映射到设备就是每个输出元素一个线程、归约维串行。softmax 的四个算子全部能上设备。

原先记录的 20% 垃圾输出（2026-10-27 之前的异步 `gpu.memcpy` 路径）没有再复现；之后又
补上了显式同步，异步拷贝路径已不存在。

### 3.4 算子与 benchmark

- softmax、layernorm、matmul、逐元素 add 四个负载，全部 C++、全部有参考数值。
- 设备上：softmax、layernorm、matmul（静态数组，桥接路径）与 add（运行时缓冲，managed
  路径）都通过。
- CPU/GPU 耗时表与压力验证记录在 `caelum/doc/benchmark.md`。

## 4. 构建与测试放到远程

- **为什么**：两台开发机都跑不了 UVM，本地编出来的 GPU 代码没有可验证的出口；远端
  服务器的性能也远好于本机，编译不占本地节奏。
- **是什么**：一条通过 SSH 使用远程原生 Linux 服务器的通道，职责是四件事：连接服务器；
  同步代码与构建配置；在远端构建；在远端跑测试。执行体是 `.profile/remote/remote.sh`
  （本地）与 `remote-impl.sh`（远端，经 stdin 送入）两个脚本，用法与边界写在
  `.profile/remote/README.md`。
- **形态**：不是 skill。skill 只会把同样的文档再分发一次，执行体仍然是这两个脚本；
  需要按需加载时加一个只指向 README 的薄包装，不复制逻辑。
- **权限**：README 里有权限矩阵——能读什么、能写什么、能执行什么，脚本强制其中可检查的
  部分（远端根目录白名单、preset、参数形状、工作区干净、`build` 不是 upstream、文件权限），
  其余是约定；不是沙箱，绕开脚本就不再受它约束。

## 5. 接下来

1. ~~删除半自动路径~~：完成。
2. ~~运行时第一版~~：完成；`target-lower` 保留为 MLIR pass，契约写在 pass 描述里，
   不做设备选择。
3. ~~建远程构建与测试通道~~：完成，形态与权限见 §4。
4. ~~修 GPU 正确性~~：完成。归约、同步、动态形状（一维）都有用例；二维动态与非 identity
   layout 仍是明确拒绝的非目标。
5. ~~定算子集合与 benchmark~~：完成，见 §2.4 与 `caelum/doc/benchmark.md`。

此后的事情（不属于本次完成范围）：扁平图上的融合 pass、一个宿主函数里的多个骨架调用、
clang driver 的 `-fskeleton`、多 executor 与多机。

## 6. 未定问题

- CIR 要覆盖到什么程度才算够：现在是整个程序（`main`、控制流、标准库调用、`printf`、
  模板、递归都经 CIR→LLVM）；例外是 `std::cout` 这类依赖 `llvm.global_ctors` 的写法，
  driver 模式下不会自动初始化，v1 用 `printf`。
- 是否需要在扁平图上做融合 pass，以及融合的边界怎么和 op 单元对齐。
- 多机 / 多 executor 的部署形态，以及缓存失效与设备实例标识。

## 7. 相关文档

- [`operators.md`](operators.md)：算子这条线——组合方式、第一个真实算子、API 与验收。
- [`cpp_codegen.md`](cpp_codegen.md)：C++ 端到端——现状、缺口拆分、工作包与顺序。
- [`../doc/runtime.md`](../doc/runtime.md)：运行时的设计与实施记录。
- [`../doc/benchmark.md`](../doc/benchmark.md)：负载、算子矩阵、数值与耗时。
- [`../doc/finish-roadmap-decisions.md`](../doc/finish-roadmap-decisions.md)：完成 roadmap
  时的取舍记录。

`caelum/report/`、`caelum/review/` 里的旧文档只作历史记录；
其中半自动路径与识别算法的内容已经作废。

## 变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v0 | 2026-10-23 | 按口述顺序记录方向、GPU、运行时三段 |
| v1 | 2026-10-23 | 按目标 / 决定 / 现状 / 接下来 / 未定重组，删除过程性记录 |
| v2 | 2026-10-23 | 记录开发机跑不了 UVM、远端性能更好，构建与测试整体放到远程（含权限问题） |
| v3 | 2026-10-27 | GPU 桥接改为 UVM；记录远端 100 次重复与大缓冲验证结果 |
| v4 | 2026-10-28 | 定下运行时形态（§2.5）：driver 入口、op 级决策、运行期重编译、产物只带 CPU；旧 pass 待新路径覆盖后删除 |
| v5 | 2026-10-28 | 运行时第一版可用（M1–M4）：driver、按需编译、产物、GPU 重编译，远端全部测试通过；M5 清理待做 |
| v6 | 2026-10-28 | M5 完成：旧 pass 删除、e2e 全迁 driver；修掉 CPU reduce 的零初始化缺陷（远端 100 次 0 失败） |
| v7 | 2026-10-31 | 代码审查修复：每设备 JIT 引擎保活、resolve 失败终止、body-region lowering、编译失败负缓存、显存探测与放置回退、跨模块 ABI 校验、模块属性/global 随单元切分、产物 O2 + bytecode、多结果/空单元/entry 约束、文档与测试同步（单元 12/12、集成 20/20、CIR 9/9） |
| v8 | 2026-11-03 | 代码审查（站 1–3）：多结果限制收窄到 GPU 偏好单元、决策日志与 provider 契约、算子表指针失效、显存回退与零结果单元测试、嵌套图留在单个单元并诚实报错；算子这条线拆到 `operators.md` |
| v9 | 2026-11-07 | 完成 roadmap：半自动路径确认删除并写入现状（§2.1、§3.1）；annotation 语法定稿（§2.2）；算子集合由 benchmark 定下并补 matmul 的 C++ 入口（§2.4）；运行时补显式同步、运行时拥有的 UVM 分配、executor 标识（§2.5、§3.2）；GPU 归约可用、softmax 上设备、一维动态形状可用（§3.3）；远程通道的形态与权限模型定稿（§4）；`target-lower` 结论写入 §5.2 |
