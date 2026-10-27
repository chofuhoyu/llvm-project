# Skeleton 最小运行时：设计与实施记录

> 2026-10-28 · 分支 `caelum/skeleton-minimal-runtime`
>
> 本文是这条工作线的**基准文档**：架构、ABI、产物格式、测试判据、已否决的方案都在这里。
> 实现过程中每次需要权衡，都追加到 §14 决策记录，并在对应提交信息里说明。
> 与 `caelum/review/`、`caelum/report/` 冲突时以本文和 `caelum/plan/roadmap.md` 为准。

## 1. 目标与非目标

目标：

- **算子级运行期放置**：每个算子单独决策，同一个程序里不同算子可以落在不同设备上。
- **driver 是系统入口**：它拥有从 skeleton IR 往下的整条编译管线，并在自己的进程里执行程序。
- **运行期可以回到编译期**：执行到某个算子时，按当时的设备状态决定设备，必要时从该算子的
  skeleton IR 重新编译，而不是在编译期就把设备定死。
- **产物自带运行时**：生成的程序是独立可执行文件，离线也能跑；有 driver 时解锁重编译。
- **单机 v1**：一台机器、一个 executor、CPU + 可选 GPU；接口为以后的多设备、多机留好位置，
  但 v1 不实现分布式。

非目标（v1 明确不做）：

- 分布式、跨机数据搬运、多个 executor 同时存在；
- 子进程 / 远程执行、把 host 放到 driver 以外的机器上；
- 动态形状、多输出、非 identity layout；
- 学习型代价模型、异步流水、算子切分；
- 产物里预先放 GPU 代码（PTX/ cubin）——见 §13；
- 半自动识别路径（按 roadmap §2.1 另行删除）。

## 2. 术语

| 词 | 含义 |
|---|---|
| node | 一台机器。v1 只有一台。 |
| executor | 执行器。逻辑执行单元，拥有 0..n 个设备，负责编译与执行算子实现；放置决策的落点。v1 一个进程一个 executor。 |
| device | 设备：CPU / GPU（以后还有别的）。executor 内部的叶子。 |
| op 单元 | 一个可以独立决策、独立编译、独立替换的算子实例。 |
| host | 程序里除 op 单元以外的部分：控制流、分配、调用、打印。 |
| impl（实现） | 某个 op 单元在某类设备上的可调用代码。CPU impl 与 GPU impl 签名相同。 |
| 解析点 | host 里调用 op 单元的位置；运行期在这里决定这次调用用哪个 impl。 |
| 产物 | 可独立运行的可执行文件：host + CPU impl + 内嵌 skeleton IR + 运行时。 |

不使用 `agent` 一词。跨进程/跨机器的部署形态以后再命名。

## 3. 架构

```
输入（二选一）
  a. skeleton 层 .mlir
  b. 产物（ELF，含 .skeleton.ir section）

driver（mlir-skeleton-driver）
  1. 读入模块，按 §4 分类：host / op 单元
  2. host：每个算子位置换成一个解析点 → 降级到 LLVM → JIT
  3. 执行 entry：每次到解析点
        probe（缓存刷新）→ 决策（executor+device）→ 查缓存
          ├ 命中：取 impl 指针，间接调用
          └ 未命中：从该 op 单元的 skeleton IR 编译该设备的 impl → 注册 → 间接调用
  4. 记录每次决策与编译耗时（供实验）

产物（--emit）
  prog = host(CPU, AOT) + 每个 op 的 CPU impl(AOT) + 内嵌 skeleton IR + 运行时
  ./prog         → 运行时表里只有 CPU impl，纯 CPU 运行
  driver prog    → 读内嵌 IR，重编译 GPU impl，注册后执行
```

关键性质：

- GPU 代码的唯一来源是运行期重编译。产物里没有 GPU 代码，所以“重编译”不是兜底，
  而是获取 GPU 能力的唯一路径。
- 决策发生在每次解析点，且 probe 结果带刷新间隔；同一进程内设备状态变化后，
  同一个 op 的后续调用可以拿到不同 impl。这是“决策真在运行期”的可验证证据。

## 4. 输入与产物

### 4.1 driver 需要知道什么

driver 只需要三样东西就能完整描述程序：

1. **entry 符号**（如 `main`）；
2. **host 代码**：除 op 单元以外的全部内容；
3. **算子表**：每个 op 单元的 (id, skeleton IR, 边界类型, preference)。

设备、目标、网格大小一概不需要：都是运行期的事。

### 4.2 分类规则

一个 op 是 op 单元，当且仅当：

- 它是 skeleton 方言算子（`skeleton.map` / `skeleton.reduce` / `skeleton.vector.add`），或
- 它是带 `skeleton.preference` 的 linalg 算子。

preference 缺失按 CPU 处理。其余一律是 host。

> 这一条按 roadmap §1 的“把每个算子分配到 CPU / GPU 上执行”执行：所有骨架算子都成为
> op 单元，不是只挑带 preference 的。

### 4.3 产物格式

单文件 ELF 可执行文件，包含：

- host 与所有 CPU impl 的机器码（AOT）；
- 自定义 section `.skeleton.ir`：op 单元已编号的 skeleton 层模块，MLIR bytecode；
- 静态链接的运行时（`MLIRSkeletonRuntime`）。

**op id 必须在构建产物时就写进内嵌 IR**（`skeleton.op_id = <n>` 属性）。原因是产物里已经
链好了 CPU impl，driver 重读 IR 时必须得到同一组 id，否则注册对错位置。不依赖遍历顺序。

driver 读 IR 用 LLVM 的 object 库按 section 名取内容，不 dlopen、不起子进程
（Linux 上 PIE 也不能 dlopen）。

### 4.4 op 单元的抽取

`skeleton-outline` pass：

- 对每个 op 单元创建 `func.func @__skeleton_op_<id>(<tensor operands>) -> <tensor result>`，
  函数体是那个算子（保留 skeleton 形态），函数上带 `skeleton.op_id`；
- 原位置替换成对该函数的调用；
- 输出模块里同时有 host 和 outline 出来的函数，driver 再把它们切成
  “host 模块 + 每个 op 一个模块”。

## 5. 运行时 ABI

头文件 `mlir/include/mlir/Dialect/Skeleton/Runtime/SkeletonRuntime.h`，纯 C，不依赖 MLIR。

```c
enum { SKELETON_DEVICE_CPU = 0, SKELETON_DEVICE_GPU = 1 };

/* 声明一个 op 单元及其偏好；由产物/驱动在启动时调用。 */
void skeleton_rt_declare(int64_t op_id, int32_t preference, int64_t bytes);

/* 注册某设备的实现；重复注册覆盖（重编译后替换）。 */
void skeleton_rt_register(int64_t op_id, int32_t device, void *impl);

/* 解析点：O(1) 决策 + 返回 impl 指针；未注册任何 impl 时返回 NULL。 */
void *skeleton_rt_resolve(int64_t op_id);

/* 设备探测（带刷新间隔缓存）。返回 0 表示不可用。 */
int32_t skeleton_rt_probe(int32_t device);

/* 诊断与实验 */
void skeleton_rt_set_trace(int32_t enable);
void skeleton_rt_dump_stats(void);
```

约定：

- `resolve` 每次调用都做一次决策，但不一定重新探测（见 §8）。
- host 侧用**带类型的间接调用**：`resolve` 返回 `!llvm.ptr`，host 用该 op 的真实签名调用它。
  不做 `void**` 参数打包——CPU 与 GPU impl 的签名本来就一样，打包只会让运行时去解释参数。
- impl 无返回值也要支持（GPU 目标函数可能返回 void，见 `b5a2a03`）。结果通过 out buffer。

可替换点：以后要支持别的设备，只增 device 枚举值与 impl 注册，ABI 不变。

## 6. op 单元的编译

输入：该 op 单元的 skeleton IR（一个函数）。
步骤：

1. `skeleton-to-linalg`；
2. `one-shot-bufferize`（`bufferizeFunctionBoundaries=true`、identity layout）——
   与 host 侧同一套配置，保证跨模块 ABI 一致；
3. 按目标设备 lower：
   - **CPU**：`convert-linalg-to-loops`、`convert-scf-to-cf`、`convert-arith-to-llvm`、
     `convert-func-to-llvm`、`convert-cf-to-llvm`、`finalize-memref-to-llvm`、
     `reconcile-unrealized-casts`；
   - **GPU**：`convert-linalg-to-parallel-loops` → 在该函数内生成 UVM 包装
     （alloc host_shared → host `memref.copy` 进 → kernel → host `memref.copy` 出 → dealloc）
     → `convert-parallel-loops-to-gpu` → `gpu-kernel-outlining` →
     `gpu-lower-to-nvvm-pipeline` → host 侧 LLVM lower。
4. JIT 出函数指针，`skeleton_rt_register(op_id, device, fn)`。

GPU 包装放在 op 单元内部，而不是调用点：host 必须与设备无关。等运行时接管分配
（分配器钩子）之后，这层包装和 host 拷贝会被去掉。

## 7. host 侧

1. `skeleton-outline`；
2. host 模块走原有 lower 管线：`convert-linalg-to-loops`（host 里剩下的 linalg）、
   bufferize、arith/func/cf/memref → LLVM、`reconcile-unrealized-casts`；
3. 把对 `@__skeleton_op_<id>` 的直接 `llvm.call` 改写成
   `%impl = llvm.call @skeleton_rt_resolve(op_id)` + 带类型的间接 `llvm.call`；
4. `ExecutionEngine` JIT host，调用 entry。

host 在 driver 模式是 JIT 的；在产物模式是 AOT 的。两条路共用同一套 pass，只有最后一步不同。

## 8. 探测与决策

- **probe 内容（v1）**：设备是否存在（`cuDeviceGetCount`）、显存余量（`cuMemGetInfo`）。
  没有 GPU 时不触碰 CUDA（`dlopen` 驱动 API，失败即视为不可用）。
- **缓存**：probe 结果带刷新间隔（默认 10ms，`SKELETON_PROBE_REFRESH_MS` 可调）。
  `resolve` 本身是 O(1) 查表，绝不在每次调用上做驱动 API。
- **策略（v1）**：preference 对应设备可用则用 preference，否则 CPU；两边都不可用时直接失败
  （不静默产出错误结果）。
- **日志**：`--trace` 或 `SKELETON_TRACE=1`，每行一次决策：
  `op=<id> pref=<GPU> probe=<GPU:free=1024MB> choose=<CPU> reason=<gpu-unavailable>`
  外加编译事件：`op=<id> compile device=<CPU> us=<n> cache=<miss|hit>`。

## 9. 缓存与重编译

- 缓存键 `(op_id, device)`；进程内有效。
- 每次解析点重新决策；impl 已注册则直接用。
- 重编译只发生在缓存未命中；`--recompile=always` 留给实验用（强制每次重编，测编译开销）。

## 10. 性能

三笔账，实现时要能量出来：

1. **启动**：host 的 JIT 编译（一次性）。codegen 级别用 LLJIT 默认（约 O2）。
   退路依次是：磁盘 object cache；产物内嵌预链接的 host 目标文件，driver 用 ORC 直接 link
   而不重新 codegen。产物自己运行没有这笔开销。
2. **代码质量**：同一套管线，O2 下与 AOT 同级；跨 op 边界不内联是设计使然。
3. **每次调用**：一次间接调用（纳秒级）+ 一次 O(1) `resolve`。probe 必须缓存。

## 11. 测试与验收

验收判据：

1. 远端：`mlir-skeleton-driver x.mlir` 跑通 `skeleton-matmul.mlir`（CPU），
   日志里有每个 op 的 resolve 与编译记录；
2. 产物 `prog` 单独运行（CPU）输出正确；
3. 远端 GPU：`mlir-skeleton-driver prog` 从产物 IR 重编 GPU impl 并执行，结果正确；
4. 同一进程内同一个 op 的两次调用可以拿到不同 impl（设备可用性变化后重新决策生效）；
5. 原有测试在**新路径**上等价通过（见下）。

“不回归”在旧 pass 删除后的确切含义：

- 删除：`skeleton-preference-partition.mlir`、`skeleton-target-lower*.mlir`、
  `skeleton-finalize-memref-to-gpu.mlir`（随 pass 一起删）；
- 原样保留：`skeleton-ops-valid/invalid.mlir`、`skeleton-pattern-merge.mlir`、
  `skeleton-to-linalg.mlir`、clang 侧 CIR 用例；
- 改写：CPU/GPU 的 e2e 与 C++ → 运行的 e2e，输入与期望输出不变，RUN 行改走 driver。

删除清单（新路径覆盖后执行）：

- `SkeletonPreferencePartition.cpp`、`SkeletonTargetLower.cpp`、
  `SkeletonFinalizeMemRefToGpu.cpp` 及其 `Passes.td` / `Passes.h` 条目；
- 对应单元测试；
- 上面的 e2e 迁移。

## 12. 里程碑与提交

| 里程碑 | 内容 | 提交 scope |
|---|---|---|
| M0 | 本文 + roadmap 更新 | `[plan]` |
| M1 | 运行时库与 ABI、CMake、`skeleton-outline` pass、单元测试 | `[skeleton]` |
| M2 | driver：读入、切分、host lower+JIT、按需编译 CPU impl、决策日志 | `[skeleton]` |
| M3 | `--emit` 产物：内嵌 IR、AOT CPU impl、链接、自运行测试 | `[skeleton]` |
| M4 | driver 读产物 IR、重编 GPU impl、远端验证 | `[skeleton]` |
| M5 | 删除旧 pass 与其测试、迁移 e2e、更新 roadmap 现状 | `[skeleton]` + `[plan]` |

每个里程碑都要能构建、能解释测试结果。只 commit，不 amend、不 rebase、不 force push。

## 13. 已否决的方案

- **产物里带 PTX / cubin**：把“生成什么代码”在构建期固化，运行期只剩选变体，
  与“从 skeleton IR 重编译”的主张矛盾。以后只作为论文的静态基线开关。
- **编译期生成双变体 + 运行期选**：同上，决策退化为选择；且无 GPU 机器上的加载问题难解。
- **`agent` 术语**：行业里指部署在节点上的运维组件，不是节点本身。统一用 executor；
  跨进程形态以后再命名。
- **子进程 / driver 监督目标进程**：换实现要求被换的代码配合，进程内最短路径；
  子进程只是把回调机制拆成两个进程。v1 不做。
- **driver dlopen 产物**：Linux 不允许 dlopen PIE；产物做成单文件 ELF 并由 driver 读 IR，
  避开这个问题。
- **运行时拥有调用（`void**` 参数打包）**：CPU/GPU 签名相同，打包只会让运行时解释参数；
  用带类型的间接调用。
- **进程内 lld 链接产物**：crt/动态链接器配置复杂；v1 调外部 `clang`
  （lit 环境下 `clang` 已在 PATH，现有 e2e 也这么用）。

## 14. 决策记录

| 日期 | 背景 | 选择 | 理由 |
|---|---|---|---|
| 2026-10-28 | 运行时形态 | driver 为入口 + op 级按需编译 | 让放置决策有承载者；op 级对应 roadmap §1 |
| 2026-10-28 | 产物是否带 GPU 代码 | 不带 | 重编译必须是 GPU 代码的唯一来源，否则主张不成立 |
| 2026-10-28 | driver 如何运行产物 | 读内嵌 IR，进程内 JIT | 单文件产物 + 无子进程 + 避开 PIE dlopen |
| 2026-10-28 | 内存模型 | UVM | 已有桥接即为 UVM；以后分配器钩子去掉拷贝 |
| 2026-10-28 | 探测信号 | 设备存在 + 显存余量 | 最基础、无需 NVML；争用曲线时再升级 |
| 2026-10-28 | 间接调用形态 | 带类型的间接 `llvm.call` | 签名天然一致，运行时不解释参数 |

## 15. 已知风险

- **跨模块 ABI 一致性**：host 与 op 单元分别 bufferize，memref 类型必须逐位一致；
  注册时校验签名，不一致直接报错。
- **JIT 实例数量**：每个 op 单元一个 `ExecutionEngine`；op 多时内存与启动成本要观察，
  必要时改成“一个 engine + 多模块”（需要 MLIR 暴露接口）。
- **运行期符号解析**：host 模块要能解析 `skeleton_rt_*`；用 `ExecutionEngine::registerSymbols`
  显式注册，不依赖可执行文件导出符号。
- **GPU 每 op 一个模块**：kernel 名冲突、常量 global、cubin 加载时机都要在实现里确认。
- **无 GPU 机器**：driver 不能链接 libcuda；必须 `dlopen`，失败即视为不可用。
- **旧 pass 删除时的迁移**：先让新路径覆盖同样用例，再删，保持树可构建。
