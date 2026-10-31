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

- host 与所有 CPU impl 的机器码（AOT，经标准 O2 管线优化）；
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

多结果算子在这一步直接报错：host 的 dispatch 和 GPU 桥接都只处理一个结果。

## 5. 运行时 ABI

头文件 `mlir/include/mlir/Dialect/Skeleton/Runtime/SkeletonRuntime.h`，纯 C，不依赖 MLIR。

```c
enum { SKELETON_DEVICE_CPU = 0, SKELETON_DEVICE_GPU = 1 };

/* 声明一个 op 单元及其偏好；由产物/驱动在启动时调用。 */
void skeleton_rt_declare(int64_t op_id, int32_t preference, int64_t bytes);

/* 注册某设备的实现；重复注册覆盖（重编译后替换）。 */
void skeleton_rt_register(int64_t op_id, int32_t device, void *impl);

/* 解析点：O(1) 决策 + 返回 impl 指针；无法执行时打印原因并终止进程。 */
void *skeleton_rt_resolve(int64_t op_id);

/* 设备探测（带刷新间隔缓存）。返回 0 表示不可用。 */
int32_t skeleton_rt_probe(int32_t device);

/* 诊断与实验 */
void skeleton_rt_set_trace(int32_t enable);
void skeleton_rt_dump_stats(void);
```

约定：

- `resolve` 每次调用都做一次决策，但不一定重新探测（见 §8）。`resolve` 不做空指针返回：
  没有可用实现时打印 op/preference 和原因后 `abort()`。host 侧生成的代码因此可以直接调用
  返回值，不会在编译失败时执行到空函数指针。
- `bytes` 是算子 operands 的静态字节数下界（动态 shape 时为 0）；运行时用它和 GPU 的空闲
  显存比较，放不下就回退 CPU。它不是峰值工作集，也不传给 provider。
- host 侧用**带类型的间接调用**：`resolve` 返回 `!llvm.ptr`，host 用该 op 的真实签名调用它。
  不做 `void**` 参数打包——CPU 与 GPU impl 的签名本来就一样，打包只会让运行时去解释参数。
- impl 无返回值也要支持（GPU 目标函数可能返回 void，见 `b5a2a03`）。结果通过 out buffer。
- 运行时状态不是线程安全的：host 程序不能在多个线程里同时 resolve 算子。

可替换点：以后要支持别的设备，只增 device 枚举值与 impl 注册，ABI 不变。

## 6. op 单元的编译

输入：该 op 单元的 skeleton IR（一个函数）。
步骤：

1. `skeleton-to-linalg`（`map`/`reduce` 的 `pure_fn` 与 `body` 两种形式都支持；
   body 里的嵌套 skeleton 算子会在这个 pass 里报错）；
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
4. 校验单元函数 lower 后的 `LLVMFunctionType` 与 host 里对应声明完全一致，不一致直接报错
   （见 §16）；
5. JIT 出函数指针，`skeleton_rt_register(op_id, device, fn)`。

每个 `(op, device)` 的 `ExecutionEngine` 都保活：host 可能还持有另一个设备的函数指针。
编译失败按 `(op, device)` 缓存，后续调用不再重试（运行时回退到另一个设备）。

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
entry 必须是 `() -> ()`：driver 用 `invokePacked` 直接调用它，不解释参数。没有 op 单元的
host-only 程序也是合法输入，driver 会跳过单元表直接执行。

## 8. 探测与决策

- **probe 内容（v1）**：设备是否存在（`cuDeviceGetCount`）、显存余量（`cuMemGetInfo`）。
  没有 GPU 时不触碰 CUDA（`dlopen` 驱动 API，失败即视为不可用）。`cuMemGetInfo` 需要当前
  CUDA context：第一次 GPU 探测（还没分配过 GPU 内存）时读不到，运行时把空闲显存记为
  unknown 并跳过显存放不下时的回退；有了 context 之后的 probe 就能读到。
- **缓存**：probe 结果（可用性 + 空闲显存）带刷新间隔（默认 10ms，
  `SKELETON_PROBE_REFRESH_MS` 可调）。`resolve` 本身是 O(1) 查表，绝不在每次调用上做驱动 API。
- **策略（v1）**：preference 对应设备可用且放得下 operands 时用 preference，否则 CPU；
  两边都不可用时直接终止（不静默产出错误结果）。
- **日志**：`--trace` 或 `SKELETON_TRACE=1`，每行一次决策：
  `op=<id> pref=<GPU> probe=<GPU> free=1024MB choose=<CPU> reason=<gpu-unavailable>`
  外加编译事件：`op=<id> compile device=<CPU> impl=<ptr>`。driver 只有显式给 `--trace`
  时才调用 `set_trace(1)`，所以环境变量在 driver 模式下也生效。

## 9. 缓存与重编译

- 缓存键 `(op_id, device)`；进程内有效。
- 每次解析点重新决策；impl 已注册则直接用。
- 重编译只发生在缓存未命中；失败也按 `(op, device)` 缓存，避免每次调用都重复
  parse/lower/JIT。v1 没有 `--recompile=always` 入口。

## 10. 性能

三笔账，实现时要能量出来：

1. **启动**：host 的 JIT 编译（一次性）。codegen 级别用 LLJIT 默认（约 O2）。
   退路依次是：磁盘 object cache；产物内嵌预链接的 host 目标文件，driver 用 ORC 直接 link
   而不重新 codegen。产物自己运行没有这笔开销。
2. **代码质量**：JIT 侧用 LLJIT 默认（约 O2）；产物侧在 codegen 前跑标准 O2 管线，
   因此 `./prog` 与 `driver prog` 的优化级别一致。跨 op 边界不内联是设计使然。
3. **每次调用**：一次间接调用（纳秒级）+ 一次 O(1) `resolve`。probe 必须缓存。

## 11. 测试与验收

验收判据：

1. 远端：`mlir-skeleton-driver x.mlir` 跑通 `skeleton-matmul.mlir`（CPU），
   日志里有每个 op 的 resolve 与编译记录；
2. 产物 `prog` 单独运行（CPU）输出正确；
3. 远端 GPU：`mlir-skeleton-driver prog` 从产物 IR 重编 GPU impl 并执行，结果正确；
4. 同一进程内同一个 op 的两次调用可以拿到不同 impl（设备可用性变化后重新决策生效，
   并且切回原设备时旧的 JIT 引擎仍然有效）；
5. 原有测试在**新路径**上等价通过（见下）。

“不回归”在旧 pass 删除后的确切含义：

- 删除：`skeleton-preference-partition.mlir`、`skeleton-finalize-memref-to-gpu.mlir`
  （随 pass 一起删）；`skeleton-target-lower*.mlir` 保留，因为 `target-lower` 仍是
  driver 编译 GPU 单元的工具；
- 原样保留：`skeleton-ops-valid/invalid.mlir`、`skeleton-pattern-merge.mlir`、
  `skeleton-to-linalg.mlir`、clang 侧 CIR 用例；
- 改写：CPU/GPU 的 e2e 与 C++ → 运行的 e2e，输入与期望输出不变，RUN 行改走 driver。

删除清单（新路径覆盖后执行）：

- `SkeletonPreferencePartition.cpp`、`SkeletonFinalizeMemRefToGpu.cpp` 及其
  `Passes.td` / `Passes.h` 条目；
- 对应单元测试（`skeleton-preference-partition.mlir`、
  `skeleton-finalize-memref-to-gpu.mlir`）；
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

状态（2026-10-31）：

- M0 完成：本文与 roadmap 更新。
- M1 完成：`MLIRSkeletonRuntime`、`skeleton-outline`、`skeleton-lower-dispatch`。
- M2 完成：driver 读入、切分、host 降级 + JIT、按需编译 CPU 实现、决策日志。
- M3 完成：`--emit` 产物（AOT CPU 实现 + 内嵌 IR + 运行时 + 生成 `main`），
  产物可直接运行。
- M4 完成：driver 读产物 IR、按需重编 GPU 实现；GPU 单元走
  wrapper + `skeleton-target-lower` 的 UVM 桥接 + NVVM 管线。
- M5 完成：`preference-partition` 与 `finalize-memref-to-gpu` 及其测试已删除；
  CPU/GPU 全部 e2e 已迁到 driver（输入与期望输出不变）；`target-lower` 保留，
  作为 driver 编译单个 GPU 单元时的 lowering 工具，不再做放置决策。

2026-10-28 远端验证：单元测试 13/13、集成测试 16/16、CIR 9/9 全部通过。
2026-10-31 审查修复后：单元测试 12/12、集成测试 19/19、CIR 9/9 全部通过。
集成测试里新增：`driver-matmul`（driver 跑 CPU 闭环）、`driver-artifact`
（产物自运行 + driver 读产物 IR）、`driver-matmul-gpu`（GPU 重编译 + 产物三种用法）、
`driver-adaptive`（同一算子在同一进程内 GPU → CPU → GPU）、`driver-body`
（body-region 算子的完整 runtime 路径）、`driver-host-only`（没有 op 单元的 host 程序）、
`driver-multi-unit-gpu`（同一进程内两个 GPU 单元）。
单元测试新增 `skeleton-outline-invalid.mlir`（多结果拒绝）；`skeleton-to-linalg.mlir`
覆盖 body region 与嵌套 skeleton 算子的诊断。

2026-10-31 审查修复的内容见 §14 末尾的决策记录和 §15。

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
| 2026-10-28 | op id 的存放 | 写在内嵌 IR 的 outline 函数上 | 产物里的 CPU 实现与 driver 的重编号必须一致 |
| 2026-10-28 | outline 的幂等性 | 父函数带 `skeleton.op_id` 时跳过 | 产物内嵌的是已 outline 的模块，否则会套娃 |
| 2026-10-28 | 产物怎么链接 | 外部 clang + `-no-pie` | 目标机默认重定位模型不能按 PIE 链接 |
| 2026-10-28 | GPU 单元的形状 | wrapper（原名）+ 标记 GPU 的 impl | 直接复用 `skeleton-target-lower` 的 UVM 桥接与 lowering |
| 2026-10-28 | 编译缓存的实现位置 | 运行时表 + provider 回调 | 运行时管策略，driver 管编译，产物没有 provider |
| 2026-10-28 | GPU 单元的编译复用 | 保留 `skeleton-target-lower` 作为 lowering 工具 | 它不再做放置决策，只负责桥接与按目标降级；删掉它要先把逻辑搬进 driver |
| 2026-10-28 | 并行 loop 到 GPU 的映射 | `func.func(...)` 嵌套调度 | 这两个是函数级 pass，模块级调度会被拒绝 |
| 2026-10-28 | 旧 pass 的去留 | 删 `preference-partition` / `finalize`，留 `target-lower` | 前两个已无人调用；`target-lower` 仍是 driver 编译 GPU 单元的工具 |
| 2026-10-28 | reduce 的初值 | 元素类型零值 | 入口点没有初值参数，`tensor.empty` 会把结果变成分配器行为 |
| 2026-10-31 | 每个单元几个 JIT 引擎 | 每个 `(op, device)` 一个，全部保活 | host 可能还持有另一个设备的函数指针；销毁引擎会让它悬空 |
| 2026-10-31 | 编译失败怎么办 | 按 `(op, device)` 缓存失败，不重试 | 每次 resolve 都重跑 parse/lower/JIT 代价太大 |
| 2026-10-31 | resolve 无实现 | 打印原因并 abort | 生成代码直接调用返回值；返回 NULL 会变成空指针调用 |
| 2026-10-31 | 显存放不下 | 回退 CPU | `bytes` 下界已超过空闲显存时，GPU 分配必然失败 |
| 2026-10-31 | body region | 降到 linalg.generic / linalg.reduce；嵌套 skeleton 图拒绝 | 方言支持 body，runtime 不能把它留给 bufferization 才报错 |
| 2026-10-31 | 跨模块 ABI | JIT 前比较 lower 后的 `LLVMFunctionType` | host 与单元分别 bufferize，不能只靠约定 |
| 2026-10-31 | 产物优化级别 | 与 JIT 一致跑 O2 | 否则 `./prog` 和 `driver prog` 的性能对比混入优化差异 |
| 2026-10-31 | 内嵌 IR 的存活 | 用 MLIR bytecode + external linkage | O2 的 GlobalDCE 会删掉无引用的 internal global |
| 2026-10-31 | 多结果算子 | outline 阶段拒绝 | dispatch 和 GPU 桥接只处理一个结果，晚报错会静默丢结果 |

## 15. 修掉的问题

- **CPU reduce 从零开始**（2026-10-28 修复）：手动路径给 `skeleton.reduce` 的
  `init` 生成的是 `tensor.empty()`，而 `linalg.reduce` 会把 `outs` 的当前内容当作
  累加初值，于是结果取决于 malloc 是否恰好返回清零的页。表现为
  `cpp-reduce-sum.mlir` 约 10% 的运行输出 `[-3.07e+38]` 而不是 `[10]`。
  修法是给 reduce 生成元素类型零值的 `arith.constant`（`[cir]` 提交），
  并记录该入口点暂不提供自定义初值这一限制。修复后远端重复 100 次 0 失败。
  这与本节早先记录的 GPU 20% 垃圾输出是同一类“读未初始化内存”的症状，
  但根因不同、互相独立。
- **切回原设备时旧 JIT 代码已释放**（2026-10-31 修复）：每个单元只有一个
  `ExecutionEngine`，编译第二个设备时销毁第一个，`impls[device]` 里的函数指针悬空。
  `driver-adaptive` 的 GPU → CPU → GPU 路径会 SIGSEGV。改为每个 `(op, device)` 一个引擎
  并全部保活。
- **resolve 失败返回 NULL**（2026-10-31 修复）：编译失败时生成代码调用空指针。
  `resolve` 现在打印原因并 `abort()`。
- **body region 算子无法编译**（2026-10-31 修复）：`skeleton-to-linalg` 只处理 `pure_fn`，
  body 形式会走到 bufferization 才报 `op was not bufferized`。现在支持标量 body，
  并对嵌套 skeleton 图给出明确诊断。
- **`SKELETON_TRACE=1` 在 driver 里失效**（2026-10-31 修复）：driver 无条件调用
  `set_trace(0)`，把环境变量标成已读。现在只在 `--trace` 时调用 `set_trace(1)`。
- **编译失败没有负缓存**（2026-10-31 修复）：GPU 编译失败但 CPU 回退成功时，每次调用都
  重新编译 GPU。现在按 `(op, device)` 缓存失败。
- **产物没有优化、内嵌 IR 是文本**（2026-10-31 修复）：codegen 前跑标准 O2；内嵌 IR 改为
  MLIR bytecode，并用 external linkage 防止 GlobalDCE 删除。
- **单元切分丢模块属性和 global**（2026-10-31 修复）：单元模块现在复制 host 的模块属性，
  并把被引用的 global 一起克隆。
- **跨模块 ABI 没有校验**（2026-10-31 修复）：JIT 前比较单元与 host 声明 lower 后的
  `LLVMFunctionType`，不一致直接报错。
- **没有 op 单元的程序跑不了**（2026-10-31 修复）：`splitUnits` 和 `--emit` 都允许空单元表。
- **多结果算子在 GPU 路径静默丢结果**（2026-10-31 修复）：outline 阶段拒绝多结果单元，
  GPU 桥接也加了防御性检查。
- **entry 签名未检查、trace 日志与文档不一致**（2026-10-31 修复）：driver 与 `--emit`
  都要求 `() -> ()`；日志增加 `free=` 字段，文档同步。

## 16. 已知风险

- **跨模块 ABI 一致性**：host 与 op 单元分别 bufferize；driver 在 JIT 前比较两者 lower 后的
  `LLVMFunctionType`，不一致直接报错。类型相同但 data layout/非 identity layout 仍有风险，
  目前靠同一套 bufferize 配置和 GPU 侧的 dynamic/non-identity 检查来兜底。
- **JIT 实例数量**：每个 op 单元一个 `ExecutionEngine`；op 多时内存与启动成本要观察，
  必要时改成“一个 engine + 多模块”（需要 MLIR 暴露接口）。
- **运行期符号解析**：host 模块要能解析 `skeleton_rt_*`；用 `ExecutionEngine::registerSymbols`
  显式注册，不依赖可执行文件导出符号。
- **GPU 每 op 一个模块**：kernel 名冲突、常量 global、cubin 加载时机都要在实现里确认；
  `driver-multi-unit-gpu.mlir` 覆盖了两个 GPU 单元的同一进程执行。
- **无 GPU 机器**：driver 不能链接 libcuda；必须 `dlopen`，失败即视为不可用。
- **旧 pass 删除时的迁移**：先让新路径覆盖同样用例，再删，保持树可构建。
- **并发构建会损坏远程静态库**：一次只跑一个 `remote.sh`；中断后若
  `libLLVMAsmPrinter.a` 之类的归档损坏，删除该归档再重建。
- **线程安全**：运行时用进程级全局表，v1 不支持多线程同时 resolve；需要时再加锁。
- **平台**：产物链接固定用外部 `clang` + `-no-pie -ldl -lstdc++ -lm`，目前只针对
  Linux/libstdc++ 验证；换平台要重新确认链接参数和 C++ 运行库。
- **body region 的嵌套图**：`skeleton.map/reduce` 的 body 只支持标量算术，嵌套 skeleton
  算子会被 `skeleton-to-linalg` 拒绝；真正的嵌套计算图还没实现。
