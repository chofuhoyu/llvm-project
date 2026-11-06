# 完成 roadmap：差距分析与实施计划

> v1 · 2026-11-06
>
> 依据：`roadmap.md`（文件头仍写 v6，变更记录到 v8，2026-11-03）、工作区 HEAD
> `280621fe799d`、最近一次远端全绿 `b12704bd523d`。本文回答两件事：现在离 roadmap
> 还差什么，按什么顺序补齐。
> 运行时设计见 [`../doc/runtime.md`](../doc/runtime.md)，前端与算子见
> [`cpp_codegen.md`](cpp_codegen.md)、[`operators.md`](operators.md)。
> 三组 skeleton 测试的范围以 `.profile/remote/remote-impl.sh` 的过滤规则为准；
> 本文不记录用例个数。

## 1. 完成判据

“roadmap 完整实现”指以下三部分都满足：

1. roadmap §1–§6 的承诺；
2. roadmap 委派给子计划的验收：[`operators.md`](operators.md) §5（整程序 C++、CPU/GPU、
   driver/产物、数值、耗时）与 [`cpp_codegen.md`](cpp_codegen.md) §6.3 的 M1–M4；
3. 实现与文档一致：roadmap 的“现状/未定”不再落后于代码。

不在判据内：

- `caelum/report/` 里的三层架构与监控/配置面板——roadmap 没有这一层；
- [`cpp_codegen.md`](cpp_codegen.md) §6.3 M5 之后的可选扩张（一个 host 多个骨架算子、
  融合 pass、clang driver 的 `-fskeleton`）；
- [`../doc/runtime.md`](../doc/runtime.md) §1 列出的 v1 非目标（多机、多输出 GPU、
  非 identity layout、异常/RTTI、线程安全、热插拔等）。

这些不阻塞项在 §10 单列，避免和 roadmap 完成混在一起。

## 2. 基线

### 2.1 工作区

- 分支 `caelum/finish-roadmap-approach`，HEAD `280621fe799d`，工作区干净。
- 最近一次远端验证是 `b12704bd523d`（`.profile/remote/logs/run-20261105-202142.log`）：
  MLIR 单元、MLIR 集成、Clang CIR 三组通过。HEAD 相对它只多一个报告提交。
- 本机不编译、不测试（AGENTS 约定）；下面的“现状”来自代码阅读、文档和远端日志。

### 2.2 文档

| 文档 | 版本/日期 | 说明 |
|---|---|---|
| `plan/roadmap.md` | 文件头 v6 / 变更记录 v8 · 2026-11-03 | 方向与主计划；**落后于代码**（见 §3） |
| `doc/runtime.md` | 2026-10-28 + 决策记录 | 运行时设计基准；§14 记录到 2026-11-03 |
| `plan/cpp_codegen.md` | 2026-11-05 | C++ 前端 W0–W7 完成记录，§10 列出未做项 |
| `plan/operators.md` | 2026-11-05 | softmax、API、§5 验收状态 |
| `doc/cpp-codegen-operators-decisions.md` | 2026-11-05 | 前端与算子的取舍记录 |

### 2.3 代码现状

与本计划直接相关的部分：

- 半自动路径已删除：`CirLoopToSkeleton`、`Passes.td` 注册、循环注解形参、三组测试都不在树上。
  只有远端测试过滤器里还留着旧名字（§4 G9）。
- 手动路径可用：`skeleton.region` / `skeleton.pure` / `skeleton.op` 由
  [`CirSkeletonSplit.cpp`](../../clang/lib/CIR/Dialect/Transforms/CirSkeletonSplit.cpp)
  切成 `__skeleton_op_<id>` 单元和 ABI thunk；host 走 `cir-to-llvm`。
- 运行时可用：[`mlir-skeleton-driver.cpp`](../../mlir/tools/mlir-skeleton-driver/mlir-skeleton-driver.cpp)
  按 op resolve、从单元的 skeleton IR 按需编译、执行；`--emit` 产出只带 CPU 实现与内嵌 IR 的
  独立可执行文件；driver 可读回产物 IR 并为 GPU 重编译。
- 第一个真实算子 softmax 已完成 CPU 的扁平 DAG；GPU 版标 `XFAIL`，归约上不了设备。
- GPU 桥接仍是 `gpu.alloc host_shared` + host `memref.copy` 进/出 + `gpu.dealloc`；
  运行时没有分配 API，也没有显式的 kernel→host 同步。
- 运行时只有 device 枚举（CPU/GPU），没有 executor 对象/标识；trace 里没有 exec 字段。
- 远程通道是 `.profile/remote/remote.sh` + `remote-impl.sh`；不是可安装的 skill 包，
  但职责（连接、同步、构建、测试）都在。

## 3. roadmap 逐条对照

### 3.1 §1 目标与 §2.1–2.3

| roadmap 条目 | 要求 | 现状 | 判定 |
|---|---|---|---|
| §1 算子放置 | 用户显式写算子，运行时决定 CPU/GPU | driver 按 op resolve，编译期不再选设备 | 完成 |
| §2.1 手动路径 | 半自动路径直接删除，不留开关 | 代码与测试已删除；远端过滤器仍引用旧名字 | 完成，文档待更新 |
| §1 不识别算子 | 循环反推、形状猜测退出主线 | 无识别算法；形状来自参数类型或目的地类型 | 完成 |
| §2.2 annotation | 只表达“算子 + 放置目标” | `skeleton.region` 带 CPU/GPU，`skeleton.pure`、`skeleton.op`（map/reduce） | 完成；字段集固定，扩展见 WP5 |
| §2.3 放置决策 | 编译期 annotation 只是依据，运行期决策 | `skeleton_rt_resolve` 每次调用决策；旧 `preference-partition`/`finalize-memref-to-gpu` 已删 | 完成 |
| §2.3 `target-lower` | 是否进一步收进 driver 未定 | 仍作为 lowering 工具被 driver 调用 | 部分，见 G10 |
| §1 annotation 完整带下去 | 源码信息完整到 IR | 当前三个字段能带到 `skeleton.preference`/op 属性；未来算子参数没有 schema | 部分，见 WP5 |

### 3.2 §2.4 算子集合

- 方言只有 `skeleton.map`、`skeleton.reduce`、`skeleton.vector.add`；任意带
  `skeleton.preference` 的 linalg op 也能成为单元（`skeleton-matmul.mlir` 用手写 MLIR 验证）。
- C++ 入口只有 `skeleton_map` / `skeleton_reduce`；没有 matmul、layernorm 等 helper。
- softmax 是第一个真实算子，CPU 通过；benchmark 未定，算子清单未定。
- 判定：**未完成**（G7）。roadmap 的“等 benchmark 定下来再决定”仍是开放状态。

### 3.3 §2.5 运行时形态

| 要求 | 现状 | 判定 |
|---|---|---|
| 入口是独立 driver，拥有 skeleton IR 往下的整条管线 | `mlir-skeleton-driver` 自己 outline、切分、lower、JIT、执行 | 完成 |
| 决策粒度是单个算子 | `skeleton-outline` 一个 op 一个单元；每个调用点一次 resolve | 完成 |
| 运行期可以回到编译期 | `compileUnit` 从单元的 skeleton IR 重新 parse/lower/JIT | 完成 |
| 产物只含 host + CPU 实现 + 内嵌 IR，不含 GPU 代码 | `emitArtifact` 只注册 CPU impl；`.skeleton.ir` 存 bytecode | 完成 |
| 内存模型是 UVM，分配交给运行时后去掉 host 拷贝 | 桥接每次调用 `gpu.alloc host_shared` + `memref.copy` 进出；没有运行时分配 API | **未完成**，见 G2 |
| node → executor → device | 只有 device 枚举；executor 在代码里没有标识，trace 也没有 | 部分，见 G8 |

产物路径的一个细节：产物自己运行时只注册 CPU 实现；`driver prog` 从 `.skeleton.ir` 重读 IR，
按需编译 GPU 实现。这与“GPU 代码的唯一来源是重编译”一致。

### 3.4 §3.1–§3.3

- §3.1 说“半自动路径仍在树上，是接下来要删的第一样东西”——**已过期**。代码里没有
  `CirLoopToSkeleton`，测试也已删除。
- §3.2 的运行时第一版描述基本准确；唯一未收口的是 `target-lower` 的去留（G10）。
- §3.3 的 GPU 现状：
  - “静态形状、identity layout、单输出、只读 global 不写回”与代码一致；
  - UVM 已替换旧异步 `gpu.memcpy`，但 host 拷贝仍在（G2）；
  - `gpu.launch_func` 与随后的 host 读回之间没有显式同步（G3）；
  - 原 20% 失败没有复现，但没有可重复的更大负载/并发样本（G6）；
  - host `memcpy` 未消除，UVM 迁移开销没有基线（G2/G6）；
  - 归约上不了设备（G4）；动态形状被 `target-lower` 拒绝（G5）。

### 3.5 §4 远程构建与测试

| 要求 | 现状 | 判定 |
|---|---|---|
| 连接服务器、同步、远端构建、远端测试 | `remote.sh` / `remote-impl.sh`，四条职责齐全；只通过 git 同步 | 完成 |
| “做成一个 skill” | 实际是脚本 + README + AGENTS 说明，不是 skill 包 | 形式不一致，见 G1/G9 |
| 权限控制单独设计 | 脚本有根目录白名单、`umask 077`、不 sudo、不装软件；但没有权限模型文档 | 部分，见 G9 |
| 三组 skeleton 测试 | 远端过滤器仍匹配已删除的 `cir-call-to-skeleton` / `cir-loop-to-skeleton`，且漏掉 `cir-func-to-arith.cir` | **有验证缺口**，见 G9 |

`cir-func-to-arith.cir` 的 RUN 行驱动 `cir-skeleton-split`，内容是纯函数翻译器的正/负例；
它的文件名不含 `skeleton`，所以不在当前过滤器里。最近一次远端日志里也只见 split 的用例，
说明这份翻译器测试没有随合成测试一起跑。

### 3.6 §5 与 §6

| roadmap 条目 | 状态 | 差距 |
|---|---|---|
| §5.1 删除半自动路径 | 完成 | roadmap 未更新（G1） |
| §5.2 运行时第一版 | 完成，除 `target-lower` 去留 | G10 |
| §5.3 远程通道 | 完成 | 权限模型、形态、过滤器（G9） |
| §5.4 修 GPU 正确性 | 部分 | 排序（G3）、归约（G4）、动态形状（G5）、压力/性能（G6） |
| §5.5 算子集合与 benchmark | 部分 | 第一个算子 CPU 通过；GPU XFAIL；benchmark 未定（G7） |
| §6 annotation 语法 | 代码已定 | roadmap 仍写“未定”（G1） |
| §6 远程权限模型 | 未定 | G9 |
| §6 算子清单 | 未定 | G7 |
| 变更记录 | 停在 v8 | 缺 v9（G1） |

## 4. 差距清单

### G1 roadmap 与文档漂移

**表现**：

- §3.1、§5.1 仍把半自动路径写成待删；
- 文件头仍写 v6，变更记录已到 v8，两者不一致；
- §6 仍把 annotation 语法写成未定，而 `SkeletonOps.h` 与决策文档已经定死；
- §4 说“做成一个 skill”，实际是脚本；
- 变更记录停在 v8，没有记录 2026-11-05 的前端/算子工作；
- `.profile/caelum/AGENTS.md` 的测试范围仍引用已删除的 pass 名；
- 远端测试过滤器仍匹配已删除的名字，并漏掉 `cir-func-to-arith`。

**影响**：后续工作会按过期文本验收，或者重复已经完成的工作；测试范围与实际覆盖面不一致，
最危险的是“改了但没有跑”。

**收口**：WP0。

### G2 UVM 分配所有权与 host 拷贝

**roadmap 要求**（§2.5、§3.3）：分配交给运行时之后，现行桥接的 host 拷贝可以去掉。

**现状**：

- [`SkeletonTargetLower.cpp`](../../mlir/lib/Dialect/Skeleton/Transforms/SkeletonTargetLower.cpp)
  对每个 GPU call 的每个 memref operand 做 `gpu.alloc host_shared` + `memref.copy` 进，
  调用后 `memref.copy` 出 + `gpu.dealloc`；
- 运行时头文件只有 declare/register/resolve/probe/trace，没有 alloc/free 或分配器钩子；
- C++ host 的缓冲区是栈数组或 `std::vector`，不经过运行时，因而不可能在设备上直接访问；
- 所以“分配交给运行时”这一步从未发生，host 拷贝是桥接的固定成本。

**影响**：UVM 只是把设备分配换成 managed allocation，并没有消除拷贝；每个 memref operand
都要一次进、一次出，实时延与带宽都不符合 roadmap 的“先正确性、后性能”的下一步。

**收口**：WP1。核心是让 GPU 单元的缓冲区来自运行时分配的可被设备访问的内存，并在单元上
标记“managed”路径，让 `target-lower` 跳过桥接；未标记的模块继续走旧桥接，保证手写 MLIR
用例不回归。

### G3 GPU kernel 与 host 读回的排序

**现状**：

- GPU 单元被包成 `call @unit` 前后的 UVM alloc/copy；`call` 内部是 `gpu.launch_func`；
- `gpu-lower-to-nvvm-pipeline` 把 launch 降成 `mgpuLaunchKernel`，而
  [`CudaRuntimeWrappers.cpp`](../../mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp) 的
  `mgpuLaunchKernel` 只调用 `cuLaunchKernel`，不等待；
- 生成代码在 `call` 返回后立刻做 host `memref.copy` 读回；中间没有 `gpu.wait`、stream
  synchronize 或 event。

**影响**：host 可能在 kernel 写完之前读 UVM 缓冲区。小规模测试通过不能证明没有竞争；
这与旧的“异步拷贝与 kernel 从未互相排序”是同一类问题。`cuMemFree` 可能同步，但它在
copy-back 之后，救不了这次读。

**收口**：WP2。显式在 GPU 单元边界同步（运行时 `skeleton_rt_sync` 或等价的 stream 同步），
再用重复/大负载/`compute-sanitizer` 验证。

### G4 GPU 归约缺失

**现状**：

- `skeleton.target` 的 GPU 路径用 `convert-linalg-to-parallel-loops`；
- `linalg.reduce` 被降成带 `scf.reduce` 组合器的 `scf.parallel`；
- GPU 尾端把 `scf.parallel` 的迭代映射成 block/thread，组合器跨线程执行；
- `GPU/cpp-softmax.mlir` 因此 `XFAIL`，注释写明需要自己的设备归约 lowering。

**影响**：softmax 的 rowmax/rowsum 两个单元在 GPU 上不可用，softmax 整体不能算“GPU
可跑”；真实算子里的任何归约都会退回 CPU，op 级放置被削弱。

**收口**：WP3。先把 `linalg.reduce` 降成“输出并行 + 归约维串行”（与 matmul 的 reduction
iterator 同形），保证正确；再按 benchmark 需要做 block 级树形归约。不要继续依赖
`scf.reduce` 的跨线程语义。

### G5 动态形状 GPU

**现状**：

- [`SkeletonTargetLower.cpp`](../../mlir/lib/Dialect/Skeleton/Transforms/SkeletonTargetLower.cpp)
  显式拒绝 `getNumDynamicDims() > 0`；
- [`SkeletonLowerHostABI.cpp`](../../mlir/lib/Dialect/Skeleton/Transforms/SkeletonLowerHostABI.cpp)
  只支持“动态维是最后一维”，前端 `span` 也只产生一维动态；
- 因此当前只有静态数组能上 GPU；`span` 的 C++ 程序只能跑 CPU。

**影响**：真实 C++ 程序用 span 表达运行时长度；不能上 GPU 意味着 op 级放置在真实负载上
只剩静态玩具。子计划把这条记在 roadmap §5.4，roadmap 正文却没写清，属于需要收口的歧义。

**收口**：WP4。范围限定为前端能产生的形式（一维动态、最后一维）；二维动态、非 identity
layout 仍是 v1 非目标。若最终决定不做，必须从 `operators.md`/`cpp_codegen.md` 删掉这条，
并接受 GPU 只覆盖静态数组——本计划不建议这样做。

### G6 GPU 正确性证据与性能基线

**现状**：

- 旧路径与 UVM 的重复运行、大缓冲验证结果记录在 `roadmap.md` §3.3；
- 没有可重复的压力脚本，没有并发/争用样本，没有 `compute-sanitizer` 记录；
- softmax 的耗时、每算子 resolve/compile 时间、UVM 迁移开销都没有基线；
- `operators.md` §5 要求“记录耗时”，尚未做。

**影响**：无法证明“已修复”，也无法回答放置策略是否值得；G2/G3 的改动没有验收出口。

**收口**：WP2/WP6。压力矩阵、sanitizer、trace 统计、计时基线；结果写入
`operators.md`/benchmark 文档。本文不记录具体用例个数，脚本按过滤规则跑。

### G7 算子集合与 benchmark

**现状**：

- 方言只有 map/reduce/vector.add；C++ 入口只有 map/reduce；
- matmul 只能手写 MLIR（`skeleton-matmul.mlir`），C++ 侧没有 helper；
- softmax 是唯一真实算子，CPU 通过，GPU XFAIL；
- 没有 benchmark、没有算子矩阵、没有 CPU/GPU 耗时基线；
- 因此 roadmap §2.4 的“按 benchmark 扩展”和 §5.5 都停在“第一个算子”。

**影响**：算子放置没有真实负载检验；API 与放置策略可能被第一个 benchmark 推翻；
`operators.md` §5 的“CPU/GPU 各一次、记录耗时”没有完整证据。

**收口**：WP5。先定 benchmark 选择门槛，再做 harness 和算子矩阵；softmax GPU 属于 WP3，
benchmark 扩张属于 WP5，两者不要混在一个提交里。

### G8 executor 层不可见

**现状**：运行时只有 `SKELETON_DEVICE_CPU/GPU`；`RuntimeState` 是进程级全局表；
trace/stats 只有 op/pref/probe/free/choose/reason，没有 executor 标识。

**影响**：roadmap 的术语是 node → executor → device，并说“决策的落点是 executor 的
device”。v1 一台机器一个 executor，用进程隐式表示是合理的，但代码和日志里没有任何锚点，
以后加多 executor 时无法从现有 ABI 平滑过渡。

**收口**：WP6。不引入多 executor 行为；只在 trace/stats/文档里把“当前进程 = executor 0”
显式化，并保留 provider 按 executor 扩展的位置。若判断不值得改代码，至少在 roadmap v9
写明“v1 executor 就是驱动进程”，不要让术语悬空。

### G9 远程权限模型与测试过滤

**现状**：

- `remote-impl.sh` 有根目录白名单（`shared/yaozhiqi/llvm-project`）、`umask 077`、
  不 sudo、不装软件；`remote.sh` 检查分支、干净工作区、`build` 不作 upstream；
- 但这些是脚本里的约束，不是写下来的权限模型；操作者仍可绕过脚本直接 ssh 执行命令，
  这是策略层约定，不是强制沙箱；
- 测试过滤器仍是 `(skeleton|cir-call-to-skeleton|cir-loop-to-skeleton)`：
  - 两个旧名字已无对应测试；
  - `cir-func-to-arith.cir` 驱动 `cir-skeleton-split`，却因为文件名不含 `skeleton` 而漏跑；
  - 最近一次远端日志也印证了这一点。

**影响**：roadmap §4/§6 的权限问题没有答案；翻译器测试存在“改了但没跑”的风险。

**收口**：WP0 修过滤器与 AGENTS 测试范围；WP7 写权限模型并补可执行的边界检查。
skill 形态作为可选包装，不改变脚本行为。

### G10 `target-lower` 的去留

**现状**：`skeleton-target-lower` 仍是 `Passes.td` 里的 pass，driver 在编译 GPU 单元时调用；
它只按 `skeleton.target` 做 bufferize、UVM 桥接和 linalg→loop/parallel 的 lowering，
不再做放置决策。roadmap §5.2 仍把“是否进一步收进 driver”列为未决。

**影响**：不是功能缺口，但影响代码归属与文档一致性；如果一直悬着，后续 WP1/WP3/WP4 会继续
在“driver 里的 pipeline”和“独立 pass”之间分散逻辑。

**收口**：WP8。建议保留为 MLIR pass（driver 与单元测试共用），把契约写清楚：
输入是带 `skeleton.target` 的单元，输出是目标相关的机器码前置 IR；不做设备选择。
如果坚持收进 driver，需要把单元测试改成 driver 级测试，成本更高，收益只是少一个 pass。

## 5. 完成判据

| 编号 | 判据 | 验证 |
|---|---|---|
| E1 | 半自动路径无代码、无测试、无过滤器引用 | grep + 远端三组测试 |
| E2 | annotation 语法与字段写进文档并有正/负例 | `skeleton-split.cpp` / `-invalid.cpp` |
| E3 | 放置决策只在运行期；trace 能看到 op/pref/probe/free/choose/reason | `driver-*` 的 `--trace` 检查 |
| E4 | 产物只带 CPU 实现与内嵌 IR；driver 能读回并为 GPU 重编译 | `driver-artifact`、`driver-matmul-gpu` |
| E5 | GPU 程序的缓冲区来自运行时分配；managed 单元的 GPU 路径不再有 `gpu.alloc`/`memref.copy` | IR 检查 + CPU/GPU 数值 |
| E6 | GPU 单元边界有显式同步；压力/大负载/sanitizer 无错误 | WP2 的压力脚本与 sanitizer 记录 |
| E7 | GPU 归约可用；softmax GPU 变 PASS；`XFAIL` 删除 | `GPU/cpp-softmax.mlir` + GPU reduce 用例 |
| E8 | 一维动态形状能上 GPU；二维动态有明确诊断 | span GPU e2e + 负例 |
| E9 | benchmark 选定；算子矩阵、C++ 入口、CPU/GPU 数值与耗时齐备 | benchmark harness + `operators.md` 更新 |
| E10 | 远程权限模型写成文档；过滤器覆盖翻译器测试 | `.profile/remote/README.md` + 远端日志 |
| E11 | `target-lower` 去留已记录；roadmap v9 与代码一致 | roadmap 变更记录 |

## 6. 工作包

执行约定：llvm-project 的改动按 `[skeleton]` / `[cir]` / `[plan]` / `[build]` 分开提交；
`.profile/` 的改动提交到它自己的 `main`（`[build]`），不要混进 llvm-project。每个 WP 的
文档更新与代码更新可以分成独立提交，但同一件改动不要拆到无法单独理解的粒度。

### WP0 文档与验证口径收口（S）

**目标**：让 roadmap、子计划、AGENTS 与代码说同一件事；把“改了但没跑”的测试接进远端。

**改动**：

- `plan/roadmap.md` 升到 v9：
  - §3.1 改为“半自动路径已删除”；
  - §5.1 标完成；
  - §6 的 annotation 语法改为已定，并写清 `skeleton.region` / `skeleton.pure` /
    `skeleton.op` 三个字段；
  - §5.2 的 `target-lower` 去留按 WP8 的结论记录；
  - §4 的“skill”改为“脚本 + README；可选 skill 包装”；
  - 变更记录补 v9。
- `plan/cpp_codegen.md`、`plan/operators.md`、`doc/runtime.md`、
  `doc/cpp-codegen-operators-decisions.md` 中的状态与未做项同步。
- `.profile/caelum/AGENTS.md` 的测试范围改为当前过滤器，不再引用已删除 pass。
- `.profile/remote/remote-impl.sh` 的 CIR 过滤器改为覆盖 `skeleton` 与
  `cir-func-to-arith`，删除 `cir-call-to-skeleton` / `cir-loop-to-skeleton`。
- 首次把 `cir-func-to-arith.cir` 纳入远端并修掉暴露的问题。

**验收**：远端三组测试通过；`grep` 不到已删除 pass 的引用；roadmap 与子计划没有互相矛盾的
“已完成/未做”。

**依赖**：无，先做。`target-lower` 的结论可先按 WP8 的建议写入，后续再改。

### WP1 UVM 分配所有权与拷贝消除（L）

**目标**：把 GPU 单元使用的缓冲区交给运行时分配，使 host 与 kernel 看到同一块内存；
managed 路径不再有 per-call host 拷贝。

**设计要点**：

1. 运行时增加分配接口（名字待定，例如 `skeleton_rt_alloc` / `skeleton_rt_free`）：
   - CPU 走对齐 malloc；
   - GPU 走 `cuMemAllocManaged`，并在需要时建立/保留 CUDA context；
   - artifact 自行运行时没有 GPU 也能分配（退回 CPU 分配），不影响纯 CPU 运行。
2. 前端增加一个“由运行时分配”的缓冲区类型/视图（例如 `skeleton_detail::buffer`）：
   - 所有权明确，构造/析构对应 alloc/free；
   - 现有 `span` 可从它构造，CPU 代码继续用栈数组/`std::vector`；
   - GPU 偏好单元的数据参数必须能用这种类型表达，否则在编译期或运行期给出诊断。
3. 切分/ABI 标记：
   - 单元带 `skeleton.managed`（或等价属性），表示操作数来自运行时分配；
   - `skeleton-lower-host-abi` 仍然只构造 memref 描述符，不关心内存来源；
   - `skeleton-target-lower` 对 managed 单元跳过 `gpu.alloc`/`memref.copy` 桥接；
   - 未标记的模块（手写 MLIR、现有 GPU 测试）继续走旧桥接，保证不回归。
4. 失败路径：
   - 运行时分配失败要干净报错，不生成悬空指针；
   - GPU 偏好但缓冲区不是运行时分配时，要么编译期拒绝，要么运行期回退 CPU 并给出原因；
   - 不把“指针碰巧能访问”当作正确性。

**验收**：

- 一个 C++ GPU e2e 用运行时缓冲区跑出期望数值；IR 里 managed 单元没有
  `gpu.alloc`/`memref.copy`；
- CPU/artifact 路径数值不变；
- 分配/释放有测试，artifact 自运行不需要 libcuda；
- 文档写清“哪些类型能上 GPU、哪些只能 CPU”。

**风险**：ABI 变更、CUDA context 生命周期、内存对齐、managed 内存的同步（与 WP2 联动）。
先用“managed 标记 + 旧桥接回退”降低风险，不要一次性删掉桥接。

### WP2 GPU kernel/host 排序与正确性基线（M）

**目标**：消除 launch 与 host 读回之间的未排序窗口；建立可重复的正确性证据。

**改动**：

- 运行时增加同步入口（例如 `skeleton_rt_sync(int32_t device)`），或者用等价的 stream/event
  机制；
- GPU 单元 wrapper 在 kernel 之后、返回之前同步；桥接路径的 copy-back 与 managed 路径的
  host 读都因此有明确顺序；
- 补压力测试：重复运行、大缓冲、内存回退、adaptive、多单元、并发进程/争用；
- `compute-sanitizer` 记录；trace 统计 resolve/compile 次数与耗时。

**验收**：

- 压力矩阵零失败、sanitizer 无错误；
- 桥接路径和 managed 路径都有“kernel 完成后再读”的检查或测试；
- 形成一份可复现的耗时基线，写入算子/benchmark 文档。

**依赖**：可与 WP1 并行；managed 路径的最终验收在 WP1 之后。

### WP3 GPU 归约 lowering（L）

**目标**：让 `linalg.reduce` 在 GPU 上正确执行；softmax GPU 从 `XFAIL` 变 PASS。

**改动**：

- 在 `skeleton-target-lower` 的 GPU 分支里，先把 `linalg.reduce` 降成“输出并行 +
  归约维串行”的形式：每个输出元素一个线程，用 `scf.for` 扫归约维，标量组合；
- 不再让 `scf.reduce` 跨线程；
- 对 add/max 等常用组合器，后续可加 block 级树形归约作为优化，但第一版以正确为准；
- 增加 GPU reduce 的单元/集成用例；把 `GPU/cpp-softmax.mlir` 的 `XFAIL` 删除；
- softmax GPU 用例同时覆盖 driver 与 `driver prog`（产物路径）。

**验收**：

- GPU reduce 数值正确，无非法访存；
- softmax GPU 三行参考值通过；
- 归约单元不再退回 CPU；
- 负例（不支持的组合器/形状）有诊断。

**依赖**：WP2 的同步；与 WP4 可并行，但 softmax GPU 的动态版本在 WP4 之后。

### WP4 动态形状 GPU（M）

**目标**：让前端能产生的动态形状（一维 `span`、最后一维）在 GPU 上工作。

**改动**：

- `target-lower` 不再一刀切拒绝动态 memref；对 managed 路径允许动态维，按运行时尺寸
  计算 launch 的 grid/block；
- ABI thunk 继续支持“动态维是最后一维”；二维动态保持非目标，给出明确诊断；
- 动态单元的 `bytes` 为 0 时，显存回退策略要写清楚（无法预判就依赖分配失败/回退）；
- 增加 span-based GPU e2e；CPU/GPU 数值一致；
- 更新 `runtime.md` 的非目标与 `operators.md`/`cpp_codegen.md` 的 §5.4 说明。

**验收**：

- 一维动态 C++ 程序在 GPU 上跑出期望值；
- 二维动态有编译期诊断；
- 静态路径与桥接路径不回归。

**依赖**：最低验证依赖 WP1（managed 缓冲）；完整 softmax 动态版依赖 WP1 + WP3。

### WP5 算子集合与 benchmark（L）

**目标**：把 roadmap §2.4/§5.5 从“等 benchmark”推进到“benchmark 选定 + 算子矩阵 + 首批实现”。

**改动**：

1. benchmark 门槛：选一个可在单机 4090 上重复运行的工业级负载（候选：transformer 单层/
   单头 attention、layernorm、GEMM 组合；或一组 PyTorch/cuBLAS 风格微基准），要求有参考
   输出、固定输入、CPU/GPU 两种实现、可测耗时。
2. harness：C++ 程序 + 参考实现 + 数值容差 + 计时；driver 与产物、CPU 与 GPU 各跑一次。
3. 算子矩阵：列出 benchmark 需要的算子，标出 skeleton op / 纯函数 / host；决定哪些进方言、
   哪些走 linalg。
4. 首批实现：至少补 matmul 的 C++ 入口（现在只有手写 MLIR）和 layernorm 所需的 reduce/map
   组合；需要新方言 op 时，同时补 verifier、`skeleton-to-linalg`、annotation、前端 lowering。
5. 记录 CPU/GPU 耗时与放置日志，作为后续策略的基线。

**验收**：

- benchmark 程序全 C++，一条命令编出；driver/产物、CPU/GPU 数值正确；
- 算子矩阵与 roadmap §2.4 的“基本算子/skeleton 算子”划分一致；
- 有耗时表；新增算子有正/负例；
- `operators.md` §5 的验收项全部有证据。

**依赖**：WP3/WP4（GPU 真实算子）；harness 可在 M1 就开始，不必等 GPU。

### WP6 executor 可观测性（S）

**目标**：让 roadmap 的 executor 术语在 v1 有落点，但不引入多 executor 行为。

**改动**：

- trace/stats 增加 `exec=0`（或等价字段）；
- 文档写明“v1 一个进程 = 一个 executor；provider 仍按 (op, device)”；
- 如需 API，只加只读的 executor 计数/当前 id，不改 provider 签名；
- 多 executor 的 ABI 留给以后，不在本轮实现。

**验收**：日志字段与文档一致；现有 driver 测试不回归。

**依赖**：无，可与 WP7 并行。

### WP7 远程权限模型与形态（S/M）

**目标**：把 roadmap §4/§6 的权限问题写成可执行的边界；决定 skill 形态。

**改动**：

- 在 `.profile/remote/README.md` 增加权限矩阵：
  - 可访问目录：仅 `~/shared/yaozhiqi/llvm-project` 及其 `.profile`；
  - 可写：仓库、build 目录、日志；不写系统目录、不写其他用户目录；
  - 可执行：git、cmake、ninja、lit、CUDA 工具；不 sudo、不装软件、不改系统配置；
  - 数据：不复制仓库外文件；日志只落本地 `LOG_DIR`；
  - 审计：每次远端命令有本地日志；失败即停。
- 补脚本检查：根目录白名单已有；再加 `preset`/`BRANCH`/路径的显式校验（已有部分），
  以及“越界参数直接拒绝”的测试。
- 形态：保留脚本为执行体；如需要 skill，新增一个薄包装的 skill 文档，内容只指向
  `remote.sh`，不复制逻辑。
- 过滤器与 AGENTS 的修正并入 WP0；WP7 只负责权限与形态。

**验收**：越界 root/preset 被拒绝；文档覆盖“能访问/能写/能执行”；skill（若做）不改变行为。

**依赖**：无。

### WP8 `target-lower` 收口（S）

**目标**：结束 roadmap §5.2 的未决项。

**建议**：保留为 MLIR pass，理由是 driver 与单元测试共用、便于单测、职责已经是纯 lowering。
需要做的是：

- 文档写清契约：输入带 `skeleton.target`，输出目标相关 IR；不做设备选择；
- 把“UVM 桥接”与“managed 跳过桥接”都写进 pass 的 description（WP1 后更新）；
- 如果决定收进 driver，则把 `skeleton-target-lower*.mlir` 改成 driver 级测试，并删除
  pass 注册；这是一次更大的重构，只有在确实要减少公共 API 时才做。

**验收**：roadmap §5.2 标完成；pass 的注释/文档与行为一致；测试仍覆盖 CPU/GPU 两条路径。

**依赖**：WP1/WP3/WP4 的行为稳定后写最终结论；可以先记录“暂保留”。

### WP9 总验收与 roadmap v9（S）

**目标**：一次性验证“roadmap 完成”，并把文档定格。

**改动**：

- 跑完整矩阵：MLIR 单元、MLIR 集成 CPU/GPU、Clang CIR（含翻译器）、benchmark、压力；
- 检查 E1–E11；
- roadmap v9 的现状/接下来/未定与代码一致；
- 在 `operators.md` 记录耗时基线；在 `runtime.md` 记录 UVM/同步/动态形状的决策；
- 清理临时 TODO 与过期注释。

**验收**：三组测试 + benchmark + 压力全绿；文档无“已完成但仍写在待办”或相反的情况。

**依赖**：所有 WP。

## 7. 顺序与里程碑

| 里程碑 | 内容 | 出口条件 |
|---|---|---|
| M0 决策与口径 | WP0、WP8 | roadmap v9 草案；过滤器覆盖翻译器；`target-lower` 结论记录 |
| M1 内存与排序 | WP1、WP2 | managed 路径无 per-call 拷贝；GPU 读回有显式同步；压力基线 |
| M2 GPU 功能 | WP3、WP4 | softmax GPU PASS；GPU reduce；一维动态 GPU |
| M3 算子与 benchmark | WP5 | benchmark 选定；算子矩阵；CPU/GPU 数值与耗时 |
| M4 可观测与运维 | WP6、WP7 | exec 字段；权限矩阵；越界拒绝 |
| M5 总验收 | WP9 | E1–E11 全部满足；roadmap v9 定格 |

依赖关系：

- 关键路径：WP0 → WP1 → WP3 → WP5 → WP9。
- WP2 可与 WP1 并行；但 WP1 的 no-copy 验收依赖 WP2 的同步。
- WP4 的最低验证依赖 WP1（managed 缓冲）；完整 softmax 动态版还依赖 WP3（归约）。
- WP7、WP6 可与 M1–M3 并行。
- WP5 的 harness 可以在 M1 就开始，算子实现等 M2。
- 每次远端只跑一个 `remote.sh`；中断后先检查归档再重跑。

## 8. 验收与回归

### 8.1 测试矩阵

| 层 | 覆盖 | 出口 |
|---|---|---|
| MLIR 单元 | split/outline/target-lower/dispatch/to-linalg/verifier | 全部通过；新增 managed 与 GPU reduce 的单元用例 |
| MLIR 集成 CPU | driver/artifact、softmax、features、负例 | 两条路径数值一致 |
| MLIR 集成 GPU | map、softmax、reduce、动态、adaptive、memory fallback、multi-unit | 数值正确；无 `XFAIL` 残留（除明确记录的非目标） |
| Clang CIR | split、split-invalid、`cir-func-to-arith` | 翻译器测试真正被跑到 |
| benchmark | C++ 全程序、CPU/GPU、driver/产物 | 数值 + 耗时 |
| 压力 | 重复、大缓冲、并发/争用、sanitizer | 零失败、sanitizer 干净 |

### 8.2 命令与规则

- 常规：`bash .profile/remote/remote.sh run`；只重跑测试用 `test`（构建过期会拒绝）。
- 三组测试以脚本过滤规则为准，不在文档里记用例个数。
- 新增的 `cir-func-to-arith` 必须进入过滤器；删除的 pass 名不再出现在过滤器。
- 每个 WP 至少跑与该改动最相关的检查；报告里只列真正跑过的命令。
- GPU 用例在无 CUDA runner 的机器上必须是 `UNSUPPORTED` 视为失败，不是跳过（沿用 AGENTS）。

### 8.3 负例清单

- GPU 偏好但缓冲区不是运行时分配：编译期诊断或运行期回退 CPU，不能越界访问；
- 二维动态形状：编译期诊断；
- GPU reduce 的不支持组合器/形状：编译期诊断；
- 无 provider 的产物在 GPU 可用时：回退 CPU，不空指针；
- 显存放不下：回退 CPU；
- 单元签名与 host 不一致：编译期报错。

## 9. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| UVM 分配 ABI 与 CUDA context 复杂 | WP1 延期或 artifact 无 GPU 时崩 | 先做设计/原型；保留旧桥接回退；artifact 分配走 CPU 分支 |
| managed 内存的同步与生命周期 | 数值错误/释放后访问 | 与 WP2 一起做；显式同步；析构在最后一次使用之后 |
| GPU 归约 lowering 复杂 | softmax GPU 继续 XFAIL | 先做“输出并行 + 归约串行”的正确版，再做树形优化 |
| 动态形状范围蔓延 | 拖垮 M2 | 只做一维/最后一维；二维明确非目标 |
| benchmark 选择反复 | 算子清单迟迟不定 | 设决策门槛与候选；先做 harness，不绑定算子 |
| 远程权限只是策略约定 | 误操作风险 | 脚本白名单 + 文档矩阵 + 日志；不声称是沙箱 |
| 文档再次漂移 | 验收口径不一致 | WP0/WP9 两次收口；每次状态变化更新 roadmap |
| 测试过滤器再次漏测 | “改了没跑” | WP0 修过滤器；新增测试必须检查过滤器匹配 |
| 范围蔓延到非目标 | 计划失控 | §10 清单；新需求先改 roadmap |

## 10. 不阻塞 roadmap 的遗留项

| 项 | 为什么不在本轮 | 何时再看 |
|---|---|---|
| clang driver `-fskeleton` | `skeleton++` 已满足“一条命令”；roadmap 没要求改 driver | 工具链分发/集成时 |
| 一个 host 多个骨架算子 | 扁平 DAG 用多个 host 函数表达；v1 明确报错 | benchmark 证明写法受限时 |
| 扁平图融合 pass | 性能优化；当前每步落内存是可接受的 v1 | benchmark 显示跨算子开销是瓶颈时 |
| `std::cout` / 静态对象 / global ctors | JIT 不跑 `llvm.global_ctors`；v1 用 `printf` | 需要真实 C++ I/O/全局对象时 |
| 可恢复的错误处理 | `resolve` 失败 `exit(1)` 是运行时契约 | 需要嵌入宿主时 |
| 多 GPU / 热插拔 / 线程安全 | `runtime.md` §1 非目标 | 多机/多实例阶段 |
| 多输出 GPU / 非 identity layout / 二维动态 | v1 非目标 | 算子集合确实需要时 |
| 监控面板 / 策略配置 UI | 报告里的第三层，roadmap 没有 | 日志与放置策略稳定后 |
| benchmark 之外的算子扩张 | 等 WP5 的矩阵 | WP5 之后 |

## 11. 需要写进 roadmap v9 的决策

| 编号 | 决策 | 建议 |
|---|---|---|
| D1 | 动态形状 GPU 是否在 roadmap 完成范围内 | 在；范围限一维/最后一维；二维仍非目标 |
| D2 | executor 在 v1 的形态 | 进程即 executor 0；只加可观测字段，不做多 executor |
| D3 | 远程交付形态 | 脚本 + 文档；skill 只做薄包装，不复制逻辑 |
| D4 | `target-lower` 去留 | 保留为 MLIR pass，契约写清；不收进 driver |
| D5 | GPU 缓冲区契约 | GPU 偏好单元使用运行时分配的 buffer；未满足时诊断/回退，不静默 |
| D6 | benchmark 选择门槛 | 先定门槛与 harness，再定算子清单；不在 roadmap 里预设具体算子 |

这六条决定不改变 roadmap 的方向，只把已经发生的实现、需要收口的歧义写进基准文档。
它们应在 WP0 写入，WP9 复核。

## 12. 变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v1 | 2026-11-06 | 首次生成：roadmap（变更记录 v8）与工作区现状的逐条对照、差距 G1–G10、WP0–WP9、E1–E11、风险与不阻塞项 |

## 附录 A：roadmap 条目与证据

| roadmap | 代码/测试证据 |
|---|---|
| §2.1 手动路径 | `clang/lib/CIR/Dialect/Transforms/CirSkeletonSplit.cpp`；`clang/test/CIR/Transforms/skeleton-split*.cpp`；无 `CirLoopToSkeleton` |
| §2.2 annotation | `clang/lib/CIR/Dialect/Transforms/CirSkeletonAnnotations.cpp`；`mlir/include/mlir/Dialect/Skeleton/SkeletonOps.h` |
| §2.3 运行时决策 | `mlir/lib/Dialect/Skeleton/Runtime/SkeletonRuntime.cpp`；`mlir/tools/mlir-skeleton-driver/mlir-skeleton-driver.cpp` |
| §2.5 产物 | driver 的 `emitArtifact` / `addSkeletonIR` / `loadProgram`；`driver-artifact`、`driver-matmul-gpu` |
| §2.5 UVM | `mlir/lib/Dialect/Skeleton/Transforms/SkeletonTargetLower.cpp`（`gpu.alloc host_shared` + `memref.copy`） |
| §3.3 GPU 正确性 | `mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp`（`mgpuLaunchKernel` 不同步）；`GPU/cpp-softmax.mlir`（XFAIL） |
| §4 远程 | `.profile/remote/remote.sh`、`remote-impl.sh`、`README.md`；远端日志 |
| §5.5 / operators §5 | `mlir/test/Integration/Dialect/Skeleton/CPU/cpp-softmax.mlir`、`GPU/cpp-softmax.mlir`、`CPU/cpp-features.mlir` |
| 算子集合 | `mlir/include/mlir/Dialect/Skeleton/IR/SkeletonOps.td`；`mlir/test/Integration/Dialect/Skeleton/CPU/skeleton-matmul.mlir` |
