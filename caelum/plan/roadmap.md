# Caelum 计划

> v6 · 2026-10-28
>
> 这份文档记录方向、决定与接下来要做的事。没写进来的部分都还没定。
> 运行时的设计细节在 `caelum/doc/runtime.md`。

## 1. 目标

重心是**运行时算子放置**（operator placement）：用户显式写出骨架算子，
放置决策在运行时做出，把每个算子分配到 CPU / GPU 上执行。

编译器不再识别算子，只做两件事：把用户写下的算子落成 IR，
把源码里的 annotation 完整地带下去。识别相关的机制（循环反推、形状猜测）退出主线。

## 2. 设计决定

### 2.1 前端只保留手动路径

- 半自动路径（从带注解的循环反推算子）**直接删除**，不设过渡期、不保留开关。
- 删除范围：
  - `CirLoopToSkeleton` 及 `Passes.td` 里的注册；
  - `clang/test/CIR/Transforms/cir-loop-to-skeleton.cir`、`skeleton-loop-path.cpp`；
  - `mlir/test/Integration/Dialect/Skeleton/CPU/loop-map-add.*`；
  - `CIRGenStmt.cpp` 中仅为循环注解服务的 `attachLoopAnnotations`，以及
    `emitForStmt` / `emitWhileStmt` / `emitDoStmt` 为传 attr 新增的形参（回到上游形态）。
- 工作流全部走手动路径：用户显式给出算子，算子上的 annotation 表达放置意图。

### 2.2 annotation：只表达算子与放置目标

- 职责收窄为「算子 + 放置目标」，不再承担「这里是什么算子」的识别线索。
- 第一版取最简：语法不限（编译器是自己的），但要求最简单、最不易出错；
  首要目标是把源码里的信息完整传到 IR，表达力以后再说。

### 2.3 放置决策在运行时

- 编译期 annotation 是决策依据之一，编译期不做「选设备」。
- `skeleton-preference-partition` 与 `skeleton-target-lower` 现在承担的编译期选择语义，
  预计由运行时取代——它们正是「只在编译期做功」的产物。

### 2.4 算子集合按 benchmark 扩展

- 需要扩展，但清单不在此刻拍：等工业级 benchmark 定下来，
  再决定基本算子与 skeleton 算子各支持哪些；可参考 cuBLAS / PyTorch 的算子库划分。

### 2.5 运行时的形态

- **入口是独立 driver**（`mlir-skeleton-driver`），拥有从 skeleton IR 往下的整条编译管线，
  并在自己的进程里执行程序。它既不是 MLIR pass，也不是构建脚本。
- **决策粒度是单个算子**。编译时把算子切成 op 单元；host 里每个算子位置是一个运行期
  解析点，由运行时决定这次调用落到哪个 executor 的哪个 device 上。
- **运行期可以回到编译期**：执行到某个 op 时，若缓存里没有可用实现，就从该 op 的
  skeleton IR 重新跑 skeleton → linalg → bufferize → 目标 lowering → JIT。
- **产物是独立可执行文件**：只含 host 与各算子的 CPU 实现、内嵌 skeleton IR、运行时，
  不含 GPU 代码。`./prog` 自己运行是纯 CPU；`driver prog` 读产物里的 IR、按需重编译出
  GPU 实现并注册后执行——GPU 代码的唯一来源是重编译。
- **内存模型是 UVM**：host 与 kernel 看到同一块内存。分配交给运行时之后，
  现行桥接的 host 拷贝可以去掉（后续步骤）。
- **术语**：node（机器）→ executor（执行器，放置决策的落点）→ device（设备）。
  第一版一台机器一个 executor；跨进程、跨机器的部署形态以后再命名。

## 3. 现状

### 3.1 编译器

手动路径可用：C++ 源码 → CIR → skeleton → linalg → CPU/GPU lowering，两条路径各有 e2e。
半自动路径仍在树上，是接下来要删的第一样东西。

### 3.2 运行时

**第一版可用**（`caelum/doc/runtime.md`）。`mlir-skeleton-driver` 按 op 决策、
按需从该 op 的 skeleton IR 重编译、在自己的进程里执行；`--emit` 产出只带 CPU 实现
与内嵌 IR 的独立可执行文件，driver 可读回其 IR 并为 GPU 重编译。远端验证：
单元 12/12、集成 20/20、CIR 9/9。2026-10-31 的审查修复补齐了按设备保活的 JIT 引擎、
resolve 失败终止、body-region 算子、编译失败负缓存、GPU 显存探测、跨模块 ABI 校验、
产物 O2 与 bytecode 内嵌 IR。编译期选择的两个 pass
（`preference-partition`、`finalize-memref-to-gpu`）已删除；全部 e2e 已迁到 driver；
`target-lower` 仅作为“把一个单元降成 GPU 代码”的工具保留。

### 3.3 GPU 执行

- 方式：`gpu.alloc host_shared`（CUDA 上即 `cuMemAllocManaged`，UVM）→ host
  `memref.copy` 写入 → kernel 直接访问同一块 buffer → host `memref.copy` 读回 →
  `gpu.dealloc`；只支持静态形状、identity layout，假设单输出，只读的
  `memref.get_global` 不写回。
- 旧方式（2026-10-27 之前）：设备内存 + 异步 `gpu.memcpy`。拷贝用的是
  `gpu.wait` 新建的 stream，和 kernel 的 stream 从未互相排序，是潜在的数据竞争；
  据记录同一用例重复 100 次约 20 次输出垃圾。
- 2026-10-27：桥接改为 UVM。远端验证：旧路径 100 次重复 100/100 通过，
  未复现 20%；UVM 版本 100/100 通过。另用 16M 元素的大缓冲在两条路径上各跑
  10 次，输出均正确。原始失败样本仍未拿到，因此“已修复”目前只能靠
  “异步拷贝路径已移除”来论证，后续用更大负载和并发继续观察。
- 仍是先正确性、后性能：host `memcpy` 还没消除，UVM 迁移开销也未调优。

## 4. 构建与测试放到远程

- **为什么**：
  1. 两台开发机都跑不了 UVM（WSL2 受 NVIDIA 驱动限制，MacBook 不具备运行条件），
     而后面的 GPU 管线都基于 UVM。本地编出来的 GPU 代码没有可验证的出口，
     与 GPU 相关的测试只能在原生 Linux 上做。
  2. 远端服务器的性能远好于这台机器和 MacBook。编译留在本地就是卡在编译上——
     全量重编四千多步，验证想法的节奏被机器拖住；把构建挪到远端，这段时间直接省掉。
- **要什么**：一条通过 SSH 使用远程原生 Linux 服务器的通道，做成一个 skill，职责是四件事：
  连接服务器；同步代码与构建配置；在远端构建；在远端跑测试。
  代码同步一旦解决，远端就是日常的构建与测试环境。
- **权限**：agent 操作远程机器时的权限控制要单独设计——能访问哪些目录、能写什么、能执行什么，
  不能默认放开整台机器。具体模型未定（见 §6）。

## 5. 接下来

1. **删除半自动路径**（见 §2.1）。成本低，先清场。
2. **运行时第一版**（见 §2.5、`caelum/doc/runtime.md`）。M1–M5 已完成并在远端验证：
   driver、op 级按需编译、产物、GPU 重编译、旧 pass 删除与 e2e 迁移。剩下的是
   `target-lower` 是否要进一步收进 driver（它现在只作为 lowering 工具被调用）。
3. **建远程构建与测试通道**（见 §4）。把日常构建与测试挪到远端；产出是那个 skill。
4. **修 GPU 正确性**（见 §3.3）。UVM 改写已完成（2026-10-27）；原 20% 失败在
   远端没有复现（旧路径 100/100，UVM 100/100），继续积累更大负载与并发下的样本。
5. **定算子集合与 benchmark**（见 §2.4）。等负载确定后一起做。

## 6. 未定问题

- annotation 的具体语法与承载位置。
- 远程操作的权限模型：agent 在远端能访问什么、写什么、执行什么，边界在哪里。
- 算子清单（等 benchmark）。

## 7. 相关文档

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
