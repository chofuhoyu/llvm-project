# Caelum 计划

> v1 · 2026-10-23
>
> 这份文档记录方向、决定与接下来要做的事。没写进来的部分都还没定。

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

## 3. 现状

### 3.1 编译器

手动路径可用：C++ 源码 → CIR → skeleton → linalg → CPU/GPU lowering，两条路径各有 e2e。
半自动路径仍在树上，是接下来要删的第一样东西。

### 3.2 运行时

**不存在**。放置决策没有承载者，target 目前是在编译期被选定的。

### 3.3 GPU 执行

- 方式：`gpu.alloc` → 异步 memcpy（H→D）→ kernel → 异步 memcpy（D→H）→ `gpu.dealloc`；
  只支持静态形状、identity layout，假设单输出。
- 缺陷：同一用例重复 100 次约 20 次输出垃圾，自发现起未修复。
  在修掉之前，GPU 上的放置实验不可信。
- 方向：用手写搬运换成 **UVM**；先正确性，后性能。

## 4. 接下来

1. **删除半自动路径**（见 §2.1）。成本低，先清场。
2. **注入一个最简运行时**（见 §3.2）。让放置决策有地方发生；
   形态、注入点、与编译产物的衔接方式都还没定。
3. **修 GPU 正确性**（见 §3.3）。先做 100 次复现拿到可分析的失败样本，
   再归因，再用 UVM 改写桥接，最后 100/100 回归。
4. **定算子集合与 benchmark**（见 §2.4）。等负载确定后一起做。

## 5. 未定问题

- annotation 的具体语法与承载位置。
- 运行时的形态与注入点：由谁读 annotation、在哪里做决策。
- `skeleton-preference-partition` 与 `skeleton-target-lower` 的具体去留。
- 算子清单（等 benchmark）。

## 6. 相关文档

`caelum/report/`、`caelum/review/` 里的旧文档只作历史记录；
其中半自动路径与识别算法的内容已经作废。

## 变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v0 | 2026-10-23 | 按口述顺序记录方向、GPU、运行时三段 |
| v1 | 2026-10-23 | 按目标 / 决定 / 现状 / 接下来 / 未定重组，删除过程性记录 |
