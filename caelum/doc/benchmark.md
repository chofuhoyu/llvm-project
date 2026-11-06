# Benchmark：负载、算子矩阵、数值与耗时

> 2026-11-07
>
> `roadmap.md` §2.4 把「算哪些算子」交给 benchmark 定；`operators.md` §5 要求真实负载
> 有固定输入、参考数值、CPU/GPU 两种实现与耗时。本文是那份 benchmark 的说明与结果。
> 程序在 [`../../mlir/test/Integration/Dialect/Skeleton/Benchmark/bench.cpp`](../../mlir/test/Integration/Dialect/Skeleton/Benchmark/bench.cpp)，
> lit 入口是同目录的 `bench.mlir`。

## 1. 选定的负载

一个可在单机 4090 上重复运行的负载，四个部分，都是真实算子的组成部分：

| 负载 | 形状 | 算子数 | 说明 |
|---|---|---|---|
| softmax | 32 × 128 | 4 | rowmax → shift_exp → rowsum → normalize，`operators.md` §7 的第一个真实算子 |
| layernorm | 32 × 128 | 6 | rowsum → center → square → sumsq → stats → scale |
| matmul | 32×128 · 128×32 | 1 | 唯一不能写成纯函数的原语 |
| map-add | 262144 | 1 | 逐元素 add，缓冲由运行时分配 |

前三个用静态数组作数据参数，所以设备侧走 Unified Memory 桥接（每个算子一次分配、一次
进、一次出）。`map-add` 用 `skeleton_detail::buffer<T>`，长度只在运行时才知道，单元标记
`managed`，设备侧没有分配也没有拷贝。两条路径都是一等公民，耗时表因此分开列。

选择标准：能重复（固定输入，无随机数）、有参考实现（同一程序里的 C++ 直白实现）、
覆盖两条内存路径、覆盖归约与 contraction 两类非逐元素计算。

## 2. 算子矩阵

| 需求 | 方言原语 | C++ 入口 | 实现 |
|---|---|---|---|
| 逐元素 | `skeleton.map` | `skeleton_map` | `linalg.generic`，标量极值经 `tensor.from_elements` 折进 region |
| 归约 | `skeleton.reduce` | `skeleton_reduce` | `linalg.fill` + `linalg.reduce` |
| contraction | （无新方言 op） | `skeleton_matmul` | `linalg.fill` + `linalg.matmul` |
| 逐元素（手写 MLIR） | `skeleton.vector.add` | —— | `linalg.add` |

- 划分原则：能被一个标量纯函数描述的进 `map`/`reduce`，不能的（contraction）单独成原语。
- 任意带 `skeleton.preference` 的 linalg op 仍是合法单元，手写 MLIR 用例用它验证运行时；
  C++ 侧只暴露上表三个入口。
- 没有进矩阵的：融合算子（一个 kernel 跑完整个 DAG）、GEMM 之外的 BLAS 级别算子、
  需要多输出的算子。前两个需要融合 pass 或新的方言 op，第三个是 v1 非目标。

## 3. 判据与数值

每个负载在计时前先和 C++ 参考实现逐元素比对（容差 1e-4，matmul 1e-2），不一致则打印
最大的偏差、位置与两个值，并以非零状态退出——**计时的数字只对已知正确的结果打印**。
另外检查「算子不改写输入」：算子写 destination，输入必须原样。

lit 入口跑三次：driver + GPU 可用、driver + 强制 GPU 不可用（同一份二进制落到 CPU）、
产物（只带 CPU 代码）。三者的数值都必须与参考一致。

## 4. 耗时

远端 4090，`debug-cuda` 构建（**未优化**），32 次迭代取平均，每次运行前先跑一遍热身
（第一次 resolve 要编译单元，那笔时间不属于测量）。同一份 `bench.cpp` 三种跑法：

| 负载 | driver，设备 | driver，CPU | 产物（CPU） |
|---|---|---|---|
| softmax 32×128（4 个算子） | 2.23 ms | 0.061 ms | 0.035 ms |
| layernorm 32×128（6 个算子） | 2.49 ms | 0.034 ms | 0.015 ms |
| matmul 32×128 · 128×32 | 0.44 ms | 0.38 ms | 0.22 ms |
| map-add 262144（运行时缓冲） | 0.51 ms | 0.15 ms | 0.12 ms |

怎么读这张表：

- **这个尺寸下设备更慢，而且慢得不是一点点。** softmax 在设备上 2.23 ms、在 CPU 上
  0.061 ms：每个算子一次 `gpu.alloc host_shared` + 两次 `memref.copy` + 一次同步，而数据只有
  16 KB。UVM 的迁移开销在首次触页时付，之后每次调用仍然要为进出拷贝付一次。表里唯一接近的
  是 matmul（一次调用、数据量对得上），managed 的 map-add 也仍然慢 3 倍——1 MB 的 UVM 在
  第一次迁移之后仍有开销。
- **结论不是"放置没用"，而是"这些负载太小"。** 每次算子调用的固定成本（分配、拷贝、同步）
  约 0.5 ms 量级，要摊薄它需要每个算子几十 MB 以上的数据；这正是 roadmap「先正确性、后性能」
  的下一步要面对的问题，也是 `operators.md` §2 说「跨算子边界开销要等真实负载才知道」的那笔账。
- **产物比 driver 快 1.7–2.3 倍（CPU）。** 两条路都按 O2 配置，但产物是 AOT 一次性编译，
  driver 的 host 走 LLJIT；具体差在优化管线的哪一段没有查，不在 roadmap 完成判据内，记在这里
  供后续追。

## 5. 压力与正确性验证

| 项 | 做法 | 结果 |
|---|---|---|
| 重复运行 | 同一份 `cpp-softmax`（GPU）连续跑 10 次 | 10/10 通过，数值一致 |
| 并发进程 | 4 个 `cpp-softmax`（GPU）同时跑 | 4/4 通过，互不干扰 |
| 大缓冲 | `map-add` 262144 个 float（1 MB，运行时缓冲） | 数值正确，见 §4 |
| 多单元 | `GPU/driver-multi-unit-gpu`（同一进程两个 GPU 单元） | 通过 |
| 放置回退 | `GPU/driver-adaptive`（GPU → CPU → GPU）、`GPU/driver-memory-fallback`（显存放不下） | 通过 |
| 内存检查 | `compute-sanitizer --tool memcheck` 跑 benchmark、`cpp-softmax`、`cpp-managed-add` | 0 个内存错误；每个进程 1–2 条驱动 API 错误 |

顺手记下 `compute-sanitizer` 的读法，因为它一开始报了两千多条：那些全是
`CUDA_ERROR_INVALID_CONTEXT`，来自运行时反复 push 一个已经 current 的 context、以及在拿不到
可用 context 时读空闲显存——不是内存错误，但会把真问题埋掉。现在剩下的 1–2 条是同一台机器的
驱动拒绝 `cuCtxPushCurrent`/`cuCtxSynchronize`（见 D8），每次运行只尝试一次：等待路径按
"当前就是 kernel 的 context → 把 kernel 的 context 压栈 → 等当前 context" 三步走，三步都被
拒绝就记下来不再重试。**memcheck 本身 0 错误。**

顺序这件事要单独说清：这台机器上三步等待都不可用，所以 managed 路径的"内核写、宿主紧接着读"
在这台机器上靠的是 UVM 自己的语义（宿主访问设备刚写过的页会触发迁移，等于等那部分工作），
而不是那行 `skeleton_rt_sync`。在能把 kernel 的 context 压栈的驱动上，等待才真正生效。两条都
写进 D8，免得把"绿了"当成"证明了"。

## 6. 复现

```bash
# 一条命令编出并运行（driver 路径，GPU 可用时落在设备上）
skeleton++ mlir/test/Integration/Dialect/Skeleton/Benchmark/bench.cpp --run \
  --shared-lib=$BUILD/lib/libmlir_cuda_runtime.so \
  --shared-lib=$BUILD/lib/libmlir_runner_utils.so \
  --shared-lib=$BUILD/lib/libmlir_c_runner_utils.so

# 强制 CPU（同一份程序：放置是运行时决定）
SKELETON_BENCH_DEVICE=cpu skeleton++ ... --run ...

# 产物路径（只带 CPU 代码）
skeleton++ ... -o bench && ./bench
```

`SKELETON_TRACE=1` 会在 stderr 打出每个算子的放置决策（含 `exec=` 字段）。
