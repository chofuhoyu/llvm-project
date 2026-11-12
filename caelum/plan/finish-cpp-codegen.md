# 完成 C++ 前端：可合并的基线与剩下四件事

> 2026-11-12 ·
> 依据：`cpp_codegen.md`、`roadmap.md`（v9）、`finish-roadmap-approach.md`、
> `operators.md`、`../doc/cpp-codegen-operators-decisions.md`、
> `../doc/finish-roadmap-decisions.md`、`../doc/benchmark.md`、
> `../review/20261109-cpp-codegen-operators.md`、
> `../review/20261109-finish-roadmap-approach.md`，以及分支
> `caelum/finish-roadmap-approach` 的代码。

## 1. 为什么需要这份文档

这条线曾经同时推进两条各自有清单的工作：运行时（driver、产物、GPU、benchmark）和
C++ 前端（整程序、切分、ABI）。两份清单分别验收，**没有一份对并集负责**：

- `cpp_codegen.md` §6.3 的 M5 三项被写成"此后的事情"，没有里程碑，也没有 owner；
- `roadmap.md` 从 v6 升到 v9 时，v6 §2.5/§3.3 里"去掉桥接的 host 拷贝、后做性能"这句
  承诺被删掉，没有留下"放弃"的记录；
- 两份 2026-11-09 的审查各自只覆盖增量，任取一份都会漏掉另一半。

于是"完成"在每一份文档里都成立，加起来不成立。本文把并集重新立成一个计划，并且只认
一件事：**每一条要么有判据和可重跑的证据命令，要么明确写进"不在判据内"**。不引用另一份
文档的"已完成"作为证据。

## 2. 完成判据

"roadmap 与 cpp_codegen 的内容全部完成"指以下五条都满足：

1. `cpp_codegen.md` §5 的 W0–W8 全部完成，含 W6 后半的 clang driver `-fskeleton`；
2. `cpp_codegen.md` §6.3 的 M1–M5 全部完成；
3. `roadmap.md` v6 §1–§6 的承诺全部兑现，或者有显式的放弃记录（当前欠一条：
   §2.5/§3.3 的"去掉桥接 host 拷贝 / UVM 迁移开销调优"）；
4. 两份 2026-11-09 审查里标着"仍在 finish 上"的条目全部处理完；
5. `caelum/finish-roadmap-approach` 合入 `caelum/skeleton-dialect`。

**不在判据内**（已声明的非目标，不重复讨论）：

- 二维动态形状、非 identity layout、多输出 GPU（`roadmap.md` §3.3）；
- 多 executor、跨进程、跨机（`roadmap.md` §2.5）；
- 异常、RTTI、`std::cout` 与静态对象（`cpp_codegen.md` §4.4）；
- benchmark 矩阵之外的算子（`roadmap.md` §2.4 的扩张规则）。

## 3. 现在的状态

### 3.1 已经成立的

有代码、有用例、有数值证据的部分：

- 函数级切分：`cir-skeleton-split` 把 host 拆成 host stub + 单元 + thunk；
- 边界 ABI：`skeleton-lower-host-abi` 在 host 降到 LLVM 之后填 thunk 的 body；
- 混合模块：`cir-to-llvm=allow-mixed-module`、`llvm.func` 入口、产物 `main` 返回退出码；
- 一条命令：`skeleton++`（等价于 `clang++ → cir-opt → driver`）；
- 纯函数词汇：`CirFuncToArith` 的标量算术、比较、select、cast、math 内建；
- 运行时：driver、按 op 按需编译、产物、UVM、managed 缓冲、显式同步、trace；
- 算子：`skeleton_map` / `skeleton_reduce` / `skeleton_matmul`，四个真实负载
  （softmax、layernorm、matmul、逐元素 add）；
- 三组 skeleton 测试与 benchmark 的数值、耗时记录。

### 3.2 还缺的三类

**一、一个没有证明的保证。** `skeleton_rt_sync` 用三步去找"该等的 context"，第三步
（`SkeletonRuntime.cpp` 的 `waitForDevice`）不 push 任何 context，等的是"当时恰好是
current 的那个"；它返回 0 时，代码无法区分"等到了 kernel"和"等在了别的 context 上"。
没有观察到错误结果——现有内存模型与用例把问题盖住了——缺的是"一定能等到"这个保证，而
`roadmap.md` §3.3 与 E6 写的是"GPU 正确性完成"。

**二、四件被挪到"此后"的功能。**

1. clang driver 的 `-fskeleton`（今天只有 `skeleton++` 脚本）；
2. 一个 host 里多个骨架算子（今天一个 host 只能一次调用）；
3. 扁平图上的融合 pass（今天每个算子一个 kernel，中间结果落内存）；
4. 去掉桥接的 host 拷贝（今天静态数组上设备要拷进拷出）。

**三、一批小账。** 两份审查里"仍在 finish 上"的条目：driver 入口对合法输入的 `cast`
会断言、host ABI 的 stride 只乘静态维、`skeleton-lower-host-abi` 没有直接测试、
`restrict` 的互不重叠契约只写在决策文档里、dispatch 对解析失败静默跳过、span 的长度
硬编码为 `i64`、`skeleton++` 的临时目录与 `--link-arg`，以及若干与代码不符的注释。

## 4. 阶段 0：可合并的基线

这一阶段不做任何新功能，只把"已有的东西"变成"记录和证据一致的东西"。

### 4.1 修同步

- `waitForDevice` 增加第四次尝试：push `driver.privateContext` 再同步（这台机器上
  kernel 跑在它上面），无论成功与否都 pop；
- `--trace` 写出实际走的分支：`waited=primary|pushed|private|none`；
- 只有"等到 kernel 所在的那个 context"才算成功；等在不认识的 context 上必须说出来，
  不能以 `error == 0` 报成功；
- `waitUnavailable` 不再一次失败就永久放弃整个进程。

验收：`--trace` 能看出真实分支；`GPU/cpp-softmax` 与 `GPU/cpp-managed-add` 仍然正确；
E6 的判定从"存疑"变成可引用的结论。

### 4.2 修入口

`mlir-skeleton-driver.cpp` 里对入口的 `cast<LLVM::LLVMFuncOp>` 改成 `dyn_cast`，按类型
决定要不要结果槽（`emitArtifact` 已经是这么做的，两处行为对齐）。补一条
`func.func @main() -> i32` 的用例。

### 4.3 文档收口

一次只改文档的提交，把"完成"改成实话：

- `roadmap.md` 升 v10：E6 的判定；把四件功能立成正式条目（M1–M4，指向本文）；写明 v6
  §2.5/§3.3 那句是**待办**还是**放弃**；"一个宿主函数只能有一次骨架调用"标注为 v1 限制、
  M1 会推翻；记录合并后的基线提交与远端日志；
- `cpp_codegen.md`：§6.3 的 M5 行标注"单列于本文"；§2.4 的"M1–M5"改成"运行时 M1–M5"；
  §2.4 的状态表加时间戳；§10 的"还没做的"改成指向本文；
- 与代码不符的注释与记录：`cpp-managed-add.cpp` 与 `cpp-softmax.mlir` 里与决策相反的句子、
  driver 两处还在讲 `tensor.empty` 的注释、`benchmark.md` §5 对同一批
  `CUDA_ERROR_INVALID_CONTEXT` 的两种解释、`runtime.md` 的 trace 格式与 D 编号、
  `cpp-codegen-operators-decisions.md` §1.3 的"标量按值传"。

验收：文档里不再有"完成"指向另一份文档的情况；每条结论都能指到 §9 台账里的一行。

### 4.4 验证与合并

按 `.profile/remote/remote.sh run` 跑一次三组测试，全绿之后：

```
git checkout caelum/skeleton-dialect
git merge --ff-only caelum/finish-roadmap-approach
```

分支关系是 84 个提交领先、0 分叉、0 个 merge commit，所以合并是快进，合并后的树与远端
验证过的 tip 逐字节相同——"绿的是哪个提交"没有歧义。合并后在 `skeleton-dialect` 上加
一个小提交，把基线提交号与远端日志名字写进 `roadmap.md` 的变更记录。

## 5. 阶段 1：`-fskeleton`

在 `clang/lib/Driver` 加 JobAction，把 `-emit-cir → cir-skeleton-split →
cir-to-llvm=allow-mixed-module → cir-strip-cir-attrs` 串进 driver，`-fskeleton` 打开它。
`skeleton++` 保留为薄封装或删除，由验收时的维护成本决定。

验收：`clang -fskeleton app.cpp -o app` 与 `skeleton++ app.cpp` 的产物行为一致；至少
map、reduce 两条 e2e 走 driver 路径并数值正确；`cpp-features` 仍通过。

## 6. 阶段 2：一个 host 里多个骨架算子

`cpp_codegen.md` §6.3 的 M5 主体，也是唯一改切分器形态的一件。分三步，各自独立提交：

1. **先把算子知识收成一张表**：`firstOperand`、参数个数、是否有纯函数，从
   `CirSkeletonSplit.cpp` 与 `CirCallLowering.cpp` 的三处 `if` 收进 `OpFacts` 常量表。
   行为不变，测试全绿。**先做这一步的原因**：M1 要在语句级按算子逐个切分，如果这份知识
   还散在四五个 `if` 里，新增的切分逻辑会把它复制一遍；
2. **语句级切分**：一个 host 里 N 次骨架调用 → N 个单元 + N 个 thunk，host 保留其余语句；
   中间结果的缓冲由 pass 在 host 侧分配（分配策略见 §10 决定 3）；
3. **翻转负例**：现在"一个 host 两次骨架调用"是负例，改成正例；补新负例（中间缓冲零长度、
   同一个 destination 被两个算子先后使用）。

同时改 `roadmap.md` §2.2 那句 v1 限制。

验收：一条 e2e，一个 host 里两个算子、第二个吃第一个的输出；CPU 与 GPU、driver 与产物、
数值与参考实现一致。

## 7. 阶段 3：融合

在扁平图之上做，不回到 region 嵌套（`operators.md` §2）。融合的粒度待定（见 §10 决定 1）。

验收：softmax 的四个单元合成一次 kernel 启动，数值不变，`benchmark.md` 的耗时表更新。

## 8. 阶段 4：去掉桥接拷贝

目标是兑现 `roadmap.md` v6 §2.5/§3.3 的承诺：静态数组上设备不再拷进拷出。前提是先定
设备路径的缓冲策略（见 §10 决定 2）。

验收：设备路径的 IR 里不再出现 `memref.copy` 进/出；或按决定 2 走"放弃"分支，把理由写进
`roadmap.md` v10。

## 9. 验收台账

台账是这份文档的执行体：每行必须有可重跑的证据，不能指向另一份文档。

| 判据 | 状态 | 证据 |
|---|---|---|
| E1 半自动路径无残留 | 达成 | `git grep -n 'CirLoopToSkeleton\|loop-map-add'` 无输出 |
| E2 annotation 语法与正负例 | 达成 | `clang/test/CIR/Transforms/skeleton-split{,-invalid}.cpp` |
| E3 放置决策在运行期，trace 可看 | 达成 | `driver-*.mlir` 的 `TRACE:` 断言 |
| E4 产物只带 CPU 实现、driver 可读回重编 | 达成 | `driver-artifact.mlir`、`driver-matmul-gpu.mlir` |
| E5 managed 单元无 `gpu.alloc`/`memref.copy` | IR 达成，端到端缺一条断言 | `skeleton-target-lower-managed.mlir`；补 `--dump-unit-ir` 的 `CHECK-NOT` |
| E6 边界显式同步并可证明等到 kernel | **待办（阶段 0.1）** | 修后：`--trace` 的 `waited=` 分支 + GPU e2e |
| E7 GPU 归约可用、softmax PASS、XFAIL 删除 | 达成 | `GPU/cpp-softmax.mlir`、`skeleton-target-lower-gpu-reduce.mlir` |
| E8 一维动态上设备、二维有诊断 | 达成 | `GPU/cpp-managed-add`、`skeleton-target-lower-invalid.mlir` |
| E9 benchmark 齐备 | 达成（耗时为一次性记录） | `../doc/benchmark.md`、`benchmark/bench.mlir` |
| E10 远程权限模型与过滤器 | 达成 | `.profile/remote/README.md`、远端日志 |
| E11 `target-lower` 去留已记录 | 达成（v10 补记录 M1–M4） | `roadmap.md` §2.3 |
| M1 一个 host 多个骨架算子 | 未开始（阶段 2） | 新增 e2e `cpp-two-ops.cpp` |
| M2 融合 pass | 未开始（阶段 3，待决定 1） | softmax 单元数 + benchmark 表 |
| M3 clang `-fskeleton` | 未开始（阶段 1） | `clang -fskeleton` 与 `skeleton++` 产物一致 |
| M4 去掉桥接拷贝 | 未开始（阶段 4，待决定 2） | 设备路径 IR 无 `memref.copy`，或 v10 的放弃记录 |

## 10. 需要拍板的三个决定

1. **融合的粒度。** 整图一个 kernel 会把放置粒度从"单算子"放大到"融合组"，与 op 级
   放置冲突（`operators.md` §2 明确反对回到嵌套）。备选：只融合同一放置偏好、同一
   executor 上的连续单元，把融合组本身作为一个放置粒度。决定之前不开工。
2. **设备路径的缓冲策略。** 去掉桥接拷贝的前提是"分配交给运行时"。对栈数组不成立，
   所以要么让设备路径统一使用 `skeleton_detail::buffer<T>`（改 benchmark 与 e2e），要么
   保留桥接，在 v10 里把 v6 的承诺标成放弃并写清代价。
3. **中间结果的缓冲分配（M1 需要）。** pass 在 host 侧为算子之间的中间结果分配缓冲：用
   `alloca`（栈，大小受限）还是 `malloc`（堆，需要配对释放并处理失败）。这决定 M1 的
   诊断边界，开工前要定。

## 11. 纪律

1. **一次只推进一条线。** 运行时与前端不在同一轮里并行收尾；M1、M2 不在同一个分支里做。
2. **一条线只有一张验收表。** 新增的"完成"必须在 §9 里有一行，且有可重跑的命令。
3. **合并点显式验证。** 合并前跑一次远端，合并后记录基线提交与日志；不用合并提交掩盖
   未验证的树。
4. **不写工作量估计。** 顺序由依赖决定，不由"哪个快"决定。
5. **文档不互相引用"已完成"作为证据。** 证据只能是测试、命令或日志。
