# 20261109：完成 roadmap 的增量（finish-roadmap-approach）

审查范围：`git log --oneline caelum/cpp-codegen-operators..caelum/finish-roadmap-approach`，
共 49 个提交（2026-11-05 → 2026-11-07，从 `03a59fad4e49` 到 `36ba098b21e4`）。
`git diff --stat` 是 39 个文件、+2914/−240。

这条分支包含 `caelum/cpp-codegen-operators` 的全部 33 个提交（两条分支在 `280621fe799d`
同一棵树），所以本文**只审增量**：C++ 前端那部分见
[`20261109-cpp-codegen-operators.md`](20261109-cpp-codegen-operators.md)，不在这里重复。
增量本身分两段：11-05/11-06 的 WP0–WP9 落地（同步、managed、可写视图、matmul、benchmark、
roadmap v9），以及 11-07 之后针对 `review/20261107.md` 的 14 个修复提交。

结论先给：这一段的硬证据比上一段强（可写视图 + GPU 归约 + 放置断言都落地了）。最需要处理的
是 **GPU 同步到底在等哪个 context**：`skeleton_rt_sync` 的第三步既等不到 kernel，又会把
“等错了”报告成成功，而 D8 与 `benchmark.md` 把同一件事记成了“驱动不让等”。这一条是 E6 的
全部风险所在（问题 1）。其次是 map 的 `restrict` 契约没有任何出口（问题 2）。其余都是证据
补强或注释收口。

## 0. 方法与边界

读过的：

- 记录：`caelum/plan/finish-roadmap-approach.md`（E1–E11、WP0–WP9）、
  `caelum/doc/finish-roadmap-decisions.md`（D1–D9）、`caelum/doc/benchmark.md`、
  `caelum/doc/runtime.md`、`caelum/plan/roadmap.md`（v9）、`caelum/review/20261107.md`。
- 代码：`SkeletonRuntime.{h,cpp}` 全文（context、alloc/free、sync、probe、exec）、
  `SkeletonTargetLower.cpp`、`SkeletonOps.h` 的 `buffer<T>`、`CirSkeletonSplit.cpp` 与
  `CirCallLowering.cpp` 的增量、driver 的 `prepareGpuUnit` / `registerRuntimeSymbols` /
  `dump-unit-ir`、以及 `mlir/test/Dialect/Skeleton/skeleton-target-lower-*.mlir`、
  `mlir/test/Integration/Dialect/Skeleton/{GPU,Benchmark,CPU}/*` 的所有相关用例。
- 上游对照：`mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp` 的 `ScopedContext`（push/pop
  行为）——判断“同步等的是哪个 context”时这是必要证据。
- `caelum/` 之外的 `.profile`（独立仓库）只看了 D1 相关的一条过滤器
  （`.profile/remote/remote-impl.sh:336` 是 `(skeleton|cir-func-to-arith)`，README 有
  “权限边界（给 agent）”一节），用来核 E10。

没做、也做不到的：

- **没有跑构建，也没有跑任何测试**，包括远端。`benchmark.md` 里的耗时、10/10、4/4、
  `compute-sanitizer` 结果都是**作者的记录**，我只能核对它们与代码/彼此是否自洽，不能复核数字。
- CUDA 驱动的实际行为无法验证。下面问题 1 的结论是“代码与记录不自洽”，不是“实测某条路径失败”；
  凡是涉及驱动的部分都写成条件句。
- 行号：`path:line` 取自 `caelum/finish-roadmap-approach` 的树（当前工作区）。

## 1. 改动摘要

一句话：把 roadmap 剩下的硬缺口补齐，并给上一轮审查的 9 条代码问题一个交代。补齐的部分是
真的（managed 缓冲、可写 destination 视图、GPU 归约、matmul、benchmark、trace 的放置断言），
`review/20261107.md` 的 9 条代码问题里 5 条被直接修掉（C++ 用例的放置断言、`exec=`、
`withPrimaryContext` 的命名与死变量、`skeleton++` 的 driver 参数、标量按引用）、2 条部分处理
（失败路径的诊断、过期注释），1 条换了形式仍在（“同步等错 context”改成“分三步找正确 context”，
第三步的语义与记录仍有洞，见问题 1），1 条没动（`static_cast` 塞 option）。

### 1.1 设备同步与 CUDA context（D3、D8）

- 新增 `skeleton_rt_sync(int32_t device)`；`skeleton-target-lower` 在每个 GPU 单元的调用之后、
  第一次 copy-back 之前插入一次，并往单元模块里声明该符号。
- 运行时改成两类 context：**等待**尽量针对 kernel 所在的 primary context（三步：当前就是
  primary → push primary → 等当前）；**分配**用运行时自己 `cuCtxCreate` 的 context，因为
  managed 内存属于设备而不属于 context。
- driver 把 `skeleton_rt_sync` 等符号注册进**单元自己的** JIT 引擎（`registerRuntimeSymbols`
  在 `compileUnit` 里也调用一次），等待才可能发生在单元模块里。

### 1.2 运行时拥有的缓冲与 managed 单元（D4、D5）

- `skeleton_detail::buffer<T>`：构造时 `skeleton_rt_alloc`，析构时 `skeleton_rt_free`；两个成员
  与 `span` 相同，边界上仍是 `memref<?xT>`。
- 切分时数据参数**全部**是 `buffer` 才置 `skeleton.managed`，混用报错；`prepareGpuUnit` 把属性
  带到 GPU 实现上；`skeleton-target-lower` 对 managed 目标跳过 `gpu.alloc` / `memref.copy` /
  `gpu.dealloc`，但仍插入同步。
- 动态维的 e2e 因此成立：一维动态（`buffer`/`span`）可以上设备。

### 1.3 destination 一律可写视图（D2）

`createToTensor` 增加 `writable` 参数；map / reduce / matmul 的 destination 都用可写视图，
不再依赖 `tensor.empty` + empty-tensor elimination，也不再“写视图 + 丢弃 fill 结果”。
`CirSkeletonSplit.cpp:278/302/333` 三处。GPU softmax 的 `XFAIL` 因此删除，归约的原因被
更正为“destination 视图不可写”，不是“归约本身不行”。

### 1.4 标量参数与 matmul

- 调用点写在 `skeleton_map` 变参位置的标量是“指向标量的指针”，切分现在把它当标量值并 load
  （`CirCallLowering.cpp` 的 `byReference` 分支、`rewriteHost` 的 load）；`skeleton_reduce` 的
  identity 仍按值。
- `skeleton_matmul(out, a, b)`：没有纯函数，单元里是 `linalg.fill` + `linalg.matmul`
  （`linalg.matmul` 是累加语义，所以先清零），destination 是第一数据参数。

### 1.5 可观测性（D6）与工具

- `skeleton_rt_executor_id/count`（v1 恒 0/1），trace 的 resolve 行与 stats 行末尾追加 `exec=`；
- `skeleton++` 增加 `--driver-arg=`，`GPU/cpp-*` 与 `Benchmark/bench` 因此能用 `--trace` 断言
  `compile device=GPU` / `choose=GPU`——这是上一轮“绿了但证明不了”的正解；
- driver 增加 `--dump-unit-ir`。

### 1.6 benchmark 与文档

四个负载（softmax / layernorm / matmul / map-add）、算子矩阵、CPU/GPU/产物三行跑法、
数值与耗时表、压力与 sanitizer 记录（`caelum/doc/benchmark.md`）；roadmap 升到 v9 并与代码
对齐；`operators.md`、`cpp_codegen.md`、`runtime.md`、`finish-roadmap-decisions.md` 同步更新。

## 2. 重点文件与逻辑

| 文件 | 这次做了什么 | 先看哪段 |
|---|---|---|
| `mlir/lib/Dialect/Skeleton/Runtime/SkeletonRuntime.cpp` | context 的两套用法、alloc/free、sync、probe、exec | `withContext`（分配）→ `waitForDevice`（等待的三步）→ `skeleton_rt_sync` → `skeleton_rt_alloc`/`free` |
| `mlir/lib/Dialect/Skeleton/Runtime/SkeletonRuntime.h` | ABI 契约 | `skeleton_rt_sync` / `alloc` / `free` 的注释（对失败的约定是“退出 1，永不返回 NULL”） |
| `mlir/lib/Dialect/Skeleton/Transforms/SkeletonTargetLower.cpp` | managed 跳过桥接、同步插点 | 收集 `gpuCalls`/`managedTargetNames` 的 walk；Phase B 的“先 sync、再 copy-back、再 dealloc” |
| `mlir/tools/mlir-skeleton-driver/mlir-skeleton-driver.cpp` | 把 managed 带到 GPU、单元引擎注册符号、`--dump-unit-ir` | `prepareGpuUnit`（属性怎么过继）、`lowerGpuUnit`（管线顺序）、`compileUnit` 里的 `registerRuntimeSymbols` |
| `clang/lib/CIR/Dialect/Transforms/CirSkeletonSplit.cpp` | managed 判定、标量解引用、可写视图 | `resolveDataOperands` 的 `byReference`/`buffer` 混用检查、`createUnit` 三个分支的 destination 视图 |
| `clang/lib/CIR/Dialect/Transforms/CirCallLowering.cpp` | `createToTensor(writable)`、`isSkeletonBuffer` | `createToTensor`、`resolveDataOperands` |
| `mlir/include/mlir/Dialect/Skeleton/SkeletonOps.h` | `buffer<T>` 与 `skeleton_matmul` | `buffer` 的构造/析构与 `allocationDevice` |
| `mlir/test/Dialect/Skeleton/skeleton-target-lower-managed.mlir`（新） | managed 不桥接的 IR 断言 | 全文（`CHECK-NOT: gpu.alloc` / `memref.copy` / `gpu.dealloc` + `call @skeleton_rt_sync`） |
| `mlir/test/Dialect/Skeleton/skeleton-target-lower-gpu-reduce.mlir`（新） | 归约在设备上的形状 | 全文（`scf.parallel` 里套 `scf.for`，累加器在 memref 里） |
| `mlir/test/Integration/Dialect/Skeleton/GPU/cpp-managed-add.{cpp,mlir}`（新） | 运行时缓冲的 e2e | `.cpp` 的哨兵设计、`.mlir` 的两条 RUN（数值 + TRACE） |
| `mlir/test/Integration/Dialect/Skeleton/Benchmark/bench.{cpp,mlir}`（新） | benchmark | `bench.cpp` 的“先比数值再计时”、`bench.mlir` 的四种跑法 |
| `caelum/doc/finish-roadmap-decisions.md` | D1–D9 的“背景/选择/放弃/代价” | D8（context）与 D2（可写视图）是本文问题的主要对照物 |

## 3. 阅读顺序

1. `caelum/plan/finish-roadmap-approach.md` §1 + §5（E1–E11）。
   回答：这 49 个提交要证明什么。
2. `caelum/doc/finish-roadmap-decisions.md` 的 D2–D5、D8、D9。
   回答：可写视图、同步位置、managed、一维动态、context 各自被怎么权衡过。这一节是本分支
   所有代码的“为什么”，也是本文问题 1 的对照物。
3. `SkeletonRuntime.cpp`：先 `withContext` 与 `waitForDevice`，再 `skeleton_rt_sync`，最后
   `skeleton_rt_alloc`/`free`。
   回答：分配和等待各用哪个 context？失败时留下什么痕迹？
4. 上游 `mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp` 的 `ScopedContext`。
   回答：kernel 到底在哪个 context 上启动、push 失败时它做了什么。读第 3 步必须配合这一段，
   否则会把“运行时自己的选择”误读成“驱动的行为”。
5. `SkeletonTargetLower.cpp`：walk 与 Phase B。
   回答：桥接和 managed 的分叉在哪、同步为什么插在 copy-back 之前。
6. `CirSkeletonSplit.cpp` + `CirCallLowering.cpp` 的增量：`createUnit` 三个分支、`byReference`
   标量、`buffer` 全有或全无。
   回答：单元的形状从哪来、managed 为什么必须是单元的属性。
7. driver：`prepareGpuUnit` → `lowerGpuUnit` → `compileUnit`。
   回答：一个 C++ 单元从 `func.func` 到“能被 host 间接调用的 GPU 实现”经过哪里。
8. 测试：`skeleton-target-lower-managed.mlir` → `gpu-reduce.mlir` → `GPU/cpp-*.mlir`（数值 +
   TRACE 两条 RUN）→ `Benchmark/bench.mlir`。
   回答：每一条判据的证据是 IR、是数值，还是 trace。

## 4. E5–E9 的判定

E1/E2/E3/E4/E11 由前一段 33 个提交决定，本文不重判（`review/20261107.md` 的结论仍成立）；
E10 只需核对过滤器，已在 `.profile` 里确认（`(skeleton|cir-func-to-arith)` + README 的
权限矩阵）。受这 49 个提交影响的是下面五项：

| 判据 | 判定 | 证据 | 备注 |
|---|---|---|---|
| E5 managed 无 `gpu.alloc`/`memref.copy` | 达成（IR 层）/ 端到端缺一条直接证据 | `skeleton-target-lower-managed.mlir` 的 `CHECK-NOT`；`GPU/cpp-managed-add` 数值 + `TRACE: op=0 compile device=GPU` | 手写 MLIR 证明了 pass，C++ 单元只由数值+trace 证明；`--dump-unit-ir` 加上一条断言就能闭环，见问题 3 |
| E6 边界显式同步；压力/大负载/sanitizer | 部分达成，同步的语义存疑 | lowering 确实插了 `skeleton_rt_sync`（`SkeletonTargetLower.cpp:240`），并有 IR 断言；压力/sanitizer 只有 `benchmark.md` §5 的文字 | 问题 1：第三步的“成功”不等于“等到了 kernel”；压力/大负载仍没有可重复的脚本，只有一次性记录 |
| E7 GPU 归约可用；softmax GPU PASS；XFAIL 删除 | 达成 | `XFAIL` 已删；`GPU/cpp-softmax.mlir` 断言四个单元 `compile device=GPU` + 数值；`skeleton-target-lower-gpu-reduce.mlir` 钉住“并行套串行”的形状 | 归约的 IR 断言与数值断言分开，这点做得好 |
| E8 一维动态上 GPU；二维动态有明确诊断 | 达成（一维）/ 诊断无用例 | `GPU/cpp-managed-add` 是 `memref<?xf32>` 的设备 e2e；`skeleton-target-lower-invalid.mlir` 有二维动态的 `expected-error` | host ABI 的“动态维必须是最后一维”仍然没有用例（见问题 4） |
| E9 benchmark 选定；矩阵/入口/数值/耗时齐备 | 达成（有保留） | `benchmark.md` 四负载、矩阵、耗时、压力表；`bench.mlir` 三种跑法 + TRACE；`skeleton_matmul` 入口 | 耗时是一次性手工记录；“产物 vs driver”的差异解释其实在代码里，见问题 6 |

## 5. 问题

### 1. ［高］同步第三步既等不到 kernel，又把“等错了”报告成成功

这条是 `review/20261107.md` 问题 1 的延续：那一轮指出 `skeleton_rt_sync` 同步的是运行时自建
的 context。本分支把它改成了三步（`SkeletonRuntime.cpp:265-300`）：

1. 当前 context 就是 primary → `cuCtxSynchronize`；
2. 否则 push primary 再等；
3. 否则直接等“当前 context”，失败就记下 `waitUnavailable`，以后不再试。

问题在于第三步**不 push 任何 context**，而“kernel 真正跑的那个 context”在第三步执行时通常
已经不是 current：

- `withContext`（`:230-259`）第一次调用时用 `cuCtxCreate` 建私有 context，这个调用会让它成为
  current；因为紧接着 `current == privateContext`，所以既不 push 也不 pop，私有 context
  就留在 current 上。managed 分配能成功，说明这一步是成功的（`skeleton_rt_alloc` 分配失败会
  `exit(1)`）。
- MLIR 的 wrapper 在每次 launch 时 `cuCtxPushCurrent(primary)`；按 D8 的记录这台机器上 push
  失败，但 `ScopedContext` 的析构仍然无条件 `cuCtxPopCurrent(nullptr)`
  （`mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp:83-100`），于是把**私有 context 弹掉**，
  之后没有 current context。
- 第三步对“没有 current context”调用 `cuCtxSynchronize`，返回 `CUDA_ERROR_INVALID_CONTEXT`。
  于是 `waitUnavailable` 置位，`--trace` 打出那条“驱动不让等”的 note。

也就是说，D8 与 `benchmark.md` §5 记录的“三步等待全部被驱动拒绝”，更准确的解释是
**第三步没有 context 可用**：kernel 实际用的就是运行时自己的私有 context，句柄一直在
`driver.privateContext` 里，代码却没有试。如果 `cuCtxCreate` 之后私有 context 仍然是 current
（例如某次 wrapper 的 push 成功、pop 回来了），第三步就会**成功**，但它等的仍然不是 kernel 的
context——而 `error == 0` 与“kernel 已经完成”在代码里没有任何区分，`waitUnavailable` 也不会
置位。

两个方向都指向同一处：返回值只表达“某个 context 同步成功”，不表达“等到了 kernel”。D8 的
诚实（“这不是已经证明了顺序正确”）是对的，但代码没有把这份不确定性暴露出来：note 只覆盖
“完全没有 context 可等”，恰好是最无害的那种；真正危险的那种（等在别的 context 上）是静默的。

建议：在 `waitForDevice` 里加第四次尝试——push `driver.privateContext` 再同步（这台机器上
kernel 就在它上面），并在 `--trace` 里写出实际走的分支（`waited=primary|pushed|private|none`）。
另外 `waitUnavailable` 一次失败就永久放弃整个进程，也值得再想想：它把“这次没有 current
context”当成了“这台驱动永远不能等”。

### 2. ［中］map 的 destination 与输入可以是同一块缓冲，而两者都被声明为 `restrict`

`createToTensor`（`CirCallLowering.cpp:52-54`）**永远**设 `restrict=true`，只把 `writable`
开放出来。于是 `skeleton_map(f, a, a)`（同一个 host 参数既作 destination 又作输入）会让单元里
出现两个 `restrict` 视图指向同一块 memref：一个是 `writable` 的 outs（`CirSkeletonSplit.cpp:278`），
一个是只读的 ins。

- 契约只写在 `cpp-codegen-operators-decisions.md` §2.6（“一次算子调用传入的所有缓冲区必须互不
  重叠”），`SkeletonOps.h` 没有写，切分 pass 也不检查；
- 同一个宿主参数在 destination 和输入各出现一次（`skeleton_map(f, a, a)`）会被接受，而它正是
  §2.6 禁止的“重叠”；rank 相同的逐元素原地写恰好是对的，所以今天观察不到错误；
- 真正会错的是“同一块内存经两个不同形状的参数进来”（调用点把同一个数组同时当作
  `float (&)[2][4]` 和 `float (&)[2]` 传进两个参数），这时广播的输入会被原地写破坏；前端既
  不拒绝也不诊断，而 `restrict` 已经向 bufferization 承诺了不重叠。

这与本分支的动机直接相关：`restrict` 是为了让 one-shot bufferization 接受 `to_tensor`，而
bufferization 真的会按“不重叠”去做决定。要么在切分时对“同一宿主参数出现在 destination 和
输入”报错，要么在 outs 视图上不加 `restrict` 并接受一次拷贝；无论哪种，都比把契约只留在
决策文档里好。

### 3. ［中］managed 单元“没有桥接”缺一条来自 C++ 前端的直接证据

`skeleton-target-lower-managed.mlir` 证明的是 pass：给定带 `skeleton.managed` 的函数，它不桥接。
C++ 侧的证据链是“切分置属性 → `prepareGpuUnit` 过继 → pass 看到属性”，中间没有任何断言；
`GPU/cpp-managed-add` 只证明数值正确 + 单元被编到 GPU。

本分支刚好加了 `--dump-unit-ir`，却没有任何测试用它（全仓只有 `mlir-skeleton-driver.cpp:120`
的定义处提到它）。给 `cpp-managed-add` 加一条 `--dump-unit-ir` 的 `CHECK-NOT: gpu.alloc` 就
能把 E5 的端到端证据闭环，成本很低。

### 4. ［中］`skeleton-lower-host-abi` 仍然没有任何直接测试

上一轮的 E8 行就记过这件事（“`skeleton-lower-host-abi` 的‘动态维必须是最后一维’诊断没有任何
用例”），本分支只改了诊断措辞（“a dynamic dimension must be the last one”）。
`mlir/test/Dialect/Skeleton/` 下依旧没有 `skeleton-lower-host-abi*.mlir`：`skeleton.host_abi`
形状、单元签名不符、动态维限制三条诊断都只能靠 C++ e2e 间接覆盖，E8 的“明确诊断”因此没有用例。

### 5. ［低］driver 里两处注释还在讲 `tensor.empty`

`mlir-skeleton-driver.cpp:234-238` 与 `:335-338` 仍然说“单元算到临时张量、再由
empty-tensor-elimination 改写进 destination”。本分支改成可写视图之后单元已经不再产生
`tensor.empty`（`cpp_codegen.md` §10 最后一条明确写了这点），这两个 pass 现在只对手写 MLIR
的单元有意义。注释与代码不符，而这两处正是别人理解“为什么还要跑这个 pass”的唯一入口。

### 6. ［低］`benchmark.md` 把“产物更快”归为没查过的问题，但差异就在自家代码里

§4 说产物比 driver 快 1.7–2.3 倍，“具体差在优化管线的哪一段没有查”。实际上
`writeAndLink`（`mlir-skeleton-driver.cpp:832-880`）对产物显式跑了
`buildPerModuleDefaultPipeline(O2)`，而 driver 侧 `engineOptions` 只设
`jitCodeGenOptLevel`（codegen 优化级别，不是 IR 优化管线）。也就是说两条路的差异在代码里
写着：产物多跑了一整条 IR 优化管线。这条应当在表下面写明，否则“driver 慢”会被误读成
LLJIT 的固有开销。

### 7. ［低］`populateCIRToLLVMPasses` 仍然用 `static_cast` 塞选项

`clang/lib/CIR/Lowering/DirectToLLVM/LowerToLLVM.cpp:5837-5839`：工厂返回
`unique_ptr<Pass>`，再强转回 `ConvertCIRToLLVMPass *` 写 option。这次给 pass 加了拷贝构造，
但绕开 option 正常构造路径这一点没变（上一轮同一问题未处理）。给工厂加一个 bool 形参，
或直接用 `PassOptions` 更稳。

### 8. ［低］`skeleton_rt_sync` 的 note 与 `stats` 的 `exec=` 没有断言

`driver-adaptive.mlir` 现在断言了 trace 行的 `exec=0`，但 stats 行的 `exec=`（`SkeletonRuntime.cpp:647`）
没有；`skeleton_rt_sync` 的“驱动不让等”那条 note（`:382-395`）也没有任何用例碰到——它恰好是
问题 1 里唯一能暴露不确定性的东西。

## 6. 文档与记录一致性问题

1. **D8 与 `benchmark.md` §5 的“三步全被拒绝”**：如问题 1，这更像是“第三步没有 current
   context”，不是“驱动拒绝等待”。记录里唯一与驱动有关的依据是 `SkeletonRuntime.cpp:246-247` 的注释
   （“Pushing a context that is already current is an error on the drivers this ran against”），
   而 CUDA 本身允许 push 已 current 的 context（引用计数 +1、pop 后恢复）。把这条当成驱动
   事实，直接导致了“cuCtxCreate 之后不 push 也不 pop”的写法。
2. **`benchmark.md` §5 内部对同一批 `CUDA_ERROR_INVALID_CONTEXT` 有两种解释**：`:90` 说来自
   “运行时反复 push 一个已经 current 的 context”（旧代码的行为），`:93` 说现在剩下的 1–2 条是
   “驱动拒绝 `cuCtxPushCurrent`/`cuCtxSynchronize`”（新代码的行为）。两句话相距三行，读者
   无法知道哪一个描述了当前代码。
3. **`GPU/cpp-managed-add.cpp:16-17`** 说“Nothing orders the host's read after the kernel except
   the synchronization the lowering emits”，而 D8 说这台机器上那次同步不可用、顺序靠 UVM 语义。
   同一分支里两句话对立；测试注释给的是比决策记录更强的承诺。
4. **`GPU/cpp-softmax.mlir:4-8` 的旧句子**（“归约上不了设备会让这条测试失败，而不是变成悄悄
   变慢的程序”）在本分支加了 TRACE 之后仍然留在文件里，而十行之下的新注释说“没有这一行，
   全部回退 CPU 也照样通过”。同一个文件对同一件事给了相反的两句话，旧的那句删掉或改写成
   “数值那条 RUN 不能区分放置”。
5. **`runtime.md` §8 的 trace 格式**仍写 `op= pref= probe= free= choose= reason=`
   （`:239`），代码已经追加 `exec=`（`SkeletonRuntime.cpp:588`）。同一文件 §14 的
   “细节见 D3–D6” 也应改成 D3–D9——那张表里就列了 D8/D9 两条。
6. **压力与 sanitizer 证据仍是散文**：`benchmark.md` §5 的 10/10、4/4、`compute-sanitizer`
   都只有结论，没有可重跑的命令（§6 只给了 benchmark 的三条跑法）。本分支已经把“这台机器上
   等待不可用”写进了记录，这是进步；但“memcheck 0 错误”仍然不可复核，而且它恰好不能证明
   顺序正确。

## 7. 结论与建议

这条增量的方向是对的，而且大部分是硬证据：managed 的跳过桥接、destination 可写视图（连带
`XFAIL` 的删除与原因更正）、GPU 归约的形状、`skeleton_matmul` 入口、benchmark 的“先比数值
再计时”、C++ GPU 用例的放置断言，都能在树里找到代码和用例。上一轮 9 条代码问题里 5 条被
直接修掉、2 条部分处理（分解见 §1），“绿了但证明不了”这条批评在本分支上基本不成立了。

仍然需要处理的是一个真问题和一组证据补强：

1. **把同步的目标钉死，并让失败可见**：在 `waitForDevice` 里试 `driver.privateContext`
   （这台机器上 kernel 就在它上面），并把走的分支写进 `--trace`；只把“等到 primary”算作
   真正的等待。否则 D8 的诚实会被代码悄悄抹平成 `error == 0`。
2. **给 `restrict` 的互不重叠契约一个出口**：切分时报错（destination 与输入同一宿主参数），
   或者 outs 视图不加 `restrict`。这条现在只写在决策文档里。
3. **两条一句话就能补的断言**：`--dump-unit-ir` 给 `cpp-managed-add` 加 `CHECK-NOT: gpu.alloc`；
   `skeleton-lower-host-abi` 补一条正例 + 三条 `expected-error`。
4. **一次文档收口**：问题 1 的两处记录、`benchmark.md` 的两种解释、`cpp-managed-add.cpp` 与
   `cpp-softmax.mlir` 里与决策相反的注释、`runtime.md` 的 trace 格式与 D 编号、driver 里两处
   `tensor.empty` 注释。

这份文件只覆盖增量。合并前还要一起看姊妹审查开头那张汇总表：那里在 finish 上仍然存在的 6 条
（driver 的 `cast`、ABI 的 stride、ABI 没有用例、dispatch 静默跳过、span 长度宽度、`skeleton++`
的临时目录）与本文的问题 4 是同一批动作。
