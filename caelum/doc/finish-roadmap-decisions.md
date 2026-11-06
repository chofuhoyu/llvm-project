# 完成 roadmap：决策记录

> v1 · 2026-11-06
>
> `plan/roadmap.md` 记方向，`plan/finish-roadmap-approach.md` 记差距与工作包
> （WP0–WP9、E1–E11）。本文件只记实现这些工作包时**需要权衡的地方**：背景、选择、
> 理由，以及放弃了什么。每条决策都对应代码或文档里的一处具体形状，不重复计划本身。

## 1. 验证口径

### D1 CIR 测试过滤器按“谁驱动切分”而不是按名字选

**背景**：远端 CIR 过滤器是 `(skeleton|cir-call-to-skeleton|cir-loop-to-skeleton)`。
后两个名字对应的 pass 随半自动路径一起删除了，`cir-func-to-arith.cir` 驱动
`cir-skeleton-split` 却因为文件名不含 `skeleton` 而长期没跑过（差距 G9）。

**选择**：过滤器改成 `(skeleton|cir-func-to-arith)`，并在 `.profile/caelum/AGENTS.md`
写明这条规则的意图，以及“新增测试要检查文件名是否落在规则里”。

**理由**：过滤器是按文件名匹配的，而文件的命名依据是它测的 pass 还是它测的东西并不统一；
把“规则要覆盖哪几类用例”写下来，比再补一个名字更能防止下一次漏测。

**放弃**：改成按目录或按内容选择（例如所有引用 `skeleton` 的 `.cir`）。那需要遍历文件
内容，脚本复杂度上升，而当前只有三份用例。

## 2. GPU 正确性

### D2 归约的 destination 用可写视图，不加新的 lowering

**背景**：`GPU/cpp-softmax.mlir` 是 `XFAIL`，注释把原因写成“归约的组合器被当成独立迭代
映射到线程”。实测不是：`linalg.reduce` 的迭代类型是 `[parallel, reduction]`，
`generateParallelLoopNest` 已经把它降成“外层 `scf.parallel`、内层 `scf.for`”，设备上
每个输出元素一个线程、归约维串行，本来是对的。

真正的原因是切分为归约单元时，destination 的 `bufferization.to_tensor` 与输入用同一个
构造函数，带 `restrict` 且**不可写**。one-shot bufferization 因此不能在 destination 上
写：它分配临时缓冲、在临时里做 fill 和归约、再把结果复制回 destination。CPU 上这只是白
做功，设备上则是内核写宿主内存——`compute-sanitizer` 报的正是内核读了一个宿主地址。

**选择**：`createToTensor` 增加 `writable` 参数，归约的 destination 用可写视图；视图仍然
是 `restrict`（one-shot analysis 拒绝非 restrict 的 `to_tensor`）。

**理由**：这是最小改动，且让 CPU 路径也少一次分配加一次复制。归约的语义本来就要求累加器
在 destination 里（identity 先写进去），可写视图是把已经在文档里写明的约定真正交给
bufferization。

**放弃**：
- 给 GPU 归约写一套共享内存树形归约。正确性上没有必要，性能上目前没有基线支撑；先让
  正确性成立，树形归约留给 benchmark 显示它是瓶颈的时候。
- 让 destination 复用 `tensor.empty` + empty tensor elimination（map 的做法）。归约需要
  identity 先落在 destination 上，`tensor.empty` 表达不了这件事。

**代价**：`restrict` 仍然成立，意味着一个归约的输入与输出不能是同一块缓冲。这在原地归约里
本来就不安全（累加器在输入读完之前就被写），所以不是一个实际被放弃的能力。

### D3 同步点放在 GPU 单元调用之后，由运行时提供

**背景**：`mgpuLaunchKernel` 只调 `cuLaunchKernel`，不等待；host 在 `call` 返回后立刻
`memref.copy` 读回。`cuMemFree` 可能在拷贝之后同步，救不了这次读（差距 G3）。

**选择**：运行时增加 `skeleton_rt_sync(int32_t device)`，`device == CPU` 时是空操作；
`skeleton-target-lower` 在每个 GPU 单元的调用之后、第一次 copy-back 之前插入一次调用。
驱动把该符号注册进**单元自己的** JIT 引擎（等待发生在单元模块里），不只是 host 引擎。

**理由**：同步的语义属于运行时（它知道设备怎么来），而调用点属于 lowering（它知道哪次
调用前面有没有内核）。放在 `mgpuLaunchKernel` 里改成同步会把所有内核都串行化，也改不到
上游的运行库；放在 host 侧 wrapper 里则需要 host 知道设备。

**放弃**：`gpu.wait` / event。桥接路径没有 async token，引入 token 会牵动整条
`gpu.alloc`/`gpu.dealloc` 链，收益只是省一次上下文切换。

### D4 managed 单元跳过桥接，而不是让桥接变聪明

**背景**：roadmap §2.5 要求“分配交给运行时之后，现行桥接的 host 拷贝可以去掉”（差距 G2）。
难点是“这块内存是不是设备可见”无法从 IR 判断——同一个单元可能被宿主缓冲、也可能被运行时
缓冲调用。

**选择**：新增 `skeleton_detail::buffer<T>`，宿主函数的数据参数**全部**是它时，单元带
`skeleton.managed`，`skeleton-target-lower` 对该单元的调用完全跳过 `gpu.alloc` /
`memref.copy` / `gpu.dealloc`。混用 buffer 与数组/span 的宿主在切分阶段报错。

**理由**：managed 是单元的属性，不是单个操作数的属性，因为单元只编译一次。要求“全有或
全无”让属性可以被无条件信任，也让错误在编译期暴露，而不是等到内核访问宿主内存才崩。
静态数组的宿主继续走桥接，一条已通过的路都没动。

**放弃**：
- 逐操作数标记（在 `skeleton.host_abi` 里记每个参数的来源）。它允许混用，但会让
  `target-lower` 需要逐参数决定要不要桥接，而桥接是成组的（alloc/copy/dealloc 三个一组、
  copy-back 依赖 call 的位置），拆开写的复杂度远高于它带来的写法自由。
- 完全删掉桥接。手写 MLIR 的用例和静态数组的 C++ 用例都依赖它。

### D5 运行时缓冲是一维、运行时长度

**背景**：前端的形状来自参数类型：数组引用给静态维，`span` 给一维动态。managed 缓冲需要
一个“存储来自运行时”的类型，而它的长度天然是运行时值。

**选择**：`buffer<T>` 只有两个成员（`data`、`size`），语义与 `span` 完全相同，因此边界上
是 `memref<?xT>`。它带来两个结果：managed 路径只支持一维动态形状；二维形状的算子继续用
静态数组 + 桥接。

**理由**：
- 一维动态正好是 roadmap §5.4 里“前端能产生的形式”（差距 G5），一个改动同时关掉 G2 与
  G5 的缺口，不需要给 ABI 增加新形状。
- 多秩缓冲需要在类型里表达形状。要么从模板实参的名字里解析维数（`buffer<float, 3, 4>`），
  要么在结构里存秩和维度数组——前者脆弱，后者要 ABI 认识新布局。roadmap 的非目标里写着
  非 identity layout、多输出，“把形状硬塞进缓冲类型”属于同一类扩张，不做。

**代价**：softmax 这种二维算子仍走桥接路径（正确，但有拷贝）。基准测试因此分成两组：
静态数组（桥接）与运行时缓冲（managed），两组都能跑设备。

**放弃**：把 `buffer` 做成可变秩。见上。

### D8 CUDA 上下文自己建、自己管

**背景**：`skeleton_rt_alloc` 与 `skeleton_rt_sync` 都要一个 current CUDA context。
实现过程中发现这台机器的驱动拒绝 `cuCtxPushCurrent` 一个已经 current 的 context（返回
`CUDA_ERROR_INVALID_CONTEXT`），而 MLIR 的 CUDA 运行时 wrapper 每次调用都 push 一次它自己
retain 的 primary context——push 失败，析构里的 pop 却照常执行，把别人压栈的 context 弹掉。

**选择**：运行时在第一次需要时 `cuCtxCreate_v2` 建一个自己的 context（创建后即 current），
之后只在 current 不是它时才 push、用完 pop。alloc 与 sync 都走这一条路。

**理由**：managed memory 属于设备而不属于 context，所以自建 context 分配的内存，主 context
里加载并启动的 kernel 仍然可以访问（`cpp-managed-add` 验证过）。自建 context 让行为不再
依赖别人压栈的顺序。

**放弃**：只用 primary context。它在本机拿不到可用句柄（retain 成功但 handle 被 push 拒绝）。

### D9 探测不再读空闲显存

**背景**：`skeleton_rt_probe` 以前用 `cuMemGetInfo` 读空闲显存，供「放不下就回退 CPU」用。
上面的 context 顺序问题让这个读在每个 probe 上都失败（`compute-sanitizer` 记为 error），
而"读不到"本来就有一个既定含义：`free=-1`，跳过显存判断。

**选择**：probe 报告空闲显存未知，不再调用驱动。`skeleton_rt_set_free_bytes` 保留，
`driver-memory-fallback` 仍用强制值验证决策路径。

**理由**：一个每次都失败、且失败原因不在本模块控制范围内的驱动调用，只会制造噪音；
把它降级成"未知"是运行时已经支持的状态，代价诚实可见。

**代价**：生产路径上不再因为显存放不下而回退 CPU，只有在显式设置强制值时才会。这写进
`runtime.md` §8 与本文；要做真正的显存判断，需要一个能稳定取到 context 的探测实现。

## 3. 可观测性与形态

### D6 executor 只加只读字段，不做多 executor

**背景**：roadmap 的术语是 node → executor → device，决策落点是 executor 的 device；
但代码里只有 device 枚举（差距 G8）。

**选择**：运行时增加 `skeleton_rt_executor_id` / `skeleton_rt_executor_count`（v1 恒为
0 / 1），trace 的 resolve 行与 stats 行末尾追加 `exec=<id>`。

**理由**：v1 一个进程一个 executor，用它做行为分支没有意义；但日志里没有锚点，以后加多
executor 时无法从现有日志判断“当时是哪个执行器”。字段追加在行尾，现有测试的
`TRACE:` 前缀匹配不受影响。

**放弃**：给 provider 加 executor 参数。ABI 一旦带上它，多 executor 就成了必须支持的
形态，而 roadmap 明确说 v1 不做。

### D7 `target-lower` 保留为 MLIR pass

**背景**：roadmap §5.2 把“是否进一步收进 driver”列为未决（差距 G10）。

**选择**：保留。契约写进 pass 的 description：输入是带 `skeleton.target` 的单元，输出是
目标相关 IR；不做设备选择；GPU 目标的桥接与 managed 跳过都在这里，两种形状都由
`mlir/test/Dialect/Skeleton/skeleton-target-lower-*.mlir` 直接覆盖。

**理由**：driver 与单元测试共用它；收进 driver 意味着所有单元测试要改成 driver 级测试，
换来的只是少一个 pass 注册。行为已经稳定（桥接、managed、同步都在这一层），契约写清比
搬家有价值。
