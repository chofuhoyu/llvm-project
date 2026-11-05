# C++ 前端与算子实现：决策记录

> 2026-11-05 · 实现 [`operators.md`](../plan/operators.md) 与
> [`cpp_codegen.md`](../plan/cpp_codegen.md) 的过程中遇到的、这两份文档没有写到的取舍：
> 每个选择记录「选了什么 / 为什么 / 放弃了什么 / 代价」。方向仍以
> [`roadmap.md`](../plan/roadmap.md) 为准；运行时的设计在
> [`runtime.md`](runtime.md)。

## 1. 边界 API

### 1.1 数据参数一律按引用传

**决策**：C++ 侧的三种数据参数都写成引用：
`const T (&a)[N]`（编译期维）、`const skeleton_detail::span<T> &`（运行时长度）、
`T &`（单个元素）。`cpp_codegen.md` §4.3 的示例把 span 按值传，实现改成按引用。

**为什么**：CIR 把按值传递的 aggregate 在函数边界上拆成若干标量参数——一个
`span<const float>` 变成 `(!cir.ptr<!cir.void>, !u64i)` 一对——然后在函数体里用一段
coercion 序列（alloca、store、bitcast、load、重新取成员）把它拼回结构体。后果是参数类型
里不再有 span 的类型：元素类型和「这是一段带长度的数据」这件事只能靠回溯那段 sequence
才能拿回来，而那段序列是 clang codegen 的实现细节，CIR 还在上游化，形状不能建在它上面。

按引用时参数类型就是 `!cir.ptr<!cir.struct<"skeleton_detail::span<...>">>`，元素类型和
成员角色都在类型里，切分 pass 直接读，和旧实现（`hostParamToMemRef`）是同一套判断。

**放弃**：按值传 span。写法更自然、和 `std::span` 的习惯一致，代价是把 ABI 建在 CIR 的
aggregate 拆分规则上。

**代价**：调用点必须写 `const span<T> &`；按值传的 span 会退化成「不是数据参数」并报错，
错误信息里说明了这一点。

### 1.2 目的地在先，且没有返回值

**决策**：`skeleton_map(fn, out, ins...)`、`skeleton_reduce(fn, out, in, init)`。
第一个数据参数永远是目的地，算子把结果写进去，不返回 tensor。

**为什么**：所有权明确（缓冲由调用方分配，单元不分配不释放），动态长度有来源（span 的
size 字段 / 数组类型的维），产物与 driver 两条路径行为一致。返回值风格还要求 pass
「发明」返回类型（旧实现的 `outputTensorType`），而这是算子语义知识；现在结果形状直接来自
目的地参数的类型，pass 不需要再猜。

**代价**：`skeleton.map` 的 `outs` 与函数返回值不再对应；纯 MLIR 用例仍然用
`skeleton.map ... -> tensor`（方言本身没变），两种风格在单元内部统一成 `outs`。

### 1.3 裸指针拒绝，标量按值，标量目的地用 `T &`

**决策**：
- `T *` 作为数据参数被拒绝（输入和目的地都是）。
- 标量参数（`float init`、`int n`）按值传，进单元后是标量参数。
- 归约的标量结果写成 `float &out`（单个元素，`memref<f32>`）。

**为什么**：`T *` 和 `T &` 在 CIR 里是同一个类型（`!cir.ptr<T>`），没法区分「一段长度未知
的数组」和「一个元素」。位置是唯一的判据：目的地只可能是后者，所以只有第一个数据参数允许
裸指针，其余位置报错并指向 span。这样既保住了「裸指针没有长度就不许当数组用」，又让归约能
用最自然的写法写出标量输出。

**放弃**：为标量额外造一个 `skeleton_detail::scalar<T>` 包装类型；或让归约把结果写成
`T (&out)[1]`。前者多一个用户要学的类型，后者在 C++ 里很别扭。

### 1.4 目的地的形状就是结果的形状

**决策**：切分 pass 不再从算子语义推导结果形状，而是直接读目的地参数的类型。

**为什么**：同一个 `skeleton_map` 既可能是逐元素、也可能广播；同一个 `skeleton.reduce`
既可能全归约、也可能只归约尾部若干维。结果形状写在调用点的类型里，pass 不需要知道算子语义，
用户也能一眼看出算子产出什么形状。

## 2. 单元与 ABI

### 2.1 一个单元 = 一个骨架算子（v1 维持）

**决策**：一个 host 函数里只能有一次骨架调用；多个算子写成多个 host 函数，由 `main` 串起来。

**为什么**：`operators.md` §2 的扁平 DAG 正好是这个形状（大图不是大单元），而一个函数里放
多个骨架算子需要 pass 逐语句改写、并为中间结果分配缓冲。后者是 M5 的工作，不是这条线的
验收前提。

**代价**：真实算子要拆成 4 个 host 函数（softmax 就是这样），用户多写几行；换来的是每个
算子独立编译、独立放置。

### 2.2 wrapper 保留符号，另起一个 thunk

**决策**：host 函数保留符号和 C++ 签名，函数体改写成对
`__skeleton_op_<id>_shim(ptr, i64..., <scalar>...)` 的调用；thunk 的声明由切分 pass 产出，
函数体在 host 降到 LLVM 之后由 `skeleton-lower-host-abi` 填。

**为什么**：两边各自只做自己懂的事。C++ 参数 → 数据指针和长度是**源码层**的事（
`cir.get_member`、`cir.load`，布局由 CIR→LLVM 负责）；指针+长度 → memref 描述符是
**LLVM/MLIR ABI** 的事（`{allocated, aligned, offset, sizes..., strides...}`）。
如果让一个 LLVM pass 从 C++ 参数里读 span 成员，它就得重复一份结构体布局知识；
如果让 CIR 去构造描述符，它又没有 memref 类型。

**放弃的备选**：
- 把 host 函数本身当 shim（清空 body、由 LLVM pass 重建）——同样的布局耦合问题；
- 让单元直接收 `(ptr, length)`，在单元内部造 memref——需要用 `llvm.insertvalue` 拼描述符
  再 `unrealized_conversion_cast` 回 memref，bufferization 对「来历不明的 memref」没有保证。
  现在的单元签名是普通的 memref 参数，和手写 MLIR 用例完全一致，bufferization 行为已知。

### 2.3 元数据挂在 thunk 声明上，靠 `cir-to-llvm` 带过去

**决策**：`skeleton.host_abi = {op_id, unit, operands = [memref 类型...]}` 放在 thunk 的
`cir.func` 上。`CIRToLLVMFuncOpLowering::lowerFuncAttributes` 会原样拷贝非专属属性，
所以 ABI pass 在 `llvm.func` 上仍然读得到。

**放弃**：放进模块级表（`cir-to-llvm` 会重写模块属性）；靠函数名编码（类型信息放不下）。

### 2.4 描述符只在 ABI pass 里构造

- `allocated` 与 `aligned` 都取调用方给的指针，`offset` 为 0；
- 静态维与行主序步长是编译期常量；动态维从 thunk 的 `i64` 参数来；
- 只支持「动态维是最后一维」。span 是一维动态，数组引用全静态，所以前端永远产生不出别的
  形状；真出现了（例如二维动态）就报错，而不是生成一段把步长算错的代码；
- 生成调用之前，用单元的 `LLVMFunctionType` 逐参数比较描述符类型；不一致就编译期报错。
  这个检查是 `cpp_codegen.md` §4.3 要求的「两条路径共用一个签名推导」，现在推导只有一份。

**代价**：ABI 是「memref 描述符按字段摊开传」这一 MLIR 内部约定的复制。约定变了
`memrefDescriptorTypes` 要跟着改，测试会在编译期发现（不是运行期）。

### 2.5 thunk 发直接调用，dispatch pass 再改成 resolve + 间接调用

**决策**：ABI pass 生成的 thunk 体里是对单元的普通 `llvm.call @__skeleton_op_N`；紧接着
`layout` 的 `skeleton-lower-dispatch` 把它改写成 `skeleton_rt_resolve` + 间接调用。

**为什么**：放置决策只有一处（dispatch pass），ABI pass 不知道运行时存在；产物的启动注册
仍然拿单元的地址，两边不冲突。

## 3. 方言与算子形状

### 3.1 `skeleton.map` 的广播规则：低秩操作数对齐**前导**维

**决策**：秩 r < R 的输入匹配结果的 d0..d(r-1)；秩 0 的输入广播到所有元素。

**为什么**：唯一需要广播的来源是归约结果，而归约降的是尾部维，结果的下标正好是输入的前导
维。选前导对齐让「归约 → 逐元素」这条边不需要 reshape。

**放弃**：NumPy 式的尾部对齐。它需要把 `float[3]` 结果 reshape 成 `[3][1]` 才能用，
DAG 里就得多一个一维变二维的张量算子。

### 3.2 `skeleton.reduce` 降的维 = 输入有、目的地没有的尾部维

**决策**：目的地秩 k、输入秩 n 时降 `[k, n)`。

**为什么**：`reduce` 的 `outs` 本来就要有结果形状，用它表达「保留哪些维」不需要新属性；
`linalg.reduce` 的 `dimensions` 直接由它算出来。秩不变就是「不降」，秩 0 就是全降。

**放弃**：给 `skeleton.reduce` 加 `dimensions` 属性。属性与 `outs` 形状会互相矛盾，得加校验。

### 3.3 归约的 identity 显式传入

**决策**：`skeleton_reduce(fn, out, in, init)`，`init` 是标量，单元里先 `linalg.fill` 进
目的地，再原地归约。

**为什么**：旧实现从零开始归约，对 max / min / product 是错的；而算子不可能知道用户的
identity。显式传入同时也让 identity 的来源唯一（调用点），不需要 pass 再造常量。

## 4. 编译器集成

### 4.1 `cir-to-llvm` 之后必须剥掉 CIR 属性

**决策**：新增 `cir-strip-cir-attrs`，把名字以 `cir.` 开头、或值里提到 CIR 方言/类型的属性
全部删除。

**为什么**：`cir-to-llvm` 之后的操作上还留着 `"cir.target-features"` 之类的字符串属性和
`func_info = #cir.cxx_ctor<...>` 这类 CIR 方言属性。它们引用了 CIR 方言和 CIR 类型，
**不注册 CIR 方言的工具（driver、mlir-opt）连模块都解析不了**——这不是「不干净」，是硬阻塞。
旧实现靠切分 pass 里的 `dropCirModuleAttrs` 回避，但那只处理模块属性，而且混合模块还要靠
这些属性做 CIR→LLVM 与 data layout，不能提前删。

**放弃**：把 strip 放进 `populateCIRToLLVMPasses`。那会改变 clang 自己 `-emit-llvm` 的输出
属性，牵动 CIR 测试；现在是前端管线里显式的一步。

### 4.2 `cir-opt` 要多注册方言

**决策**：`cir-opt` 注册 func / arith / tensor / bufferization / linalg / scf / math / cf /
index。

**为什么**：切分之后模块里同时有 CIR、`func.func` 单元、`bufferization.to_tensor`、
thunk 声明。`cir-opt` 原来只认识 CIR + memref + LLVM，连读回自己的输出都不行。
只注册方言、不注册转换 pass：前端管线不需要它们。

### 4.3 一条命令用 `skeleton++` 脚本，不先改 clang driver

**决策**：`mlir/tools/skeleton++` 是个 CMake 配置出来的 shell 脚本，串起 clang++ →
cir-opt → driver。文档 §W6 建议先做包装工具，这里照做。

**理由**：三个工具都已经是完整命令，脚本把管线摆在明面上，改起来不用重编译；clang driver
的 JobAction 要动 `clang/lib/Driver`，并且和 `-cc1` 的参数、临时文件、链接器调用纠缠，收益
只是「少一个工具」。等 API 与 ABI 稳定后再做（见 §6）。

**代价**：`skeleton++` 不是编译器的一部分，toolchain 换路径时要重新 configure。

### 4.4 入口放宽到 `llvm.func`，产物 `main` 返回程序退出码

**决策**：driver 与产物都接受两种入口；C++ 的 `int main()` 在产物里被改名成
`__skeleton_entry`，生成的 `main` 调它并返回它的返回值。

**为什么**：CIR→LLVM 之后 `main` 是 `llvm.func`，driver 原来只找 `func.func`；另外
`int main` 返回 int，原来的产物 `main` 一律返回 0，把程序自己的退出码丢了。

### 4.5 driver 只在有 tensor 级工作时才跑 bufferize

**决策**：`hasTensorLevelOps` 检查模块里还有没有 skeleton 算子或 tensor 类型的值；没有就跳过
`skeleton-to-linalg` + bufferization。

**为什么**：C++ 前端的 host 已经是 LLVM 方言，让 one-shot bufferize 去处理一个
LLVM-only 模块只是给编译器找麻烦；纯 MLIR 用例与单元编译仍然走原路径。

## 5. 第一个真实算子：softmax

### 5.1 选 softmax

候选是 softmax / layernorm / 单个 attention 头。选 softmax：它同时需要归约、逐元素、
广播和标量常量，正好检验 §3.1–3.3 的三条形状规则；layernorm 还要求对同一输入做两次归约
（均值、方差），attention 头要多一个矩阵乘。

### 5.2 写成扁平 DAG，不新增 `skeleton.softmax`

**决策**：`rowmax → shift_exp → rowsum → normalize` 四个 host 函数，各一个单元。

**为什么**：`operators.md` §2 说得很明确——大算子是大图。新增一个融合的
`skeleton.softmax` 会让放置粒度变成整个 softmax，与 op 级放置冲突，也绕开了「现有 map /
reduce 够不够」这个问题。实现之后答案是：够，但需要补广播和尾部维归约。

### 5.3 行内 softmax 用二维静态数组，动态形状只在 CPU

**决策**：测试用 `float[3][4]`（静态），因此同时能跑 GPU；span 版本（动态）只跑 CPU。

**为什么**：GPU 桥接仍然拒绝动态形状的 operand（roadmap §5.4 的独立工作）。要么把第一个
算子做成静态的，要么就接受它跑不了 GPU；验收要求 CPU/GPU 各一次，所以选静态。

## 6. 没做与遗留

- **clang driver `-fskeleton`**：见 §4.3，等 API 稳定。
- **一个 host 里多个骨架算子**：v1 明确报错（M5）。
- **动态形状 GPU**：仍在 `skeleton-target-lower` 里被拒绝。
- **扁平图上的融合 pass**：没开始；现在的每一步中间结果都落内存。
- **错误处理**：`skeleton_rt_resolve` 失败仍然 `exit(1)`；C++ 侧还没有可恢复的错误约定。
- **`std::cout` / 静态对象**：JIT 路径不跑 `llvm.global_ctors`，v1 只用 `printf`。
- **纯函数词汇**：直线标量代码 + 比较 + ternary + 算术 cast + math builtin。调用、
  循环、多 block 仍然报错。
