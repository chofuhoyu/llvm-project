# Skeleton Dialect 全流程打通：从 C++ 注解到后端代码生成

> 口头汇报用，语气口语化

---

## 一句话概括

这次提交做的事情就是：**让程序员在 C++ 代码里写个注解（annotate），编译器就能自动识别出来，最终生成对应硬件的加速代码**。我们把从 C++ 源码注解到 Skeleton dialect 的整条链路打通了。

---

## 背景

Skeleton dialect 是我们项目里设计的一个 MLIR 方言，核心思路是：**程序员在源码层面标注"这段计算我想跑在什么硬件上"，编译器拿到这个偏好之后，自动在对应的 target 上生成代码**。

之前 Skeleton dialect 已经有了基本框架（`skeleton.custom_matmul`、`skeleton.preference` 等），但缺了一个关键环节：**从 C++ 源码到 Skeleton dialect 的整条链路是断的**。之前只能在 MLIR 层面手写 `.mlir` 文件来用 Skeleton，没办法从正常的 C++ 程序出发。这次提交把这个缺口补上了。

---

## 整体架构：三段式流水线

```
C++ 源码（带注解）
    │
    ▼  Phase 2: Clang → ClangIR
ClangIR（cir.for / cir.while 带 annotation 属性）
    │
    ▼  Phase 3: CIR → Skeleton
Skeleton dialect（skeleton.annotate）
    │
    ▼  Phase 1: Skeleton → Linalg（已有 + 增强）
Linalg → 后端代码
```

下面按阶段展开。

---

## Phase 1：Skeleton Dialect 本身的增强

### 1.1 新增 `skeleton.vector_add` op

给 Skeleton dialect 加了逐元素向量加法操作，C[i] = A[i] + B[i]。

为什么加这个？矩阵乘（matmul）太复杂，向量加法简单、好测试，适合验证整个流程。

这个 op 带一个可选的 `preference` 属性，用来标注目标硬件：

```
%0 = skeleton.vector_add preference = #skeleton.preference<"CPU">
    ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
    outs(%out : tensor<8xf32>) -> tensor<8xf32>
```

### 1.2 新增 `skeleton.annotate` op

一个标记操作（marker op），本身不做计算，只携带元数据——来自 C/C++ 源码的 annotation name 和 preference。

它的角色是 **CIR 前端注解和最终 Skeleton 计算 op 之间的桥梁**。CIR 那边的 pass 先把带注解的循环转成 `skeleton.annotate`，后面的 pass 再根据 annotate 里的名字去生成具体计算 op（比如 `skeleton.vector_add`）。

设计成两步而不是一步到位，是为了解耦——前端注解形式可以多样化，Skeleton 内部只需要认 `skeleton.annotate` 这一种标记。后面支持新注解类型时只改后面的 pass，不用动前端。

### 1.3 `vector_add` 降级到 Linalg

在 `skeleton-to-linalg` pass 里加了 `ConvertVectorAdd` rewrite pattern，把 `skeleton.vector_add` 转换成 `linalg.add`，同时保留 `preference` 属性。

### 1.4 修了一个 round-trip bug

`PreferenceAttr` 里原来用 `std::string` 存 preference 值，MLIR 体系里应该用 `StringAttr`。这会导致打印再解析（round-trip）时出错，已修复。

### 1.5 加了集成测试

`test-skeleton-vector-add-e2e.mlir` 和 `test-skeleton-vector-add.mlir`，验证从 Skeleton dialect 到 Linalg 的 lowering 链路。

---

## Phase 2：C++ 注解 → ClangIR

这是让信息从源码进入 IR 的关键环节。

### 2.1 在 CIR 循环操作上支持 annotation

ClangIR（CIR）里有三种循环：`cir.for`、`cir.while`、`cir.do`，都继承自 `CIR_LoopOpBase`。

在 `CIR_LoopOpBase` 上加了一个 `OptionalAttr<CIR_AnnotationArrayAttr>`，三种循环自动全部继承。

### 2.2 从 Clang AST 转发注解

Clang 前端本身支持 `__attribute__((annotate("xxx")))`，解析出来是 AST 上的 `AnnotateAttr` 节点。但之前 CIRGen 在遇到 `AttributedStmt` 时，没有把注解信息传到生成的 CIR 操作上——旧代码里有行注释 `// TODO: pass in an array of attributes`，说明这个缺口是已知的。

这次把这个链路补上了：在 `emitAttributedStmt` 里，遇到 for/while/do-while 循环时，把 `AttributedStmt` 上的注解属性传下去；然后在 `emitForStmt`/`emitWhileStmt`/`emitDoStmt` 末尾调用新的辅助函数 `attachLoopAnnotations`，把注解附着到生成的循环 op 的 `annotations` 属性上。

这样，程序员写的：

```c
__attribute__((annotate("skeleton_vector_add", "CPU")))
for (int i = 0; i < N; ++i)
    C[i] = A[i] + B[i];
```

就能在生成的 CIR 里看到 `cir.for` 上带有 `cir.annotation<"skeleton_vector_add", ["CPU"]>` 属性。

### 2.3 暴露 annotation 辅助函数

`emitAnnotateAttr` 和 `getOrCreateAnnotationArgs` 从 CIRGenModule 的私有方法改成 public，方便其他地方复用。

---

## ⭐ 深度分析：为什么语句级 annotation 只能在 CIR 层传递？

CIRGenStmt 这个改动虽然代码量不大，但背后涉及编译器架构层面的差异。把这个问题讲清楚，才能理解为什么 CIR 在这条链路里是不可替代的。

### 第一层：传统 Clang → LLVM IR 路径——annotation 在第一关就被丢了

先看源码。传统 Clang CodeGen 的语句处理入口在 `clang/lib/CodeGen/CGStmt.cpp`。当 C++ 源码里 `__attribute__((annotate("xxx")))` 包在 for 循环外面，AST 的结构是：

```
AttributedStmt
  └── ForStmt
```

dispatch 会走到：

```cpp
// CGStmt.cpp 第 526-527 行
case Stmt::AttributedStmtClass:
    EmitAttributedStmt(cast<AttributedStmt>(*S));
    break;
```

然后 `EmitAttributedStmt`（第 776 行起）用一个 switch 来分派属性处理：

```cpp
void CodeGenFunction::EmitAttributedStmt(const AttributedStmt &S) {
    for (const auto *A : S.getAttrs()) {
        switch (A->getKind()) {
        default:
            break;          // ← AnnotateAttr 走到这里，什么都不做
        case attr::NoMerge:       // 处理 nomerge
        case attr::NoInline:      // 处理 noinline
        case attr::AlwaysInline:  // 处理 alwaysinline
        case attr::NoConvergent:  // 处理 noconvergent
        case attr::MustTail:      // 处理 musttail
        case attr::CXXAssume:     // 处理 assume
        case attr::Atomic:        // 处理 atomic
        case attr::HLSLControlFlowHint:  // 处理 HLSL
            ...
        }
    }
    EmitStmt(S.getSubStmt(), S.getAttrs());  // 递归进入内部 for
}
```

**关键事实：switch 里没有 `case attr::Annotate`。** `AnnotateAttr` 走到 `default: break`，直接丢弃。然后递归进入内部的 ForStmt，ForStmt 的 emit 函数也不接收任何 annotation 参数——注解信息在第一关就没了。

为什么会这样设计？因为 `__attribute__((annotate))` 在 Clang 生态里的原始定位是标注**声明**（变量、函数、结构体成员），不是标注**语句**。对应的 LLVM 发射路径是 `EmitVarAnnotations` / `EmitFieldAnnotations`（`CodeGenFunction.cpp` 第 2806/2814 行），底层用的是 `llvm.var.annotation` 和 `llvm.ptr.annotation` 这两个 LLVM intrinsic——它们的语义是"给某个内存位置/指针打标签"，跟循环这种控制流结构毫无关系。语句级 annotation 从一开始就不在 Clang 的设计范围内。

### 第二层：LLVM IR 的结构问题——循环根本不存在

退一步讲，即使 Clang 想传递语句级 annotation，LLVM IR 的形态也会让这件事变得不可行。因为 LLVM IR 里**没有"循环"这个结构化概念**。

一个 `for (int i = 0; i < N; ++i) { body; }` 经过传统 Clang CodeGen 后不产生任何"循环对象"，而是变成一组**基本块**：

```
entry:
    ...
    br label %for.header

for.header:                            ← phi 结点 + 条件判断
    %i = phi i32 [ 0, %entry ], [ %inc, %for.latch ]
    %cmp = icmp slt i32 %i, %N
    br i1 %cmp, label %for.body, label %for.exit

for.body:
    ...
    br label %for.latch

for.latch:                             ← 递增值 + 跳回 header
    %inc = add i32 %i, 1
    br label %for.header

for.exit:
    ...
```

问题是：

1. **循环不是一个 IR 实体**——循环是 4-5 个基本块靠分支指令连起来的拓扑结构。LLVM 里的 `LoopInfo` 分析是后验推断出来的，不是 IR 本身的结构。
2. **annotation 没有挂载点**——该挂在哪？`for.header` 的 phi 上？`for.latch` 的 branch 上？`for.body` 的第一条指令上？无论挂在哪，都是循环的局部片段，语义不完整。
3. **优化 pass 会移动/消除基本块**——循环展开、循环旋转、基本块合并等优化会让 annotation 位置变得毫无意义。
4. **LLVM 的 `!llvm.loop` metadata 不是通用解决方案**——它是专门为循环优化参数设计的（`llvm.loop.unroll.count`、`llvm.loop.vectorize.enable` 等），挂在 latch 的 branch 上，语义固定。它不是为"传递程序员的任意 annotation"设计的，整个 LLVM 优化管线也不会帮你维护自定义元数据的语义。

### 第三层：CIR 为什么能做到——循环是一个 op

CIR（ClangIR）的设计跟传统路径不同。在 CIR 里，一个 for 循环是**一个单体 op**：

```
cir.for {
    // 初始化
} condition {
    // 条件判断 → %cond
    cir.yield %cond
} body {
    // 循环体
} step {
    // 递增值
}
```

核心差异在于：

1. **循环是 IR 级别的实体**——`cir::ForOp` 是一个 MLIR 操作，有自己的 location、region、attributes。不是一个"被推断出来的概念"，而是一个真实存在于 IR 中的操作。
2. **annotation 是 op 的 attribute**——我们在 `CIR_LoopOpBase`（三种循环的基类）上加了 `annotations` 属性，它就是 op 的一部分，和 op 一样持久。后续 pass 随时可以 query。
3. **annotation 跟着 op 走**——从 CIR 到 Skeleton 到 Linalg，只要 op 还在，属性就在。只有当某个 pass 明确把结构化循环"炸开"成非结构化控制流时，annotation 才会被消费并移除——但那时它的使命已经完成。

用一句话总结：

> **LLVM IR 里循环是推断出来的抽象概念；CIR 里循环是 IR 层面的实体。annotation 只能挂在实体上，不能挂在"推断出来的概念"上。**

### 这意味着什么？

这意味着语句级 annotation 传递**不是"选择在 CIR 做"的问题，而是"天然只能在 CIR 做"的问题**。

| 层级 | 能否挂 annotation | 原因 |
|------|------------------|------|
| Clang AST | ✅ 可以 | `AttributedStmt` 是结构化 AST 节点 |
| CIR | ✅ 可以（本次实现） | `cir.for` 是 op，有 attribute 体系 |
| LLVM IR | ❌ 不行 | 循环被拆散成基本块图，没有挂载点 |
| MLIR (其他 dialect) | ✅ 可以 | `scf.for` / `affine.for` 同样是结构化 op |

从 AST 到 MLIR 的 `scf.for`，CIR 的 `cir.for` 是中间的桥。如果不过 CIR，要直达 `scf.for`，那就是 Clang → MLIR 直接生成，这条路径在社区目前不存在。传统 Clang → LLVM IR 路径上，语句级 annotation 注定会被丢掉。

### 我们实际做了什么

代码改动核心就两步：

1. 在 `CIR_LoopOpBase` 上加了 `annotations` 属性，三种循环全部继承。
2. 在 `emitForStmt`/`emitWhileStmt`/`emitDoStmt` 末尾调用 `attachLoopAnnotations`，把 AST 层的 `AnnotateAttr` 翻译成 CIR 层的 `cir.annotation`。

改动不大，但补上了传统路径上"语句级 annotation 无法进入 IR"这个结构性的缺口。

---

## Phase 3：CIR → Skeleton 的转换

### 3.1 新增 `cir-loop-to-skeleton` pass

新文件 `CirLoopToSkeleton.cpp`，约 186 行。

逻辑：遍历所有 CIR 操作，找到带 `cir.annotation` 属性的循环，在循环前面插入 `skeleton.annotate` 标记。

两个层次：

- **信息提取**：`extractLoopInfo` 从 CIR 循环 op 上提取 annotation name 和 preference。例如 `cir.annotation<"skeleton_vector_add", ["CPU"]>` → name="skeleton_vector_add"，preference="CPU"。没有 preference 时默认 "CPU"。
- **标记生成**：三个 rewrite pattern（`ConvertAnnotatedForOp`/`ConvertAnnotatedWhileOp`/`ConvertAnnotatedDoWhileOp`）分别在循环前插入 `skeleton.annotate`。同时保留了一个 walk 方式的 fallback 实现。

### 3.2 注册 & 编译

- 在 `cir-opt` 工具里注册了 SkeletonDialect 和该 pass。
- `CMakeUserPresets.json` 里加了 `CLANG_ENABLE_CIR=ON`（CIR 默认关闭）。

---

## 测试情况

7 个测试全部通过：

| 测试 | 验证内容 |
|------|---------|
| `annotate-loop.cpp` | Clang → CIR 注解传递，覆盖 for/while/do-while 三种循环及无注解情况 |
| `annotate-attribute.{c,cpp}` | C 和 C++ 模式下的注解属性 |
| `cir-loop-to-skeleton.mlir` | CIR 到 Skeleton 转换：有注解生成 `skeleton.annotate`，无注解不受影响 |
| `skeleton-to-linalg.mlir` | `skeleton.vector_add` → `linalg.add` 降级 |
| `skeleton-preference-partition.mlir` | preference 属性 round-trip 修复 |
| `skeleton-target-lower.mlir` | 已有测试，确保不被破坏 |
| `skeleton-finalize-memref-to-gpu.mlir` | 已有测试，确保不被破坏 |

---

## 总结：完整流程

用程序员视角走一遍：

1. 程序员写带注解的 for 循环：
   ```c
   __attribute__((annotate("skeleton_vector_add", "CPU")))
   for (int i = 0; i < N; ++i)
       C[i] = A[i] + B[i];
   ```

2. Clang 前端解析出 `AttributedStmt` + `AnnotateAttr`（已有能力）。

3. CIRGen 在 `emitForStmt` 中调用 `attachLoopAnnotations`，把注解写成 CIR 循环 op 的 `annotations` 属性。

4. `cir-loop-to-skeleton` pass 发现带注解的循环，插入 `skeleton.annotate`。

5. 后续 Skeleton pass（待实现）可根据 `skeleton.annotate` 生成 `skeleton.vector_add`。

6. `skeleton-to-linalg` pass 把 `skeleton.vector_add` 转换成 `linalg.add`。

7. MLIR 标准流程：Linalg → Affine/SCF → LLVM IR → 机器码。

---

## 下一步工作

这次搭好的是"骨架"。已完成：

- ✅ C++ 源码 annotation → CIR 循环属性（信息不丢失）
- ✅ CIR 循环属性 → `skeleton.annotate` 标记
- ✅ `skeleton.vector_add` → `linalg.add` lowering

待完成：

- [ ] `skeleton.annotate` → 具体计算 op（如 `skeleton.vector_add`）的自动生成 pass
- [ ] `skeleton.preference` 驱动后端代码生成的 pass（GPU 偏好 → GPU kernel，CPU 偏好 → CPU 向量化等）

**整体上，从 C++ 源码到 MLIR Skeleton 层的整条信息传递链路已经跑通了，架构验证目标达成。**
