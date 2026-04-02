# 1.2.4 错误处理与调试机制

LLVM的错误处理和调试机制是其工程实践的重要组成部分。这套机制不是简单的异常或断言，而是经过深思熟虑的多层次错误处理策略。理解这些机制，不仅能帮助我们正确使用LLVM，更能学习到大型C++项目的错误处理最佳实践。

## 致命错误处理：设计哲学的体现

LLVM将致命错误分为多个类别，这种分类反映了其对不同错误场景的深入理解。

### report_fatal_error：历史与演进

从 [ErrorHandling.h](/llvm/include/llvm/Support/ErrorHandling.h#L63) 可以看到，`report_fatal_error` 是LLVM最早的致命错误处理机制：

```cpp
[[noreturn]] void report_fatal_error(const char *reason,
                                      bool gen_crash_diag = true);
```

在早期的LLVM中，错误处理相对简单，选择快速失败，遇到不可恢复的错误时立即终止，避免级联错误，防止一个错误导致更多错误，同时简化调试，立即发现问题所在。但这个设计有一个问题，它不能区分不同类型的致命错误。

### 错误分类的必要性：为什么要拆分？

LLVM 15+将`report_fatal_error`拆分为两个函数，这不是随意的改动，而是基于实践经验的改进。这种拆分实现了责任明确，内部错误是LLVM开发者需要修复的问题，使用错误是用户需要解决的问题。这种拆分也改善了用户体验，内部错误生成崩溃报告，请求用户反馈，使用错误则优雅退出，不生成崩溃报告。最后，这种拆分提高了调试效率，内部错误的崩溃转储有助于开发者分析，使用错误则不需要崩溃转储，错误信息足够。

从 [ErrorHandling.h](/llvm/include/llvm/Support/ErrorHandling.h#L77) 和 [ErrorHandling.h](/llvm/include/llvm/Support/ErrorHandling.h#L95) 可以看到这个区别：

| 函数 | 错误类型 | 行为 | 用户反馈 |
|-----|---------|------|---------|
| `reportFatalInternalError` | LLVM内部bug | abort() + 崩溃转储 | "请报告这个bug" |
| `reportFatalUsageError` | 用户使用错误 | exit(1) + 普通消息 | "请检查你的输入" |

这种设计体现了几个重要的权衡。为什么不都用abort()？因为用户错误不需要也不应该生成崩溃报告。为什么不都用exit()？因为内部错误需要详细的调试信息来诊断。为什么引入新函数而不是修改旧函数？这是为了保持API兼容性，实现渐进式迁移。

### 自定义错误处理：可扩展性设计

LLVM允许安装自定义错误处理函数，这是一个重要的可扩展性点：

```cpp
void install_fatal_error_handler(fatal_error_handler_t handler,
                                  void *user_data = nullptr);
```

这种灵活性是必要的，因为不同场景有不同的需求。命令行工具可以直接输出到stderr，集成应用可能需要显示GUI对话框，服务器应用需要记录到日志系统，测试环境可能需要收集错误信息。

LLVM默认不使用异常，但允许用户通过自定义处理函数抛出异常，不过注释警告不要盲目抛出异常。从 [ErrorHandling.h](/llvm/include/llvm/Support/ErrorHandling.h#L36) 的注释可以看到这种谨慎：

> It is dangerous to naively use an error handler which throws an exception.
> Even though some applications desire to gracefully recover from arbitrary
> faults, blindly throwing exceptions through unfamiliar code isn't a way to
> achieve this.

这种谨慎是有原因的。LLVM代码不是为异常安全设计的，异常可能导致资源无法正确释放，异常可能在破坏不变量的位置抛出，异常也会使问题定位更复杂。

### llvm_unreachable：既是文档也是优化

[llvm_unreachable](/llvm/include/llvm/Support/ErrorHandling.h#L146) 是一个非常有趣的设计，它同时服务于多个目的：

```cpp
#define llvm_unreachable(msg) \
  llvm_unreachable_internal(msg, __FILE__, __LINE__)
```

这个设计同时扮演了多重角色。首先是文档角色，明确标记代码路径不可达，告诉读者这个分支不应该被执行，替代注释，更正式。其次是调试角色，在Debug模式下打印错误消息和位置，帮助捕获逻辑错误，是assert的增强版。最后是优化角色，在Release模式下告诉优化器这个路径不可达，允许优化器删除相关代码，可能启用更多优化。

**使用场景示例：**

```cpp
enum class Opcode { Add, Sub, Mul };

int compute(Opcode Op, int A, int B) {
  switch (Op) {
    case Opcode::Add: return A + B;
    case Opcode::Sub: return A - B;
    case Opcode::Mul: return A * B;
  }
  // 如果程序逻辑正确，这里永远不会到达
  // 如果到达了，说明有bug
  llvm_unreachable("Invalid opcode");
}
```

为什么不用assert(false)？因为llvm_unreachable语义更明确，专门标记不可达代码，优化器理解，优化器特别处理llvm_unreachable，并且提供更好的消息，自动包含文件名和行号。

## 可恢复错误处理：现代错误处理

LLVM的`Error`和`Expected`是其现代错误处理机制的核心。这套机制体现了函数式编程的影响，同时保持了C++的效率。

### Error的设计：为什么这样设计？

从 [Error.h](/llvm/include/llvm/Support/Error.h#L89) 可以看到`Error`的设计：

```cpp
class Error {
  ErrorInfoBase *Ptr;
  bool Checked;
  // ...
};
```

Checked标志是一个非常关键的设计点。这个标志确保错误不能被忽略，如果创建了一个Error但没有检查它，析构时会触发运行时错误，这强制程序员正确处理错误。检查方式包括成功状态，转换为bool即可标记为已检查，失败状态则必须使用handleErrors或handleAllErrors处理。

这种设计也体现了一些权衡。为什么用运行时检查而不是编译期检查？因为编译期检查会使API更复杂，运行时检查在实践中足够有效，这种设计平衡了易用性和安全性。

### Expected的设计：类型安全的错误处理

`Expected<T>`包装可能失败的操作，这种设计有几个重要动机。首先是返回值与错误统一，单一返回类型，同时包含成功值或错误，避免输出参数的丑陋，更接近函数式编程的Either类型。其次是类型安全，成功值的类型在编译期确定，不能错误地解释错误为成功值，通过类型系统保证安全。

**使用模式：**

```cpp
llvm::Expected<int> parseInt(StringRef Str) {
  int Result;
  if (Str.getAsInteger(10, Result))
    return llvm::make_error<llvm::StringError>("Invalid integer",
                                              llvm::inconvertibleErrorCode());
  return Result;
}

// 使用
auto MaybeInt = parseInt("42");
if (!MaybeInt) {
  // 处理错误路径
  return MaybeInt.takeError();
}
int Value = *MaybeInt;  // 安全，已经检查过
```

**与其他方案的比较：**

| 方案 | 优点 | 缺点 |
|-----|------|------|
| 异常 | 简洁的控制流 | 性能开销、异常安全问题 |
| 错误码+输出参数 | 性能好 | 丑陋、容易忘记检查 |
| Expected<T> | 类型安全、单一返回 | 需要一些样板代码 |

LLVM选择Expected<T>是一个平衡的选择，它避免了异常的性能和安全问题，比错误码+输出参数更优雅，同时通过Error的Checked标志强制错误检查。

### ErrorInfo的设计：可扩展的错误类型

LLVM允许用户定义自定义错误类型，通过继承`ErrorInfo`：

```cpp
template<typename ThisT>
class ErrorInfo : public ErrorInfoBase {
  // ...
};
```

这种设计有几个要点。首先是CRTP模式，静态多态，没有虚函数表开销，类型ID通过静态成员实现，实现高效的类型检查。其次是错误类型ID，每个错误类型有唯一的静态ID，支持isa/dyn_cast风格的类型检查，类似于LLVM的其他RTTI系统。最后是错误信息日志，每个错误类型必须实现log()方法，可以自定义错误格式，支持rich error信息。

**自定义错误示例分析：**

```cpp
class MyError : public llvm::ErrorInfo<MyError> {
  std::string Msg;
  int ErrorCode;

public:
  static char ID;  // 唯一ID

  MyError(StringRef Msg, int Code) 
    : Msg(Msg.str()), ErrorCode(Code) {}

  void log(llvm::raw_ostream &OS) const override {
    OS << "MyError: " << Msg << " (code " << ErrorCode << ")";
  }

  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }
};

char MyError::ID = 0;  // 定义静态成员
```

这种设计有几个原因。ID用于类型鉴别，不需要完整的虚函数表，只需要一个指针。log()用于显示，提供统一的错误显示接口。convertToErrorCode()用于兼容，与传统std::error_code互操作。

### 错误处理函数：模式匹配风格

LLVM提供了`handleErrors`和`handleAllErrors`，实现了类似模式匹配的错误处理。这种设计理念包括类型安全的模式匹配，按错误类型分派处理函数，编译期类型检查，类似于函数式语言的模式匹配。未处理错误的处理，handleErrors返回未处理的错误，handleAllErrors要求所有错误都被处理，实现灵活性与安全性的平衡。

**使用模式分析：**

```cpp
auto E = doSomething();
auto Remaining = llvm::handleErrors(std::move(E),
  [](const MyError &E) {
    // 处理MyError类型
    llvm::errs() << E.message() << "\n";
  },
  [](const OtherError &E) -> llvm::Error {
    // 处理并可能返回新错误
    if (canRecover(E))
      return llvm::Error::success();
    return llvm::make_error<OtherError>(E);
  });

if (Remaining) {
  // 向上传递未处理的错误
  return Remaining;
}
```

为什么不直接用switch-case？因为switch-case无法处理类型分派，添加新错误类型不需要修改现有代码，而且这种设计支持表达式风格，可以作为表达式使用，更灵活。

## 调试输出：可观测性设计

LLVM的调试输出机制体现了其对可观测性的重视。

### dbgs()：为什么需要这个？

[dbgs()](/llvm/include/llvm/Support/Debug.h) 是LLVM的调试输出流，这种设计有几个动机。首先是与outs()分离，dbgs()用于调试输出，outs()用于正常输出，可以独立重定向。其次是统一接口，所有调试输出使用同一个流，方便重定向到文件或其他位置，避免了使用std::cerr的混乱。最后是性能考虑，raw_ostream比std::ostream更高效，更少的虚函数调用，更好的缓冲策略。

### LLVM_DEBUG：条件编译的艺术

[LLVM_DEBUG](/llvm/include/llvm/Support/Debug.h) 宏是条件调试输出的经典设计，这种设计有几个要点。首先是编译期条件，调试代码在Release模式下完全不编译，零运行时开销，不会影响发布版本性能。其次是按类型启用，DEBUG_TYPE宏定义当前模块的调试类型，运行时可以选择性启用某些类型的调试输出，灵活控制调试信息量。

**使用模式：**

```cpp
#define DEBUG_TYPE "my-pass"

void runMyPass() {
  LLVM_DEBUG(llvm::dbgs() << "Running my pass\n";);

  for (auto &BB : F) {
    LLVM_DEBUG({
      // 可以是任意代码块
      llvm::dbgs() << "Processing basic block: ";
      BB.printName(llvm::dbgs());
      llvm::dbgs() << "\n";
    });
  }
}
```

为什么这样设计？因为遵循零开销原则，不使用就不付费，灵活控制，可以精细控制哪些调试输出可见，同时易于使用，简洁的语法，不会使代码混乱。

与NDEBUG的关系是，NDEBUG控制assert和LLVM_DEBUG，Debug模式下NDEBUG未定义，启用所有检查和调试，Release模式下NDEBUG定义，禁用检查和调试。

## 断言：防御性编程的实践

LLVM使用标准C断言，但有自己的使用模式和哲学。

### 断言的使用哲学

LLVM的断言使用体现了几个原则。首先是检查不变量，验证假设总是成立，捕获逻辑错误，文档化代码假设。其次是Debug模式仅启用，Release模式不影响性能，快速反馈开发阶段的问题，不增加发布版本负担。最后是与错误处理的区别，断言检查不应该发生的内部错误，错误处理处理可能发生的外部错误，两者有明确的界限。

**常见断言模式分析：**

```cpp
// 非空检查
assert(V && "Value must not be null");

// 类型检查
assert(isa<Instruction>(V) && "Expected instruction");

// 范围检查
assert(Idx < Vec.size() && "Index out of bounds");

// 状态检查
assert(isValidState() && "Invalid state");

// 算法检查
assert(isSorted(Vec) && "Vector not sorted");
```

### LLVM_ENABLE_ASSERTIONS：构建时控制

从 [llvm/CMakeLists.txt](/llvm/CMakeLists.txt) 可以看到，断言受CMake选项控制。这种设计有几个权衡。首先是默认策略，Debug模式默认启用断言，Release模式默认禁用断言，平衡性能和安全性。其次是可覆盖性，可以显式启用或禁用，适应不同需求，不强制单一策略。

为什么Release模式默认禁用？因为断言可能有开销，有些断言检查在Release模式下可能过度严格，这也是历史原因，是传统做法。

## 堆栈跟踪：诊断的艺术

LLVM的PrettyStackTrace机制体现了其对诊断能力的重视。

### 为什么需要PrettyStackTrace？

堆栈跟踪面临一些挑战，原始堆栈跟踪只有地址，缺少上下文，不知道每个函数在做什么，在Release模式下没有调试符号时更糟。

PrettyStackTrace提供了解决方案，它人工维护的堆栈，显式跟踪当前操作，提供有意义的描述，描述正在做什么，而不是在哪里，并且独立于调试符号，即使在Release模式也有用。

**使用模式分析：**

```cpp
void compileModule(Module &M) {
  llvm::PrettyStackTraceString Trace("Compiling module " + M.getName());

  for (auto &F : M) {
    llvm::PrettyStackTraceEntry FuncTrace("Compiling function " + F.getName());
    // 编译函数
  }
}
```

这种设计有几个特点。首先是RAII风格，构造时添加，析构时移除。其次是嵌套支持，可以形成堆栈结构。最后是轻量级，通常只是一个字符串和指针，零开销，如果不崩溃就没有成本。

## 职责边界与最佳实践

理解LLVM错误处理机制，最重要的是知道什么时候用什么。

### 场景选择指南

| 场景 | 推荐机制 | 理由 |
|-----|---------|------|
| 内部逻辑错误 | llvm_unreachable | 标记不可达代码 |
| 内部bug | reportFatalInternalError | 生成崩溃报告 |
| 用户错误 | reportFatalUsageError | 优雅退出 |
| 可恢复错误 | Error/Expected | 返回错误信息 |
| 不变量检查 | assert | Debug模式检查 |
| 调试输出 | LLVM_DEBUG | 条件编译输出 |
| 崩溃诊断 | PrettyStackTrace | 有意义的上下文 |

### 常见陷阱与避坑指南

**陷阱1：忽略Error返回值**

```cpp
// 危险！
parseFile("file.txt");  // 返回Error但忽略

// 正确
if (auto E = parseFile("file.txt"))
  return E;  // 或其他处理
```

**陷阱2：过度使用report_fatal_error**

```cpp
// 不推荐：用户错误不应该崩溃
if (!fileExists(filename))
  llvm::report_fatal_error("File not found");

// 推荐：使用Error/Expected
if (!fileExists(filename))
  return llvm::make_error<FileNotFoundError>(filename);
```

**陷阱3：在自定义错误处理中盲目抛出异常**

```cpp
// 危险！
void badHandler(void *, const char *, bool) {
  throw std::runtime_error("error");  // 可能穿越非异常安全代码
}

// 推荐：只在你完全理解后果时才这样做
```

**陷阱4：滥用assert进行用户输入验证**

```cpp
// 错误：assert在Release模式被禁用
void processInput(int x) {
  assert(x > 0 && "x must be positive");  // 不能依赖这个！
  // ...
}

// 正确：使用真正的错误检查
llvm::Error processInput(int x) {
  if (x <= 0)
    return llvm::make_error<InvalidInputError>(x);
  // ...
}
```

## 总结

LLVM的错误处理和调试机制体现了几个核心设计思想。首先是多层防御，从assert到Error，多层次的错误处理。其次是零开销原则，Release模式不支付未使用功能的成本。第三是安全与性能平衡，Debug模式安全，Release模式快速。第四是诊断友好，PrettyStackTrace等机制辅助调试。最后是强制错误检查，Error的Checked标志防止忽略错误。理解这些设计理念，不仅能帮助我们正确使用LLVM，更能学习到工业级C++项目的错误处理最佳实践。
