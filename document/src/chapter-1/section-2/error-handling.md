# 1.2.4 错误处理与调试机制

LLVM提供了一套完整的错误处理和调试机制，用于处理可恢复错误、致命错误以及辅助调试。本节分析这些机制的设计和使用方式。

## 致命错误处理

### report_fatal_error

[report_fatal_error](/llvm/include/llvm/Support/ErrorHandling.h#L63) 用于报告不可恢复的致命错误：

```cpp
[[noreturn]] void report_fatal_error(const char *reason,
                                      bool gen_crash_diag = true);
[[noreturn]] void report_fatal_error(StringRef reason,
                                      bool gen_crash_diag = true);
[[noreturn]] void report_fatal_error(const Twine &reason,
                                      bool gen_crash_diag = true);
```

使用示例：

```cpp
#include "llvm/Support/ErrorHandling.h"

void processFile(StringRef Filename) {
  if (!fileExists(Filename)) {
    llvm::report_fatal_error("File not found: " + Filename);
  }
}
```

### reportFatalInternalError vs reportFatalUsageError

LLVM 15+将致命错误分为两类：

| 函数 | 用途 | 行为 |
|-----|------|------|
| [reportFatalInternalError](/llvm/include/llvm/Support/ErrorHandling.h#L77) | LLVM内部bug | 生成崩溃跟踪，要求用户报告bug |
| [reportFatalUsageError](/llvm/include/llvm/Support/ErrorHandling.h#L95) | 用户使用错误 | 优雅退出，不生成崩溃跟踪 |

```cpp
// 内部错误：LLVM的bug
if (!isValidState(State)) {
  llvm::reportFatalInternalError("Invalid internal state");
}

// 使用错误：用户的问题
if (!isValidOption(Option)) {
  llvm::reportFatalUsageError("Invalid option: " + Option);
}
```

### 自定义错误处理

可以安装自定义致命错误处理函数：

```cpp
#include "llvm/Support/ErrorHandling.h"

void myErrorHandler(void *user_data, const char *reason,
                   bool gen_crash_diag) {
  // 自定义错误处理
  fprintf(stderr, "ERROR: %s\n", reason);
  // ...
}

// 安装处理函数
llvm::install_fatal_error_handler(myErrorHandler, nullptr);

// 作用域版本
{
  llvm::ScopedFatalErrorHandler Handler(myErrorHandler);
  // 在此作用域内使用自定义处理
}
// 离开作用域后恢复默认处理
```

### llvm_unreachable

[llvm_unreachable](/llvm/include/llvm/Support/ErrorHandling.h#L146) 标记不可达的代码位置：

```cpp
#include "llvm/Support/ErrorHandling.h"

enum class Opcode { Add, Sub, Mul };

int compute(Opcode Op, int A, int B) {
  switch (Op) {
    case Opcode::Add: return A + B;
    case Opcode::Sub: return A - B;
    case Opcode::Mul: return A * B;
  }
  llvm_unreachable("Invalid opcode");
}
```

在Debug模式下，会打印消息和位置；在Release模式下，优化器利用这个信息进行优化。

## 可恢复错误处理

### Error

[Error](/llvm/include/llvm/Support/Error.h#L89) 是LLVM的轻量级错误类，用于可恢复错误：

```cpp
class Error {
  ErrorInfoBase *Ptr;
  bool Checked;
  
public:
  // 构造成功的Error
  Error() : Ptr(nullptr), Checked(false) {}
  
  // 从ErrorInfo构造
  Error(std::unique_ptr<ErrorInfoBase> E);
  
  // 移动构造和赋值
  Error(Error &&Other);
  Error &operator=(Error &&Other);
  
  // 析构前必须检查
  ~Error();
  
  // 布尔转换：true表示有错误
  explicit operator bool() const;
  
  // 检查是否成功
  static Error success();
};
```

### Expected

[Expected](/llvm/include/llvm/Support/ErrorOr.h)（或新的 `Expected<T>`）包装可能失败的操作结果：

```cpp
#include "llvm/Support/Error.h"

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
  // 处理错误
  llvm::handleAllErrors(MaybeInt.takeError(),
    [](const llvm::StringError &E) {
      llvm::errs() << "Error: " << E.getMessage() << "\n";
    });
} else {
  int Value = *MaybeInt;
  // 使用Value
}
```

### ErrorInfo

自定义错误类型需要继承 [ErrorInfo](/llvm/include/llvm/Support/Error.h#L44)：

```cpp
#include "llvm/Support/Error.h"

class MyError : public llvm::ErrorInfo<MyError> {
  std::string Msg;
  
public:
  static char ID;  // 必须有
  
  MyError(StringRef Msg) : Msg(Msg.str()) {}
  
  void log(llvm::raw_ostream &OS) const override {
    OS << "MyError: " << Msg;
  }
  
  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }
};

char MyError::ID = 0;

// 使用
llvm::Error doSomething() {
  if (somethingFailed)
    return llvm::make_error<MyError>("Something went wrong");
  return llvm::Error::success();
}
```

### 错误处理函数

| 函数 | 用途 |
|-----|------|
| `handleErrors` | 处理特定类型的错误，未处理的错误返回 |
| `handleAllErrors` | 处理所有错误 |
| `cantFail` | 断言错误不会发生，否则终止 |
| `consumeError` | 消耗错误，不处理 |

```cpp
// handleErrors示例
auto E = doSomething();
auto Remaining = llvm::handleErrors(std::move(E),
  [](const MyError &E) {
    // 处理MyError
    llvm::errs() << E.message() << "\n";
  },
  [](const OtherError &E) -> llvm::Error {
    // 处理并可能返回新错误
    if (canRecover(E))
      return llvm::Error::success();
    return llvm::make_error<OtherError>(E);
  });

if (Remaining) {
  // 未处理的错误
  return Remaining;
}

// cantFail示例 - 断言成功
int Result = llvm::cantFail(parseInt("42"));
```

## 调试输出

### dbgs()

[dbgs()](/llvm/include/llvm/Support/Debug.h) 是LLVM的调试输出流：

```cpp
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

void debugFunction(Function *F) {
  llvm::dbgs() << "Function: " << F->getName() << "\n";
  F->print(llvm::dbgs());
}
```

### LLVM_DEBUG

[LLVM_DEBUG](/llvm/include/llvm/Support/Debug.h) 宏用于条件调试输出：

```cpp
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "my-pass"

void runMyPass() {
  LLVM_DEBUG(llvm::dbgs() << "Running my pass\n");;
  
  for (auto &BB : F) {
    LLVM_DEBUG({
      llvm::dbgs() << "Processing basic block: ";
      BB.printName(llvm::dbgs());
      llvm::dbgs() << "\n";
    });
  }
}
```

启用调试输出：
```bash
# 编译时启用断言和调试
cmake -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_DEBUG=ON ...

# 运行时启用特定调试类型
opt -debug-only=my-pass ...
```

### DEBUG_WITH_TYPE

使用不同的调试类型：

```cpp
void analyzeCFG(Function &F) {
  DEBUG_WITH_TYPE("cfg",
    llvm::dbgs() << "CFG analysis for " << F.getName() << "\n";
  );
}
```

## 断言

LLVM使用标准C断言，但有自己的扩展：

### 基本断言

```cpp
#include <cassert>

void processValue(Value *V) {
  assert(V && "Value must not be null");
  // 处理V
}
```

### LLVM_ENABLE_ASSERTIONS

断言受 `LLVM_ENABLE_ASSERTIONS` CMake选项控制：

```cmake
# CMakeLists.txt
option(LLVM_ENABLE_ASSERTIONS "Enable assertions" ON)
```

Debug构建默认启用，Release构建默认禁用。

### 自定义断言宏

LLVM代码中常见的断言模式：

```cpp
// 类型检查
assert(isa<Instruction>(V) && "Expected instruction");

// 范围检查
assert(Idx < Vec.size() && "Index out of bounds");

// 状态检查
assert(isValidState() && "Invalid state");
```

## 堆栈跟踪

### PrettyStackTrace

[PrettyStackTrace](/llvm/include/llvm/Support/PrettyStackTrace.h) 用于在崩溃时打印有意义的堆栈信息：

```cpp
#include "llvm/Support/PrettyStackTrace.h"

void compileModule(Module &M) {
  llvm::PrettyStackTraceString Trace("Compiling module " + M.getName());
  
  for (auto &F : M) {
    llvm::PrettyStackTraceEntry FuncTrace("Compiling function " + F.getName());
    // 编译函数
  }
}
```

### 自定义堆栈跟踪条目

```cpp
class PrettyStackTraceFunction : public llvm::PrettyStackTraceEntry {
  Function &F;
  
public:
  PrettyStackTraceFunction(Function &F) : F(F) {}
  
  void print(llvm::raw_ostream &OS) const override {
    OS << "  Function: " << F.getName() << "\n";
  }
};
```

## 职责边界总结

| 机制 | LLVM用途 | 用户代码职责 |
|-----|---------|------------|
| `reportFatalInternalError` | LLVM内部bug报告 | 不应该需要调用 |
| `reportFatalUsageError` | 用户错误/无效输入 | 用于工具开发 |
| `Error`/`Expected` | 可恢复错误 | 在扩展中使用处理错误 |
| `LLVM_DEBUG` | 调试输出 | 在开发中添加调试信息 |
| `assert` | 不变量检查 | 在修改中保持/添加断言 |
| `PrettyStackTrace` | 崩溃诊断 | 在自定义Pass中添加跟踪 |

## 调试技巧

1. **启用断言构建** - Debug构建捕获更多问题
2. **使用LLVM_DEBUG** - 添加有条件的调试输出
3. **使用dbgs()** - 直接输出调试信息
4. **添加PrettyStackTrace** - 帮助诊断崩溃
5. **使用opt的调试选项**：
   ```bash
   opt -debug ...              # 启用所有调试
   opt -debug-only=pass-name ...  # 启用特定Pass的调试
   opt -print-after-all ...    # 打印所有变换后的IR
   opt -print-before-all ...   # 打印所有变换前的IR
   ```
