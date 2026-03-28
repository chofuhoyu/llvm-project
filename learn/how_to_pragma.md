# 如何在 Clang 中添加自定义 Pragma 指令

本文档基于一个具体的实现示例，详细介绍如何在 Clang 中添加一个自定义的 `#pragma` 编译指令。我们将使用 `#pragma learn` 作为示例，展示完整的实现过程。

## 目录

- [概述](#概述)
- [实现步骤总结](#实现步骤总结)
- [详细实现](#详细实现)
  - [步骤 1：定义诊断消息](#步骤-1定义诊断消息)
  - [步骤 2：创建 PragmaHandler 类](#步骤-2创建-pragmahandler-类)
  - [步骤 3：实现 HandlePragma 方法](#步骤-3实现-handlepragma-方法)
  - [步骤 4：在 Parser 中添加成员变量](#步骤-4在-parser-中添加成员变量)
  - [步骤 5：注册和注销 Handler](#步骤-5注册和注销-handler)
- [核心接口和原理](#核心接口和原理)
- [测试验证](#测试验证)

---

## 概述

在 Clang 中添加自定义 pragma 指令需要修改多个文件，主要涉及：
1. **诊断消息定义** - 在 TableGen 文件中定义错误、警告和备注消息
2. **PragmaHandler 实现** - 创建继承自 `PragmaHandler` 的类来处理 pragma
3. **集成到 Parser** - 将 handler 注册到预处理器中

我们的示例实现了一个 `#pragma learn` 指令，支持两个子指令：
- `#pragma learn warn` - 生成自定义警告
- `#pragma learn print` - 标记位置以便后续插入 printf 语句

---

## 实现步骤总结

| 步骤 | 文件 | 操作 |
|------|------|------|
| 1 | `clang/include/clang/Basic/DiagnosticParseKinds.td` | 添加诊断消息定义 |
| 2 | `clang/lib/Parse/ParsePragma.cpp` | 定义 PragmaHandler 结构体 |
| 3 | `clang/lib/Parse/ParsePragma.cpp` | 实现 HandlePragma 方法 |
| 4 | `clang/include/clang/Parse/Parser.h` | 添加成员变量 |
| 5 | `clang/lib/Parse/ParsePragma.cpp` | 在 initializePragmaHandlers() 中注册 |
| 6 | `clang/lib/Parse/ParsePragma.cpp` | 在 resetPragmaHandlers() 中注销 |

---

## 详细实现

### 步骤 1：定义诊断消息

首先，我们需要在 TableGen 文件中定义我们的 pragma 将要使用的诊断消息。

**文件位置**: `clang/include/clang/Basic/DiagnosticParseKinds.td`

```td
// Learn Pragma Diagnostics
def warn_pragma_learn_activated : Warning<
  "#pragma learn activated with directive: %0">,
  InGroup<DiagGroup<"pragma-learn">>;

def err_pragma_learn_unknown_directive : Error<
  "unknown directive for #pragma learn: %0">;

def note_pragma_learn_available_directives : Note<
  "available directives: warn, print">;

def warn_pragma_learn_warn_directive : Warning<
  "#pragma learn warn: This is a custom warning from learn pragma">,
  InGroup<DiagGroup<"pragma-learn">>;

def remark_pragma_learn_print : Remark<
  "#pragma learn print will insert printf statement here">,
  InGroup<DiagGroup<"pragma-learn">>;
```

**TableGen 诊断消息语法说明**:
- `def` - 定义一个新的诊断消息
- `Warning<...>` / `Error<...>` / `Note<...>` / `Remark<...>` - 诊断类型
- `InGroup<DiagGroup<"name">>` - 将诊断归入特定的警告组
- `%0`, `%1` 等 - 参数占位符，用于在运行时插入值

---

### 步骤 2：创建 PragmaHandler 类

接下来，我们创建一个继承自 `PragmaHandler` 的类。

**文件位置**: `clang/lib/Parse/ParsePragma.cpp`

```cpp
/// PragmaLearnHandler - "\#pragma learn ...".
struct PragmaLearnHandler : public PragmaHandler {
  PragmaLearnHandler() : PragmaHandler("learn") {}
  void HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer,
                    Token &FirstToken) override;
};
```

**关键接口**:
- `PragmaHandler("learn")` - 构造函数参数指定我们要处理的 pragma 名称
- `HandlePragma()` - 这是核心方法，当预处理器遇到 `#pragma learn` 时会调用它

---

### 步骤 3：实现 HandlePragma 方法

这是最核心的部分，我们在这里实现 pragma 的实际逻辑。

**文件位置**: `clang/lib/Parse/ParsePragma.cpp`

```cpp
// Handle '#pragma learn <directive>'.
void PragmaLearnHandler::HandlePragma(Preprocessor &PP,
                                        PragmaIntroducer Introducer,
                                        Token &FirstToken) {
  SourceLocation PragmaLoc = FirstToken.getLocation();
  
  // Lex the next token to get the directive
  Token Tok;
  PP.Lex(Tok);
  
  if (Tok.is(tok::eod)) {
    PP.Diag(PragmaLoc, diag::err_pragma_missing_argument)
        << "learn" << /*Expected=*/true << "identifier";
    return;
  }
  
  if (Tok.isNot(tok::identifier)) {
    PP.Diag(Tok.getLocation(), diag::err_pragma_learn_unknown_directive)
        << Tok.getName();
    PP.Diag(PragmaLoc, diag::note_pragma_learn_available_directives);
    return;
  }
  
  IdentifierInfo *Directive = Tok.getIdentifierInfo();
  PP.Lex(Tok); // Consume the directive token
  
  // Check for extra tokens after the directive
  if (Tok.isNot(tok::eod)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_extra_tokens_at_eol)
        << "learn";
    // Continue processing anyway
  }
  
  // Emit a warning that the pragma was activated
  PP.Diag(PragmaLoc, diag::warn_pragma_learn_activated)
      << Directive->getName();
  
  // Handle different directives
  if (Directive->isStr("warn")) {
    // Emit a custom warning
    PP.Diag(PragmaLoc, diag::warn_pragma_learn_warn_directive);
  } else if (Directive->isStr("print")) {
    // For print directive, we just emit a remark for now
    // In a real implementation, we would need to track this
    // and insert printf during code generation
    PP.Diag(PragmaLoc, diag::remark_pragma_learn_print);
  } else {
    // Unknown directive
    PP.Diag(PragmaLoc, diag::err_pragma_learn_unknown_directive)
        << Directive->getName();
    PP.Diag(PragmaLoc, diag::note_pragma_learn_available_directives);
  }
}
```

**核心接口和方法详解**:

| 接口/方法 | 用途 |
|-----------|------|
| `Preprocessor &PP` | 预处理器对象，用于与词法分析器交互 |
| `Token &FirstToken` | pragma 的第一个 token，包含位置信息 |
| `PP.Lex(Tok)` | 从 token 流中读取下一个 token |
| `Tok.is(tok::eod)` | 检查是否到达行尾（end of directive） |
| `Tok.is(tok::identifier)` | 检查 token 是否是标识符 |
| `Tok.getIdentifierInfo()` | 获取标识符信息 |
| `Directive->isStr("name")` | 比较标识符是否匹配特定字符串 |
| `PP.Diag(Loc, diag::xxx)` | 发出诊断消息（错误、警告等） |
| `SourceLocation` | 表示源代码中的位置 |

---

### 步骤 4：在 Parser 中添加成员变量

我们需要在 Parser 类中添加一个成员变量来持有我们的 handler。

**文件位置**: `clang/include/clang/Parse/Parser.h`

```cpp
private:
  // ... 其他 pragma handler ...
  std::unique_ptr<PragmaHandler> RISCVPragmaHandler;
  std::unique_ptr<PragmaHandler> LearnPragmaHandler;  // 新增这一行

  /// Initialize all pragma handlers.
  void initializePragmaHandlers();

  /// Destroy and reset all pragma handlers.
  void resetPragmaHandlers();
```

---

### 步骤 5：注册和注销 Handler

最后，我们需要在 Parser 初始化时注册 handler，在清理时注销它。

**文件位置**: `clang/lib/Parse/ParsePragma.cpp`

在 `initializePragmaHandlers()` 中添加：

```cpp
void Parser::initializePragmaHandlers() {
  // ... 其他 handler 注册 ...

  if (getTargetInfo().getTriple().isRISCV()) {
    RISCVPragmaHandler = std::make_unique<PragmaRISCVHandler>(Actions);
    PP.AddPragmaHandler("clang", RISCVPragmaHandler.get());
  }

  // Register our custom #pragma learn handler
  LearnPragmaHandler = std::make_unique<PragmaLearnHandler>();
  PP.AddPragmaHandler(LearnPragmaHandler.get());
}
```

在 `resetPragmaHandlers()` 中添加：

```cpp
void Parser::resetPragmaHandlers() {
  // ... 其他 handler 注销 ...

  if (getTargetInfo().getTriple().isRISCV()) {
    PP.RemovePragmaHandler("clang", RISCVPragmaHandler.get());
    RISCVPragmaHandler.reset();
  }

  // Unregister our custom #pragma learn handler
  PP.RemovePragmaHandler(LearnPragmaHandler.get());
  LearnPragmaHandler.reset();
}
```

**注册接口说明**:
- `PP.AddPragmaHandler(handler)` - 注册处理 `#pragma name` 的 handler
- `PP.AddPragmaHandler("namespace", handler)` - 注册处理 `#pragma namespace name` 的 handler
- `PP.RemovePragmaHandler(handler)` - 注销 handler

---

## 核心接口和原理

### PragmaHandler 基类

`PragmaHandler` 是所有 pragma 处理器的基类，定义在 `clang/include/clang/Lex/Pragma.h` 中。

```cpp
class PragmaHandler {
public:
  explicit PragmaHandler(StringRef Name);
  virtual ~PragmaHandler();
  
  /// HandlePragma - This method is called when the preprocessor
  /// encounters the pragma that this handler is registered for.
  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer,
                            Token &FirstToken) = 0;
  
  // ... 其他方法 ...
};
```

### 预处理器工作流程

当预处理器遇到 `#pragma` 时：

1. 读取 `#pragma` 后的标识符
2. 查找注册的对应 `PragmaHandler`
3. 调用 `handler->HandlePragma(PP, Introducer, FirstToken)`
4. `HandlePragma` 方法可以：
   - 使用 `PP.Lex()` 继续读取后续 token
   - 使用 `PP.Diag()` 发出诊断消息
   - 修改预处理器状态
   - 注入注解 token 供后续阶段使用

### Token 流操作

关键的 Token 相关接口：

| 方法 | 说明 |
|------|------|
| `PP.Lex(Tok)` | 从输入流读取下一个 token 到 Tok |
| `Tok.getKind()` | 获取 token 类型 |
| `Tok.is(kind)` | 检查 token 类型 |
| `Tok.getLocation()` | 获取 token 在源文件中的位置 |
| `Tok.getIdentifierInfo()` | 如果是标识符，获取 IdentifierInfo |
| `Tok.getName()` | 获取 token 的字符串表示 |

常见的 token 类型：
- `tok::identifier` - 标识符
- `tok::numeric_constant` - 数字常量
- `tok::string_literal` - 字符串字面量
- `tok::eod` - 指令结束（end of directive）
- `tok::eof` - 文件结束

### 诊断系统

使用 `PP.Diag()` 发出诊断消息：

```cpp
// 基本用法
PP.Diag(Location, diag::diagnostic_name);

// 带参数的用法
PP.Diag(Location, diag::diagnostic_name) << "parameter1" << 42;
```

诊断类型：
- `Error` - 错误，会阻止编译成功
- `Warning` - 警告
- `Remark` - 备注信息
- `Note` - 附加说明，通常跟随在错误或警告之后

---

## 测试验证

创建测试文件 `test_learn_pragma.c`：

```c
#include <stdio.h>

void test_warn_directive() {
    printf("Testing #pragma learn warn\n");
    
    // This should generate a warning
    #pragma learn warn
    
    printf("After #pragma learn warn\n");
}

void test_print_directive() {
    printf("Testing #pragma learn print\n");
    
    // This should mark where to insert printf
    #pragma learn print
    
    printf("After #pragma learn print\n");
}

int main() {
    test_warn_directive();
    test_print_directive();
    return 0;
}
```

编译测试：

```bash
cd /home/gorun/code/llvm-project/build
cmake --build . -j$(nproc)  # 编译 Clang
./bin/clang -Wall -Wextra test_learn_pragma.c
```

预期输出应该包含：
- `#pragma learn activated with directive: warn`
- `#pragma learn warn: This is a custom warning from learn pragma`
- `#pragma learn activated with directive: print`
- `#pragma learn print will insert printf statement here`

---

## 扩展方向

这个示例只是基础实现，你可以进一步扩展：

1. **支持参数** - 让指令接受字符串或表达式参数
2. **AST 集成** - 使用 AST Consumer 在语义分析阶段处理 pragma
3. **代码生成** - 在 CodeGen 阶段实际插入代码
4. **作用域管理** - 支持 `push`/`pop` 来管理 pragma 的作用域
5. **注解 Token** - 通过注入注解 token 与 Parser/Sema 通信

---

## 总结

在 Clang 中添加自定义 pragma 的关键步骤：

1. ✅ 在 `.td` 文件中定义诊断消息
2. ✅ 创建继承自 `PragmaHandler` 的类
3. ✅ 实现 `HandlePragma()` 方法处理逻辑
4. ✅ 在 `Parser` 中添加成员变量
5. ✅ 在 `initializePragmaHandlers()` 中注册
6. ✅ 在 `resetPragmaHandlers()` 中注销
7. ✅ 编译并测试

这个流程适用于大多数简单的 pragma 实现。对于更复杂的功能，可能需要与 AST、Sema 或 CodeGen 阶段集成。
