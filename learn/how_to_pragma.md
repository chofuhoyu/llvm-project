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
- [高级实现：使用 FixItHint 自动插入代码](#高级实现：使用-fixit-hint-自动插入代码)

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
    // Parse optional identifier argument for print directive
    std::string PrintIdentifier;
    if (Tok.is(tok::identifier)) {
      PrintIdentifier = Tok.getIdentifierInfo()->getName().str();
      PP.Lex(Tok); // Consume the identifier
    }
    
    // Find the end of the pragma line to insert after it
    SourceLocation InsertLoc = PragmaLoc;
    while (true) {
      bool Invalid = false;
      const char *Ptr = SM.getCharacterData(InsertLoc, &Invalid);
      if (Invalid || !*Ptr) break;
      
      if (*Ptr == '\n' || *Ptr == '\r') {
        break;
      }
      InsertLoc = InsertLoc.getLocWithOffset(1);
    }

    // Move to beginning of next line
    InsertLoc = InsertLoc.getLocWithOffset(1);
    
    // Build the printf statement
    std::string PrintfStmt;
    if (PrintIdentifier.empty()) {
      PrintfStmt = "    printf(\"#pragma learn print directive received\\n\");\n";
    } else {
      PrintfStmt = "    printf(\"#pragma learn print directive received identifier " + PrintIdentifier + "\\n\");\n";
    }

    // Create diagnostic and add FixItHint to insert printf
    DiagnosticBuilder DiagBuilder = PP.Diag(PragmaLoc, diag::remark_pragma_learn_print);
    DiagBuilder << FixItHint::CreateInsertion(InsertLoc, PrintfStmt);
    PP.Diag(PragmaLoc, diag::note_pragma_learn_insert_printf);
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

## 高级实现：使用 FixItHint 自动插入代码

### 概述

在基础版本中，`#pragma learn print` 只是发出一个备注信息。在高级版本中，我们将使用 Clang 的 `FixItHint` 机制来：
1. 自动插入 `printf` 语句
2. 支持可选的标识符参数
3. 自动检测并添加缺失的 `stdio.h` 或 `cstdio` 头文件

### 新增的诊断消息

首先，在 `DiagnosticParseKinds.td` 中添加新的诊断消息：

```td
def note_pragma_learn_insert_printf : Note&lt;
  "inserting printf statement here"&gt;;

def note_pragma_learn_add_include : Note&lt;
  "adding #include &lt;%0&gt; for printf"&gt;;
```

### 更新 HandlePragma 方法

在 `ParsePragma.cpp` 中更新 `PragmaLearnHandler::HandlePragma()` 方法：

#### 1. 解析可选参数

```cpp
else if (Directive-&gt;isStr("print")) {
  // Parse optional identifier argument for print directive
  std::string PrintIdentifier;
  if (Tok.is(tok::identifier)) {
    PrintIdentifier = Tok.getIdentifierInfo()-&gt;getName().str();
    PP.Lex(Tok); // Consume the identifier
  }
```

#### 2. 找到插入位置

```cpp
// Get the source manager to find insertion points
SourceManager &amp;SM = PP.getSourceManager();

// Find the end of the pragma line to insert after it
SourceLocation InsertLoc = PragmaLoc;
while (true) {
  bool Invalid = false;
  const char *Ptr = SM.getCharacterData(InsertLoc, &amp;Invalid);
  if (Invalid || !*Ptr) break;
  
  if (*Ptr == '\n' || *Ptr == '\r') {
    break;
  }
  InsertLoc = InsertLoc.getLocWithOffset(1);
}

// Move to beginning of next line
InsertLoc = InsertLoc.getLocWithOffset(1);
```

#### 3. 构建 printf 语句并使用 FixItHint 插入

```cpp
// Build the printf statement
std::string PrintfStmt;
if (PrintIdentifier.empty()) {
  PrintfStmt = "    printf(\"#pragma learn print directive received\\n\");\n";
} else {
  PrintfStmt = "    printf(\"#pragma learn print directive received identifier " + PrintIdentifier + "\\n\");\n";
}

// Create diagnostic and add FixItHint to insert printf
DiagnosticBuilder DiagBuilder = PP.Diag(PragmaLoc, diag::remark_pragma_learn_print);
DiagBuilder << FixItHint::CreateInsertion(InsertLoc, PrintfStmt);
PP.Diag(PragmaLoc, diag::note_pragma_learn_insert_printf);
```

#### 4. 检测并自动添加头文件

```cpp
// Check if we need to add stdio.h include
const LangOptions &amp;LangOpts = PP.getLangOpts();
std::string StdioHeader = LangOpts.CPlusPlus ? "cstdio" : "stdio.h";

// Check if the file already includes stdio.h or cstdio
bool HasStdioInclude = false;
FileID MainFileID = SM.getMainFileID();
SourceLocation FileStart = SM.getLocForStartOfFile(MainFileID);
SourceLocation CurrentPos = FileStart;

// Scan the beginning of the file for includes
const unsigned ScanLimit = 10000;
unsigned Scanned = 0;
while (Scanned &lt; ScanLimit) {
  bool Invalid = false;
  const char *Ptr = SM.getCharacterData(CurrentPos, &amp;Invalid);
  if (Invalid || !*Ptr) break;
  
  if (*Ptr == '#') {
    // ... check if it's an include for stdio.h or cstdio ...
    // (详见完整实现代码)
  }
  
  CurrentPos = CurrentPos.getLocWithOffset(1);
  Scanned++;
}

// If no stdio include found, add one
if (!HasStdioInclude) {
  std::string IncludeStmt = "#include &lt;" + StdioHeader + "&gt;\n";
  PP.Diag(FileStart, diag::note_pragma_learn_add_include) &lt;&lt; StdioHeader;
  DiagnosticBuilder IncludeDiag = PP.Diag(FileStart, diag::remark_pragma_learn_print);
  IncludeDiag &lt;&lt; FixItHint::CreateInsertion(FileStart, IncludeStmt);
}
```

### FixItHint 核心接口

| 方法 | 说明 |
|------|------|
| `FixItHint::CreateInsertion(Loc, Text)` | 在指定位置插入文本 |
| `FixItHint::CreateReplacement(Range, Text)` | 替换指定范围的文本 |
| `FixItHint::CreateRemoval(Range)` | 删除指定范围的文本 |

使用方式：
```cpp
DiagnosticBuilder Diag = PP.Diag(Loc, diag::some_diagnostic);
Diag &lt;&lt; FixItHint::CreateInsertion(InsertLoc, "code to insert");
```

### 编译 Clang

```bash
cd /home/gorun/code/llvm-project/build
cmake --build . -j$(nproc)
```

### 使用 -fixit 选项自动应用修改

**重要说明**：使用 `-fixit` 需要分两步进行：

1. **第一步**：使用 `-fsyntax-only -Xclang -fixit` 修改源文件（只进行语法检查和修改，不编译）
2. **第二步**：正常编译修改后的文件

```bash
# C 语言测试 - 步骤1: 修改文件
./bin/clang -fsyntax-only -Xclang -fixit test_pragma_print.c

# C 语言测试 - 步骤2: 编译修改后的文件
./bin/clang test_pragma_print.c -o test_pragma_print

# C++ 语言测试 - 步骤1: 修改文件
./bin/clang++ -fsyntax-only -Xclang -fixit test_pragma_print.cpp

# C++ 语言测试 - 步骤2: 编译修改后的文件
./bin/clang++ test_pragma_print.cpp -o test_pragma_print_cpp
```

**注意**：请确保测试文件中已经包含了必要的头文件（`#include &lt;stdio.h&gt;` 或 `#include &lt;cstdio&gt;`）和 `main()` 函数。

### 测试文件示例

创建 `test_pragma_print.c`（确保包含 stdio.h 和 main 函数）：

```c
#include &lt;stdio.h&gt;

void test_without_identifier() {
    #pragma learn print
}

void test_with_identifier() {
    #pragma learn print abc
}

int main() {
    test_without_identifier();
    test_with_identifier();
    return 0;
}
```

使用 `-fsyntax-only -Xclang -fixit` 修改后，文件会自动修改为：

```c
#include &lt;stdio.h&gt;
void test_without_identifier() {
    #pragma learn print
    printf("#pragma learn print directive received\n");
}

void test_with_identifier() {
    #pragma learn print abc
    printf("#pragma learn print directive received identifier abc\n");
}

int main() {
    test_without_identifier();
    test_with_identifier();
    return 0;
}
```

### 支持的语法

```c
// 基本用法 - 不带参数
#pragma learn print

// 带标识符参数
#pragma learn print variable_name
#pragma learn print abc
#pragma learn print my_test_id
```

### C++ 支持

在 C++ 模式下，会自动检测并添加 `#include &lt;cstdio&gt;` 而不是 `#include &lt;stdio.h&gt;`。

### 关键实现要点

1. **位置计算** - 使用 `SourceManager` 和字符扫描找到正确的插入位置
2. **头文件检测** - 扫描文件开头检查是否已包含必要的头文件
3. **语言感知** - 根据 `LangOptions.CPlusPlus` 选择正确的头文件
4. **FixIt 集成** - 通过 `DiagnosticBuilder` 传递 `FixItHint`

---

## 完整文件清单

### 修改的文件
1. `clang/include/clang/Basic/DiagnosticParseKinds.td` - 添加诊断消息
2. `clang/lib/Parse/ParsePragma.cpp` - 实现 pragma 处理逻辑
3. `clang/include/clang/Parse/Parser.h` - 添加 handler 成员变量

### 新增的测试和文档文件
1. `learn/how_to_pragma.md` - 本文档
2. `learn/LEARN_PRAGMA_IMPLEMENTATION.md` - 实现总结
3. `learn/LEARN_PRAGMA_ADVANCED_EXAMPLE.cpp` - 高级示例代码
4. `learn/test_learn_pragma.c` - 基础测试文件
5. `learn/test_pragma_print.c` - 高级功能 C 测试文件
6. `learn/test_pragma_print.cpp` - 高级功能 C++ 测试文件

---

## 参考资料

- Clang 源代码中的 `FixItHint` 使用例子
- `clang/include/clang/Basic/Diagnostic.h` - 诊断系统
- `clang/include/clang/Lex/Preprocessor.h` - 预处理器接口
- `clang/include/clang/Basic/SourceManager.h` - 源码位置管理
