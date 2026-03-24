# #pragma learn 实现总结

## 概述
本文档总结了在clang中实现新的 `#pragma learn` 功能的完整过程。

## 实现的功能

### 1. 支持的指令
- `#pragma learn warn` - 生成自定义警告
- `#pragma learn print` - 标记需要插入printf语句的位置
- 错误处理：未知指令时提供错误提示

### 2. 诊断消息
- `warn_pragma_learn_activated` - pragma激活警告
- `err_pragma_learn_unknown_directive` - 未知指令错误
- `note_pragma_learn_available_directives` - 可用指令提示
- `warn_pragma_learn_warn_directive` - warn指令的自定义警告
- `remark_pragma_learn_print` - print指令的备注

## 修改的文件

### 1. `/home/gorun/code/llvm-project/clang/include/clang/Basic/DiagnosticParseKinds.td`
添加了新的诊断消息定义。

### 2. `/home/gorun/code/llvm-project/clang/lib/Parse/ParsePragma.cpp`
- 添加了 `PragmaLearnHandler` 结构体定义
- 实现了 `HandlePragma` 方法
- 在 `initializePragmaHandlers()` 中注册handler
- 在 `resetPragmaHandlers()` 中注销handler

### 3. `/home/gorun/code/llvm-project/clang/include/clang/Parse/Parser.h`
添加了 `LearnPragmaHandler` 成员变量。

## 实现细节

### PragmaLearnHandler 的工作流程

1. **接收pragma** - 当预处理器遇到 `#pragma learn` 时调用
2. **解析指令** - 读取下一个token作为指令
3. **验证参数** - 检查是否缺少参数或参数类型错误
4. **处理指令** - 根据指令类型执行相应操作
5. **发出诊断** - 生成警告、错误或备注信息

### 代码结构

```cpp
struct PragmaLearnHandler : public PragmaHandler {
  PragmaLearnHandler() : PragmaHandler("learn") {}
  void HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer,
                    Token &FirstToken) override;
};
```

## 测试文件

创建了 `/home/gorun/code/llvm-project/test_learn_pragma.c` 用于测试：

```c
#include <stdio.h>

void test_warn_directive() {
    #pragma learn warn
}

void test_print_directive() {
    #pragma learn print
}

int main() {
    test_warn_directive();
    test_print_directive();
    return 0;
}
```

## 编译和测试

### 编译clang
```bash
cd /home/gorun/code/llvm-project/build
cmake --build . -j$(nproc)
```

### 测试新pragma
```bash
./bin/clang -Wall -Wextra test_learn_pragma.c
```

## 扩展建议

### 1. 真正的printf插入
当前实现只是发出备注信息。要真正插入printf语句，需要：

- 在Sema阶段跟踪#pragma learn print的位置
- 在CodeGen阶段插入printf调用
- 或者使用AST Consumer在AST遍历时插入

### 2. 添加更多指令
可以轻松扩展支持更多指令：

```cpp
else if (Directive->isStr("info")) {
    PP.Diag(PragmaLoc, diag::remark_pragma_learn_info);
} else if (Directive->isStr("debug")) {
    PP.Diag(PragmaLoc, diag::warn_pragma_learn_debug);
}
```

### 3. 支持参数
让指令接受参数：

```cpp
#pragma learn warn "Custom warning message"
#pragma learn print "Variable x = %d", x
```

### 4. 作用域控制
让pragma只在特定作用域内生效：

```cpp
#pragma learn push
#pragma learn warn
// ... code ...
#pragma learn pop
```

## 在clang中添加pragma的通用步骤

1. **定义诊断消息** - 在 `Diagnostic*.td` 文件中
2. **创建PragmaHandler** - 继承 `PragmaHandler` 类
3. **实现HandlePragma** - 处理pragma逻辑
4. **注册Handler** - 在 `initializePragmaHandlers()` 中
5. **注销Handler** - 在 `resetPragmaHandlers()` 中
6. **添加成员变量** - 在 `Parser.h` 中
7. **测试验证** - 创建测试文件

## 参考资料

- clang源代码中的其他pragma实现（如 `PragmaLoopHintHandler`）
- `clang/include/clang/Lex/Pragma.h` - PragmaHandler基类定义
- `clang/lib/Parse/ParsePragma.cpp` - 所有pragma实现

## 总结

这个实现展示了如何在clang中添加一个新的pragma。虽然print指令的完整功能（实际插入printf）需要更复杂的实现（涉及AST遍历和代码生成），但当前的实现已经提供了一个完整的框架，展示了pragma处理的基本流程。
