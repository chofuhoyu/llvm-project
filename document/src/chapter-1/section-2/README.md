# 1.2 C++源代码惯用法分析

本节深入分析LLVM源代码中使用的C++惯用法、设计模式和编码风格。理解这些惯用法是区分LLVM本体代码与用户代码职责边界的关键，也是有效修改和扩展LLVM的基础。

## 本节内容

- [内存管理与所有权模型](memory-management.md) - 分析LLVM的智能指针、内存管理策略和所有权语义
- [ADT库详解](adt-library.md) - 详细介绍LLVM自定义的容器和数据结构
- [模板元编程技术](template-metaprogramming.md) - 探讨LLVM中的模板元编程和类型特征
- [错误处理与调试机制](error-handling.md) - 了解LLVM的错误处理和调试工具
- [LLVM类型系统](type-system.md) - 分析LLVM的类型系统和类型唯一化机制
- [IR核心类](ir-core.md) - 理解LLVM IR的核心类和SSA表示
- [Pass管理器](pass-manager.md) - 了解LLVM的Pass管理器和Pass框架
