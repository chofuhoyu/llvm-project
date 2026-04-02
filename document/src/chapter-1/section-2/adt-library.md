# 1.2.2 ADT库详解

ADT是LLVM提供的一套自定义数据结构和容器库。这套库不是对STL的简单替代，而是针对编译器工作负载的特点进行了深度优化的结果。理解这些容器的设计理念和使用场景，能帮助我们写出既高效又符合LLVM风格的代码。

## 设计理念的深层分析

LLVM的ADT库设计遵循的原则不仅仅是简单的列表，每个原则背后都有深刻的工程考量。

### 性能优先：为什么不能直接用STL？

从历史上看，LLVM曾经使用STL，但后来逐步实现了自己的容器。这是因为编译器有特殊的工作负载，需要处理大量小型对象，如指令、操作数、基本块等，这些对象生命周期短，创建销毁频繁，而编译器的访问模式相对固定，可以针对性优化。

STL在这种场景下存在一些性能瓶颈，包括内存分配开销，STL容器每次增长都可能触发内存分配，以及某些STL实现的内存布局对缓存不友好，调试模式下的调试检查在Debug模式下可能非常慢。

LLVM的ADT库提供了特定优化机会，小型容器可以完全在栈上操作，利用指针对齐特性进行空间优化，并且不需要处理复杂的异常安全场景，从而实现了简单内存模型。

### 内存效率：空间优化的艺术

LLVM的ADT在内存效率方面下了很大功夫。SmallVector的内联存储实现了小对象优化，避免了小型列表的堆分配，大幅减少内存分配器压力。PointerIntPair利用指针低位存储整数，PointerUnion利用指针低位存储类型标签，这种优化在64位系统上尤为有效，充分实现了指针空间利用。DenseMap的桶布局更紧凑，减少内存碎片，提高缓存利用率，体现了紧凑的数据结构设计。

### 简单实用：接口设计的哲学

LLVM的ADT接口设计追求简单性，但这不是简陋。ADT库不追求完全兼容STL，只实现常用功能，不追求接口完整，避免过度设计，保持接口简洁，降低误用的可能性。ArrayRef和StringRef明确表示非所有者，减少生命周期错误，使代码意图更清晰，体现了明确的所有权语义。在Debug模式下提供断言检查和边界检查，体现了安全优先的设计理念。

### 堆栈优先：性能与安全的平衡

栈上存储是LLVM ADT的重要设计策略。栈具有分配零开销的优势，栈指针移动即可完成分配，离开作用域自动释放，栈内存通常在缓存中，也不会造成内存碎片。但栈也有其限制，栈空间有限，通常只有几MB，不能超出作用域返回，大型对象在栈上移动成本高。

LLVM的策略是小型数据用栈，通过SmallVector的内联存储实现，大型数据用堆，超过阈值时切换到堆，并且这种切换对用户代码是透明的，用户代码不需要关心。

## 非所有者引用类型：ArrayRef与StringRef

ArrayRef和StringRef是LLVM中最常用的类型之一，它们的设计体现了"引用而非拥有"的理念。

### ArrayRef：为什么需要这样一个类型？

从 [ArrayRef.h](/llvm/include/llvm/ADT/ArrayRef.h) 的设计可以看出，这不是一个简单的"数组引用"。

#### 设计动机分析

在C++中，函数接受数组参数有几种传统方式：

```cpp
// 方式1：接受std::vector
void process(const std::vector<int> &Vec) { /* ... */ }
// 问题：用户必须使用std::vector，不能用C数组或initializer_list

// 方式2：接受指针和大小
void process(const int *Data, size_t Size) { /* ... */ }
// 问题：接口不友好，容易出错，忘记传Size

// 方式3：模板迭代器
template<typename Iter>
void process(Iter Begin, Iter End) { /* ... */ }
// 问题：模板膨胀，实现必须在头文件
```

ArrayRef的设计解决了这些问题，它可以接受任何连续序列，提供统一接口，只有指针和大小，没有额外分配，实现零开销，同时提供编译期类型检查，类型安全，并且一个参数就能清晰表达意图，简洁明了。

#### 隐式转换的设计权衡

ArrayRef的一个重要特性是支持隐式转换：

```cpp
template<typename C>
ArrayRef(const C &V) : Data(V.data()), Length(V.size()) {}
```

这种设计优先考虑易用性，函数参数自动转换，调用方代码更简洁，同时ArrayRef是不可变引用，不会意外修改数据，提供了安全保障，ArrayRef作为参数，生命周期仅限于函数调用，生命周期清晰。

但隐式转换也存在潜在风险：

```cpp
// 危险！返回局部对象的ArrayRef
llvm::ArrayRef<int> bad() {
  std::vector<int> Vec = {1, 2, 3};
  return Vec;  // 编译通过，但Vec被销毁，ArrayRef悬空
}

// 安全：返回vector
std::vector<int> good() {
  std::vector<int> Vec = {1, 2, 3};
  return Vec;  // 正确，转移所有权
}
```

这就是为什么ArrayRef的注释中反复强调生命周期问题。

#### 切片操作的设计

ArrayRef提供了丰富的切片操作：

```cpp
ArrayRef<T> slice(size_t N, size_t M) const;
ArrayRef<T> drop_front(size_t N = 1) const;
ArrayRef<T> drop_back(size_t N = 1) const;
ArrayRef<T> take_front(size_t N = 1) const;
ArrayRef<T> take_back(size_t N = 1) const;
```

这些切片操作零成本，只是调整指针和大小，不复制数据，支持链式调用，如`arr.drop_front(1).take_back(2)`，切片返回新的ArrayRef，原对象不变，保证了不可变性，在Debug模式下通常还有断言检查，确保边界安全。

### StringRef：不仅仅是ArrayRef&lt;char&gt;

StringRef是ArrayRef&lt;char&gt;的特化，但它不是简单的别名，而是增加了字符串特有的操作。

#### 设计决策分析

StringRef在C++17之前就存在了，这是历史原因。StringRef的API更符合LLVM的习惯，LLVM可以完全控制StringRef的演进，保证API稳定性。StringRef提供了字符串特有的操作，如startswith、endswith、trim、find等，这些操作对于ArrayRef&lt;char&gt;没有意义，但对StringRef很有用。StringRef支持隐式转换自std::string，也提供显式转换到std::string的str()方法，这种设计平衡了易用性和清晰性。

## 小型容器：SmallVector的设计深度分析

SmallVector是LLVM中最具代表性的容器之一，其设计体现了对性能的极致追求。

### 为什么SmallVector如此重要？

让我们分析一个常见场景：

```cpp
// 场景：收集一条指令的操作数
// 大多数指令有1-3个操作数，少数有更多

// 使用std::vector
std::vector<Value*> Operands;
Operands.push_back(Op1);  // 可能触发分配！
Operands.push_back(Op2);
Operands.push_back(Op3);

// 使用SmallVector
llvm::SmallVector<Value*, 4> Operands;  // 4个内联槽
Operands.push_back(Op1);  // 纯栈操作！
Operands.push_back(Op2);
Operands.push_back(Op3);
```

**性能差异分析：**

| 操作 | std::vector | SmallVector<4> |
|-----|-------------|---------------|
| 创建空容器 | 可能分配（取决于实现） | 无分配 |
| 插入3个元素 | 1-2次分配 | 无分配 |
| 销毁 | 释放内存 | 无释放 |
| 总开销 | 多次系统调用 | 纯用户态操作 |

对于编译器这样创建百万个小型容器的场景，这种差异会累积成显著的性能提升。

### 内联存储的实现细节

从 [SmallVector.h](/llvm/include/llvm/ADT/SmallVector.h) 可以看到其精妙的实现：

```cpp
template<typename T, unsigned N = 4>
class SmallVector : public SmallVectorImpl<T> {
  char InlineElts[N * sizeof(T)];  // 内联存储
  // ...
};
```

这种设计有几个重要考量。首先是为什么用char数组而不是T数组，这是为了避免默认构造，char数组不需要构造T对象，SmallVectorImpl精确控制对象生命周期，通过alignment属性确保正确对齐。其次是为什么N=4是默认值，这是一个经验值，大多数小型列表不超过4个元素，4个元素在栈上占用空间适中，同时用户可以根据场景调整。最后是何时切换到堆，当元素数量超过N时，切换是透明的，用户代码不需要改变，已存在的元素会被移动到堆上。

### 内联大小N的选择策略

选择合适的N是使用SmallVector的艺术。过小的N如N=1，栈占用小，但频繁切换到堆会失去优势，适用于几乎总是0或1个元素的场景。适中的N如N=4或N=8，能够平衡栈空间和堆分配率，虽然栈占用稍大，但适用于大多数情况。过大的N如N=64，几乎不用堆，但栈空间占用大，有栈溢出风险，仅适用于确实需要这么多元素且有把握栈空间足够的场景。

**推荐实践：**

```cpp
// 指令操作数：通常1-4个
llvm::SmallVector<Value*, 4> Operands;

// 基本块前驱：可能多个，但通常不多
llvm::SmallVector<BasicBlock*, 8> Preds;

// 临时缓冲区：可能需要更大
llvm::SmallVector<char, 128> TempBuffer;
```

## 哈希表容器：DenseMap的优化策略

DenseMap是LLVM对哈希表的重新设计，相比std::unordered_map有显著优势。

### std::unordered_map的问题

标准库的无序映射通常采用节点基设计，每个元素是独立分配的节点，导致指针chasing，缓存不友好，内存碎片严重。空间效率也较低，每个节点需要额外的元数据，如next指针、哈希值等，Load factor通常不高，约0.75左右，空间利用率可能只有50%。此外，插入可能导致重新哈希，所有迭代器失效，实现细节导致使用复杂。

### DenseMap的设计改进

DenseMap采用开放寻址策略，所有桶在一个连续数组中，更好的缓存局部性，更少的内存碎片。DenseMap使用空键标记空桶，墓碑键标记已删除的桶，从 [DenseMapInfo](/llvm/include/llvm/ADT/DenseMap.h#L442) 可以看到这种设计。DenseMap没有next指针，不存储哈希值，而是重新计算，空间利用率更高。

### DenseMapInfo的设计哲学

DenseMapInfo是DenseMap的关键设计，值得深入分析：

```cpp
template<typename T>
struct DenseMapInfo<T*> {
  static inline T* getEmptyKey() {
    return (T*)(uintptr_t)-1;  // 全1的指针
  }
  static inline T* getTombstoneKey() {
    return (T*)(uintptr_t)-2;  // 全1减1
  }
  // ...
};
```

选择这些特殊值有几个原因。首先是为什么是-1和-2，这些是无效的指针值，存在对齐问题，不太可能是真实的键值，易于识别。其次是用户键冲突怎么办，用户需要确保键不会是这两个特殊值，对于指针，这通常不是问题，因为-1和-2不会是有效地址，但对于自定义类型，需要小心处理。最后是为什么需要墓碑键，删除元素时不能直接清空桶，需要保留探测链的完整性，墓碑键标记已删除但仍影响探测的位置。

## 其他容器的设计亮点

### SmallPtrSet：指针的专用优化

SmallPtrSet专门针对指针进行优化，对于小型指针集，使用线性搜索而非哈希，对于少量指针如8个以下，线性搜索更快，避免哈希计算和探测开销。SmallPtrSet实现了渐进式切换，小型时用简单数组加线性搜索，超过阈值时切换到哈希表，这种切换对用户是透明的，用户无需关心。

### StringMap：字符串键的优化

StringMap针对字符串键进行优化，键和值在同一个内存块中，减少内存分配次数，更好的缓存局部性。字符串哈希一次计算后存储在节点中，避免重复哈希，同时提供字符串比较优化。

### MapVector与SetVector：有序性的需求

有时我们需要哈希表的快速查找，但也需要保持插入顺序。MapVector结合DenseMap和std::vector，DenseMap提供快速查找，std::vector维护插入顺序，双重存储，内存换效率。这种设计适用于需要按插入顺序迭代、需要稳定的迭代顺序、可以接受双倍内存开销的场景。

## 职责边界与使用指南

理解LLVM ADT容器，最重要的是知道什么时候用什么。

### 场景分析与容器选择

| 场景 | 推荐容器 | 理由 |
|-----|---------|------|
| 函数参数，只读数组 | ArrayRef<T> | 非所有者，零开销，接受任何输入 |
| 函数参数，只读字符串 | StringRef | 同上，字符串专用操作 |
| 小型临时列表 | SmallVector<T, N> | 栈上优先，减少分配 |
| 指针集合 | SmallPtrSet<T*, N> | 指针专用优化 |
| 指针到值的映射 | DenseMap<K*, V> | 高效哈希表，缓存友好 |
| 字符串到值的映射 | StringMap<V> | 字符串键专用优化 |
| 需要顺序的映射 | MapVector<K, V> | 保持插入顺序 |
| 需要顺序的集合 | SetVector<T> | 保持插入顺序 |

### 常见错误与避坑指南

**错误1：返回ArrayRef/StringRef指向局部对象**

```cpp
// 错误
llvm::ArrayRef<int> getData() {
  std::vector<int> Vec = {1, 2, 3};
  return Vec;  // Vec被销毁，ArrayRef悬空
}

// 正确
std::vector<int> getData() {
  std::vector<int> Vec = {1, 2, 3};
  return Vec;  // 返回vector，转移所有权
}
```

**错误2：SmallVector的N选择不当**

```cpp
// 不推荐：栈占用太大
llvm::SmallVector<int, 1024> BigStackVec;  // 4KB栈空间

// 推荐：适度的内联大小
llvm::SmallVector<int, 32> SmallerVec;
```

**错误3：DenseMap使用自定义类型但未特化DenseMapInfo**

```cpp
// 如果自定义类型可能等于emptyKey或tombstoneKey，需要特化
struct MyKey { /* ... */ };

// 可能需要
namespace llvm {
template<> struct DenseMapInfo<MyKey> {
  // 自定义实现
};
}
```

**错误4：迭代DenseMap时删除元素**

```cpp
// 危险：可能导致迭代器失效或遗漏元素
for (auto It = Map.begin(); It != Map.end(); ++It) {
  if (shouldDelete(It))
    Map.erase(It);  // 问题！
}

// 正确：使用后递增
for (auto It = Map.begin(); It != Map.end(); ) {
  auto Curr = It++;
  if (shouldDelete(Curr))
    Map.erase(Curr);
}
```

## 总结

LLVM的ADT库体现了几个核心设计思想。首先是针对场景优化，不追求通用，而是针对编译器场景深度优化。其次是性能优先，减少分配、提高缓存利用率、降低开销。第三是简单实用，接口简洁，易于正确使用。第四是安全与效率平衡，Debug模式检查，Release模式性能。最后是明确的所有权语义，ArrayRef等类型清晰表达所有权。理解这些设计理念，能帮助我们不仅正确使用LLVM的ADT，更能学习到工业级库的设计经验。
