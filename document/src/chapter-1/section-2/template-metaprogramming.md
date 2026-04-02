# 1.2.3 模板元编程技术

LLVM广泛使用模板元编程（Template Metaprogramming, TMP）来实现编译期计算、类型分派和代码生成。本节分析LLVM中常用的模板元编程技术，以及它们如何定义LLVM与用户代码之间的职责边界。

## 类型特征扩展

LLVM在 [STLExtras.h](/llvm/include/llvm/ADT/STLExtras.h) 中扩展了C++标准库的类型特征，提供了更多编译期类型检查和操作。

### function_traits

[function_traits](/llvm/include/llvm/ADT/STLExtras.h#L67) 用于萃取可调用对象的类型信息：

```cpp
template <typename T, bool isClass = std::is_class<T>::value>
struct function_traits : public function_traits<decltype(&T::operator())> {};

// 成员函数指针的特化
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType (ClassType::*)(Args...) const, false> {
  enum { num_args = sizeof...(Args) };
  using result_t = ReturnType;
  
  template <size_t Index>
  using arg_t = std::tuple_element_t<Index, std::tuple<Args...>>;
};

// 普通函数指针的特化
template <typename ReturnType, typename... Args>
struct function_traits<ReturnType (*)(Args...), false> {
  enum { num_args = sizeof...(Args) };
  using result_t = ReturnType;
  
  template <size_t i>
  using arg_t = std::tuple_element_t<i, std::tuple<Args...>>;
};
```

使用示例：

```cpp
// 普通函数
int foo(double, std::string);

using Traits = function_traits<decltype(foo)>;
static_assert(Traits::num_args == 2, "");
using ResultType = Traits::result_t;  // int
using Arg0Type = Traits::arg_t<0>;     // double
using Arg1Type = Traits::arg_t<1>;     // std::string

// Lambda
auto Lambda = [](int x) -> float { return x * 2.0f; };
using LambdaTraits = function_traits<decltype(Lambda)>;
static_assert(LambdaTraits::num_args == 1, "");
```

### is_one_of

[is_one_of](/llvm/include/llvm/ADT/STLExtras.h#L110) 检查类型是否在给定的类型列表中：

```cpp
template <typename T, typename... Ts>
using is_one_of = std::disjunction<std::is_same<T, Ts>...>;
```

使用示例：

```cpp
// 检查是否是整数类型之一
template <typename T>
void process(T Value) {
  if constexpr (is_one_of<T, int, long, short>::value) {
    // 处理有符号整数
  } else if constexpr (is_one_of<T, unsigned, unsigned long>::value) {
    // 处理无符号整数
  } else {
    // 其他类型
  }
}
```

### are_base_of

[are_base_of](/llvm/include/llvm/ADT/STLExtras.h#L115) 检查一个类型是否是所有其他类型的基类：

```cpp
template <typename T, typename... Ts>
using are_base_of = std::conjunction<std::is_base_of<T, Ts>...>;
```

### TypesAreDistinct

[TypesAreDistinct](/llvm/include/llvm/ADT/STLExtras.h#L131) 检查所有类型是否互不相同：

```cpp
template <typename... Ts> struct TypesAreDistinct;

template <> struct TypesAreDistinct<> : std::true_type {};

template <typename T, typename... Us>
struct TypesAreDistinct<T, Us...>
  : std::conjunction<
      std::negation<is_one_of<T, Us...>>,
      TypesAreDistinct<Us...>> {};
```

使用示例：

```cpp
static_assert(TypesAreDistinct<int, double, float>::value, "");
// static_assert(TypesAreDistinct<int, int, double>::value, "");  // 编译失败
```

### FirstIndexOfType

[FirstIndexOfType](/llvm/include/llvm/ADT/STLExtras.h#L148) 查找类型在类型列表中的第一个索引：

```cpp
template <typename T, typename... Us> struct FirstIndexOfType;

template <typename T, typename U, typename... Us>
struct FirstIndexOfType<T, U, Us...>
  : std::integral_constant<size_t, 1 + FirstIndexOfType<T, Us...>::value> {};

template <typename T, typename... Us>
struct FirstIndexOfType<T, T, Us...>
  : std::integral_constant<size_t, 0> {};
```

使用示例：

```cpp
// 查找类型索引
constexpr size_t Idx = FirstIndexOfType<double, int, double, float>::value;
static_assert(Idx == 1, "");
```

### TypeAtIndex

[TypeAtIndex](/llvm/include/llvm/ADT/STLExtras.h#L159) 获取类型列表中指定索引的类型：

```cpp
template <size_t I, typename... Ts>
using TypeAtIndex = std::tuple_element_t<I, std::tuple<Ts...>>;
```

## SFINAE与类型分派

LLVM大量使用SFINAE（Substitution Failure Is Not An Error）技术来实现编译期类型分派。

### 基于iterator_category的分派

```cpp
// 使用iterator_category进行分派
template <typename Iterator>
void processImpl(Iterator It, std::random_access_iterator_tag) {
  // 随机访问迭代器的优化实现
}

template <typename Iterator>
void processImpl(Iterator It, std::bidirectional_iterator_tag) {
  // 双向迭代器的实现
}

template <typename Iterator>
void process(Iterator It) {
  using Category = typename std::iterator_traits<Iterator>::iterator_category;
  processImpl(It, Category{});
}
```

### enable_if的使用

LLVM使用 `std::enable_if_t` 来实现条件重载：

```cpp
// 只接受随机访问迭代器
template <typename Iterator>
std::enable_if_t<
  std::is_same_v<
    typename std::iterator_traits<Iterator>::iterator_category,
    std::random_access_iterator_tag
  >,
void> processRandomAccess(Iterator It) {
  // 实现
}

// 使用concepts风格的SFINAE
template <typename T,
          typename = std::enable_if_t<std::is_integral_v<T>>>
void processIntegral(T Value) {
  // 处理整数类型
}

template <typename T,
          typename = std::enable_if_t<std::is_floating_point_v<T>>>
void processFloatingPoint(T Value) {
  // 处理浮点类型
}
```

## 迭代器框架

LLVM在 [iterator.h](/llvm/include/llvm/ADT/iterator.h) 中提供了迭代器辅助类，简化自定义迭代器的实现。

### iterator_facade_base

[iterator_facade_base](/llvm/include/llvm/ADT/iterator.h) 使用CRTP模式帮助实现迭代器：

```cpp
template <typename DerivedT, typename IteratorCategoryT, typename T,
          typename DifferenceTypeT = std::ptrdiff_t, typename PointerT = T *,
          typename ReferenceT = T &>
class iterator_facade_base {
public:
  using iterator_category = IteratorCategoryT;
  using value_type = T;
  using difference_type = DifferenceTypeT;
  using pointer = PointerT;
  using reference = ReferenceT;

protected:
  DerivedT &getDerived() { return *static_cast<DerivedT *>(this); }
  const DerivedT &getDerived() const {
    return *static_cast<const DerivedT *>(this);
  }

public:
  reference operator*() const { return getDerived().operator*(); }
  pointer operator->() const { return &getDerived().operator*(); }
  
  DerivedT &operator++() {
    getDerived().operator++();
    return getDerived();
  }
  
  DerivedT operator++(int) {
    DerivedT Copy(getDerived());
    ++(*this);
    return Copy;
  }
  
  // 其他运算符...
};
```

### 自定义迭代器示例

```cpp
// 使用iterator_facade_base实现自定义迭代器
class MyIterator : public llvm::iterator_facade_base<
                      MyIterator,
                      std::forward_iterator_tag,
                      int> {
  int *Ptr;
  
public:
  MyIterator(int *P) : Ptr(P) {}
  
  // 必须实现的方法
  reference operator*() const { return *Ptr; }
  
  MyIterator &operator++() {
    ++Ptr;
    return *this;
  }
  
  bool operator==(const MyIterator &RHS) const {
    return Ptr == RHS.Ptr;
  }
};
```

## GraphTraits

[GraphTraits](/llvm/include/llvm/ADT/GraphTraits.h) 是LLVM中用于图遍历的重要抽象，允许以统一的方式遍历不同类型的图结构。

### GraphTraits概念

```cpp
// GraphTraits的基本形式
template <typename GraphType>
struct GraphTraits {
  // 节点类型
  using NodeRef = typename GraphType::NodeType;
  
  // 边迭代器
  using ChildIteratorType = typename GraphType::ChildIterator;
  
  // 获取起始节点
  static NodeRef getEntryNode(const GraphType &G);
  
  // 获取子节点迭代器
  static ChildIteratorType child_begin(NodeRef N);
  static ChildIteratorType child_end(NodeRef N);
};
```

### LLVM IR GraphTraits特化

LLVM为多种IR结构提供了GraphTraits特化：

```cpp
// Function的GraphTraits
template <> struct GraphTraits<Function*> {
  using NodeRef = BasicBlock*;
  using ChildIteratorType = succ_iterator;
  
  static NodeRef getEntryNode(Function *F) {
    return &F->getEntryBlock();
  }
  
  static ChildIteratorType child_begin(NodeRef N) {
    return succ_begin(N);
  }
  
  static ChildIteratorType child_end(NodeRef N) {
    return succ_end(N);
  }
};
```

### 图遍历算法

有了GraphTraits，可以编写通用的图遍历算法：

```cpp
// 通用DFS遍历
template <typename GraphT>
void dfs(const GraphT &G) {
  using NodeRef = typename GraphTraits<GraphT>::NodeRef;
  using GT = GraphTraits<GraphT>;
  
  SmallPtrSet<NodeRef, 8> Visited;
  std::vector<NodeRef> Stack;
  
  NodeRef Entry = GT::getEntryNode(G);
  Stack.push_back(Entry);
  
  while (!Stack.empty()) {
    NodeRef N = Stack.back();
    Stack.pop_back();
    
    if (!Visited.insert(N).second)
      continue;
    
    // 处理节点N
    process(N);
    
    // 添加子节点
    for (auto It = GT::child_begin(N), E = GT::child_end(N); It != E; ++It) {
      if (!Visited.count(*It))
        Stack.push_back(*It);
    }
  }
}

// 使用：遍历Function的控制流图
Function *F = ...;
dfs(F);
```

## 职责边界与使用原则

| 技术 | LLVM使用场景 | 用户代码职责 |
|-----|------------|------------|
| `function_traits` | 回调函数处理、函数包装 | 了解其用于类型萃取，不直接修改 |
| `is_one_of` | 编译期类型检查、SFINAE | 在扩展时用于条件分派 |
| `TypesAreDistinct` | 变体类型、标签分发 | 用于验证类型列表唯一性 |
| `GraphTraits` | CFG遍历、Pass基础设施 | 实现自定义图结构的特化 |
| `iterator_facade_base` | IR迭代器、容器迭代器 | 用于实现自定义迭代器 |
| SFINAE/enable_if | 重载解析、特化选择 | 理解其工作原理，正确使用 |

## 总结

LLVM的模板元编程技术是其设计的重要组成部分，它们：

1. **提供编译期类型安全** - 在编译期捕获类型错误
2. **实现零开销抽象** - 没有运行时成本
3. **支持通用算法** - GraphTraits等抽象使算法可重用
4. **简化代码生成** - 通过类型萃取自动生成代码

理解这些技术对于正确使用和扩展LLVM至关重要。
