# 1.2.2 ADT库详解

ADT（Algebraic Data Types，代数数据类型）是LLVM提供的一套自定义数据结构和容器库。这些容器针对编译器的特定使用场景进行了优化，相比STL容器通常具有更好的性能特性或更适合的接口。

## 设计理念

LLVM的ADT库设计遵循以下原则：

1. **性能优先** - 针对编译器工作负载优化
2. **内存效率** - 减少内存分配和碎片
3. **简单实用** - 接口简洁，易于正确使用
4. **堆栈优先** - 小型数据优先使用栈上存储

## 非所有者引用类型

### ArrayRef

[ArrayRef](/llvm/include/llvm/ADT/ArrayRef.h) 是一个数组的常量引用视图，不拥有数据所有权。它是LLVM中最常用的类型之一。

#### 核心定义

```cpp
template<typename T>
class ArrayRef {
private:
  const T *Data = nullptr;
  size_type Length = 0;

public:
  // 构造函数
  ArrayRef() = default;
  ArrayRef(const T &OneElt);
  ArrayRef(const T *data, size_t length);
  ArrayRef(const T *begin, const T *end);
  
  // 从容器构造（隐式转换）
  template<typename C>
  ArrayRef(const C &V) : Data(V.data()), Length(V.size()) {}
  
  // 从C数组构造
  template<size_t N>
  ArrayRef(const T (&Arr)[N]) : Data(Arr), Length(N) {}
  
  // 从initializer_list构造
  ArrayRef(std::initializer_list<T> Vec);
  
  // 迭代器
  iterator begin() const { return Data; }
  iterator end() const { return Data + Length; }
  
  // 容量查询
  bool empty() const { return Length == 0; }
  size_t size() const { return Length; }
  
  // 元素访问
  const T *data() const { return Data; }
  const T &operator[](size_t Index) const { return Data[Index]; }
  const T &front() const { return Data[0]; }
  const T &back() const { return Data[Length-1]; }
  
  // 切片操作
  ArrayRef<T> slice(size_t N, size_t M) const;
  ArrayRef<T> drop_front(size_t N = 1) const;
  ArrayRef<T> drop_back(size_t N = 1) const;
  ArrayRef<T> take_front(size_t N = 1) const;
  ArrayRef<T> take_back(size_t N = 1) const;
};
```

#### 使用示例

```cpp
#include "llvm/ADT/ArrayRef.h"

// 函数参数：使用ArrayRef接受任何连续序列
void processArray(llvm::ArrayRef<int> Values) {
  for (int V : Values) {
    // 处理V
  }
}

// 使用方式1：传入std::vector
std::vector<int> Vec = {1, 2, 3, 4};
processArray(Vec);  // 隐式转换

// 使用方式2：传入C数组
int Arr[] = {1, 2, 3, 4};
processArray(Arr);  // 隐式转换

// 使用方式3：传入单个元素
int X = 42;
processArray(X);  // 隐式转换

// 使用方式4：传入initializer_list
processArray({1, 2, 3, 4});

// 使用方式5：切片
llvm::ArrayRef<int> All = Vec;
llvm::ArrayRef<int> Slice = All.slice(1, 2);  // {2, 3}
llvm::ArrayRef<int> WithoutFirst = All.drop_front();  // {2, 3, 4}
llvm::ArrayRef<int> FirstTwo = All.take_front(2);  // {1, 2}
```

#### MutableArrayRef

[MutableArrayRef](/llvm/include/llvm/ADT/ArrayRef.h) 是 `ArrayRef` 的可变版本，允许修改底层元素：

```cpp
void modifyArray(llvm::MutableArrayRef<int> Values) {
  for (int &V : Values) {
    V *= 2;
  }
}

std::vector<int> Vec = {1, 2, 3};
modifyArray(Vec);  // Vec变为 {2, 4, 6}
```

#### 重要注意事项

**生命周期警告**：`ArrayRef` 不拥有数据，必须确保底层数据在 `ArrayRef` 的整个生命周期内有效。

```cpp
// 错误示例：返回指向临时对象的ArrayRef
llvm::ArrayRef<int> bad() {
  std::vector<int> Vec = {1, 2, 3};
  return Vec;  // Vec被销毁，ArrayRef指向无效内存
}

// 正确示例
std::vector<int> good() {
  std::vector<int> Vec = {1, 2, 3};
  return Vec;  // 返回vector，转移所有权
}
```

### StringRef

[StringRef](/llvm/include/llvm/ADT/StringRef.h) 是 `ArrayRef<char>` 的特化，用于字符串视图，概念上类似 `std::string_view`（但在C++17之前就存在）。

#### 核心特性

```cpp
class StringRef {
public:
  // 构造函数
  StringRef() = default;
  StringRef(const char *Str);  // 从C字符串构造
  StringRef(const char *Str, size_t Length);
  StringRef(const std::string &Str);  // 从std::string构造
  
  // 字符串操作
  bool empty() const;
  size_t size() const;
  const char *data() const;
  
  // 转换
  std::string str() const;
  operator std::string() const { return str(); }
  
  // 比较
  int compare(StringRef RHS) const;
  bool equals(StringRef RHS) const;
  
  // 查找
  size_t find(char C, size_t From = 0) const;
  size_t find(StringRef Str, size_t From = 0) const;
  bool contains(StringRef Other) const;
  bool startswith(StringRef Prefix) const;
  bool endswith(StringRef Suffix) const;
  
  // 切片
  StringRef substr(size_t Start, size_t N = npos) const;
  StringRef drop_front(size_t N = 1) const;
  StringRef drop_back(size_t N = 1) const;
  
  // 修改（返回新StringRef）
  StringRef ltrim(char Char = ' ') const;
  StringRef rtrim(char Char = ' ') const;
  StringRef trim(char Char = ' ') const;
};
```

#### 使用示例

```cpp
#include "llvm/ADT/StringRef.h"

// 函数参数：使用StringRef接受任何字符串
void processString(llvm::StringRef Str) {
  if (Str.startswith("llvm.")) {
    // 处理LLVM相关字符串
  }
}

// 使用方式
processString("hello");              // 字符串字面量
processString(std::string("world")); // std::string
processString("llvm.test", 5);       // 指针+长度

// 字符串操作
llvm::StringRef S = "  hello world  ";
llvm::StringRef Trimmed = S.trim();      // "hello world"
llvm::StringRef NoPrefix = S.drop_front(2); // "hello world  "
llvm::StringRef First5 = S.substr(0, 5);   // "  hel"

// 比较
if (Trimmed == "hello world") { /* ... */ }
if (Trimmed.startswith("hello")) { /* ... */ }
```

#### 与std::string的转换

```cpp
// StringRef -> std::string
llvm::StringRef SR = "hello";
std::string S = SR.str();  // 显式转换
std::string S2 = SR;       // 隐式转换

// std::string -> StringRef
std::string S = "hello";
llvm::StringRef SR = S;  // 隐式转换
```

## 小型容器

### SmallVector

[SmallVector](/llvm/include/llvm/ADT/SmallVector.h) 是一个优化的向量容器，在元素数量少时优先使用栈上存储，避免动态内存分配。

#### 核心设计

```cpp
template<typename T, unsigned N = 4>
class SmallVector : public SmallVectorImpl<T> {
  // 内部存储：N个元素的栈上缓冲区
  char InlineElts[N * sizeof(T)];
  
public:
  // 构造函数
  SmallVector() = default;
  SmallVector(size_t Size, const T &Value = T());
  SmallVector(std::initializer_list<T> IL);
  
  // 迭代器
  iterator begin() { return this->BeginX; }
  iterator end() { return this->BeginX + this->Size; }
  
  // 容量
  size_t size() const { return this->Size; }
  size_t capacity() const { return this->Capacity; }
  bool empty() const { return this->Size == 0; }
  
  // 元素访问
  T &operator[](size_t idx) { return this->BeginX[idx]; }
  T &front() { return this->BeginX[0]; }
  T &back() { return this->BeginX[this->Size-1]; }
  
  // 修改操作
  void push_back(const T &Elt);
  void push_back(T &&Elt);
  void pop_back();
  void clear();
  void resize(size_t N);
  void reserve(size_t N);
  
  // 检查是否使用内联存储
  bool isSmall() const { return this->BeginX == getFirstEl(); }
};
```

#### 使用示例

```cpp
#include "llvm/ADT/SmallVector.h"

// 存储最多4个元素在栈上
llvm::SmallVector<int, 4> Vec;

// 添加元素（前4个在栈上）
Vec.push_back(1);
Vec.push_back(2);
Vec.push_back(3);
Vec.push_back(4);

assert(Vec.isSmall());  // true，仍使用内联存储

// 第5个元素触发堆分配
Vec.push_back(5);
assert(!Vec.isSmall());  // false，现在使用堆存储

// 接口与std::vector兼容
for (int V : Vec) { /* ... */ }
Vec[0] = 10;
Vec.pop_back();

// 初始化
llvm::SmallVector<int, 4> Vec2 = {1, 2, 3, 4};
llvm::SmallVector<int, 4> Vec3(10, 0);  // 10个0

// 转换为ArrayRef
llvm::ArrayRef<int> Ref = Vec;
```

#### 选择内联大小

选择 `N` 的原则：
- **N = 0-8**: 最常见，适用于大多数小型列表
- **N = 16-32**: 较大的临时缓冲区
- **避免过大的N**: 会增加栈帧大小，可能导致栈溢出

```cpp
// 推荐：小型临时列表
llvm::SmallVector<llvm::Value*, 8> Operands;

// 推荐：参数列表
llvm::SmallVector<llvm::Type*, 4> ParamTypes;

// 避免：太大的内联存储
llvm::SmallVector<int, 1024> TooBig;  // 栈上4KB，可能有问题
```

### SmallPtrSet

[SmallPtrSet](/llvm/include/llvm/ADT/SmallPtrSet.h) 是针对指针优化的小型集合容器。

```cpp
#include "llvm/ADT/SmallPtrSet.h"

llvm::SmallPtrSet<llvm::Value*, 8> Visited;

// 插入
Visited.insert(Val);

// 查找
if (Visited.count(Val)) { /* ... */ }

// 迭代
for (llvm::Value *V : Visited) { /* ... */ }

// 删除
Visited.erase(Val);
Visited.clear();
```

## 哈希表容器

### DenseMap

[DenseMap](/llvm/include/llvm/ADT/DenseMap.h) 是一个密集的哈希表实现，相比 `std::unordered_map` 具有更好的缓存局部性和更高的内存效率。

#### 核心特性

```cpp
template<typename KeyT, typename ValueT,
         typename KeyInfoT = DenseMapInfo<KeyT>>
class DenseMap {
public:
  // 类型
  using key_type = KeyT;
  using mapped_type = ValueT;
  using value_type = std::pair<KeyT, ValueT>;
  
  // 构造函数
  DenseMap() = default;
  explicit DenseMap(unsigned NumInitBuckets);
  
  // 容量
  bool empty() const;
  unsigned size() const;
  
  // 插入
  std::pair<iterator, bool> insert(const std::pair<KeyT, ValueT> &KV);
  template<typename InputIt>
  void insert(InputIt I, InputIt E);
  
  // 查询
  iterator find(const KeyT &Key);
  const_iterator find(const KeyT &Key) const;
  bool count(const KeyT &Key) const;
  ValueT &operator[](const KeyT &Key);
  ValueT &lookup(const KeyT &Key, const ValueT &Default = ValueT());
  
  // 删除
  void erase(iterator I);
  size_type erase(const KeyT &Key);
  void clear();
  
  // 迭代器访问
  iterator begin();
  iterator end();
  
  // 键和值范围
  auto keys();
  auto values();
};
```

#### 使用示例

```cpp
#include "llvm/ADT/DenseMap.h"

llvm::DenseMap<llvm::Value*, unsigned> ValueNumbering;

// 插入
ValueNumbering[Val] = 42;
auto Result = ValueNumbering.insert({Val, 42});
if (Result.second) { /* 新插入 */ }

// 查询
if (ValueNumbering.count(Val)) {
  unsigned Num = ValueNumbering[Val];
}

// 查找，带默认值
unsigned Num = ValueNumbering.lookup(Val, 0);

// 迭代
for (auto &KV : ValueNumbering) {
  llvm::Value *Key = KV.first;
  unsigned Val = KV.second;
}

// 迭代键
for (llvm::Value *Key : ValueNumbering.keys()) { /* ... */ }

// 迭代值
for (unsigned Val : ValueNumbering.values()) { /* ... */ }

// 删除
ValueNumbering.erase(Val);
ValueNumbering.clear();
```

#### DenseMapInfo

`DenseMap` 需要 `DenseMapInfo` 来提供特殊的空标记和墓碑标记：

```cpp
// 指针的DenseMapInfo特化
template<typename T>
struct DenseMapInfo<T*> {
  static inline T* getEmptyKey() {
    return (T*)(uintptr_t)-1;  // 空键：全1
  }
  static inline T* getTombstoneKey() {
    return (T*)(uintptr_t)-2;  // 墓碑键：全1减1
  }
  static unsigned getHashValue(const T* Ptr) {
    return (unsigned((uintptr_t)Ptr) >> 4) ^
           (unsigned((uintptr_t)Ptr) >> 9);
  }
  static bool isEqual(const T* LHS, const T* RHS) {
    return LHS == RHS;
  }
};
```

#### 自定义DenseMapInfo

```cpp
namespace llvm {
template<> struct DenseMapInfo<MyType> {
  static inline MyType getEmptyKey() {
    return MyType::EmptyMarker;
  }
  static inline MyType getTombstoneKey() {
    return MyType::TombstoneMarker;
  }
  static unsigned getHashValue(const MyType &Val) {
    return hash_value(Val);
  }
  static bool isEqual(const MyType &LHS, const MyType &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm
```

### StringMap

[StringMap](/llvm/include/llvm/ADT/StringMap.h) 是针对字符串键优化的哈希表。

```cpp
#include "llvm/ADT/StringMap.h"

llvm::StringMap<int> SymbolTable;

// 插入
SymbolTable["foo"] = 1;
SymbolTable.getOrInsertValue("bar", 0);

// 查询
if (SymbolTable.count("foo")) {
  int Val = SymbolTable["foo"];
}

// 查找
auto It = SymbolTable.find("foo");
if (It != SymbolTable.end()) {
  StringRef Key = It->first();
  int &Val = It->second;
}
```

## 其他常用容器

### SmallString

[SmallString](/llvm/include/llvm/ADT/SmallString.h) 是 `SmallVector<char, N>` 的包装，用于字符串操作。

```cpp
#include "llvm/ADT/SmallString.h"

llvm::SmallString<64> Path;
Path += "/usr";
Path += "/local";
Path += "/bin";

// 转换
StringRef Ref = Path;
std::string S = Path.str().str();
```

### MapVector

[MapVector](/llvm/include/llvm/ADT/MapVector.h) 结合了 `std::map` 的有序性和 `DenseMap` 的快速查找。

```cpp
#include "llvm/ADT/MapVector.h"

llvm::MapVector<llvm::Value*, unsigned> OrderedMap;
OrderedMap[Val1] = 1;
OrderedMap[Val2] = 2;

// 按插入顺序迭代
for (auto &KV : OrderedMap) { /* ... */ }
```

### SetVector

[SetVector](/llvm/include/llvm/ADT/SetVector.h) 是有序集合，保持插入顺序。

```cpp
#include "llvm/ADT/SetVector.h"

llvm::SetVector<llvm::Value*> WorkList;
WorkList.insert(Val1);
WorkList.insert(Val2);

// 按插入顺序迭代
for (llvm::Value *V : WorkList) { /* ... */ }
```

## 职责边界总结

| 场景 | LLVM容器 | 说明 |
|-----|----------|------|
| 函数参数 | `ArrayRef`/`StringRef` | 非所有者引用，避免复制 |
| 小型列表 | `SmallVector<N>` | 栈上优先，减少分配 |
| 指针集合 | `SmallPtrSet<N>` | 指针专用优化 |
| 快速查找 | `DenseMap` | 高效哈希表 |
| 字符串键 | `StringMap` | 字符串专用优化 |
| 有序映射 | `MapVector` | 保持插入顺序 |
| 有序集合 | `SetVector` | 保持插入顺序 |

理解这些ADT容器的特性和使用场景，是编写高效LLVM代码的关键。
