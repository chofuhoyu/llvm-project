# 1.2.1 内存管理与所有权模型

LLVM使用多种内存管理策略和所有权语义来确保高效、安全地管理资源。本节深入分析LLVM的智能指针、内存管理模式和所有权模型，帮助理解LLVM与用户代码之间的职责边界。

## 侵入式引用计数指针：IntrusiveRefCntPtr

LLVM广泛使用侵入式引用计数来管理需要共享所有权的对象。与 `std::shared_ptr` 不同，`IntrusiveRefCntPtr` 将引用计数直接存储在对象内部，从而减少内存开销和提高性能。

### 核心类定义

位于 [llvm/include/llvm/ADT/IntrusiveRefCntPtr.h](/llvm/include/llvm/ADT/IntrusiveRefCntPtr.h) 的核心类包括：

#### RefCountedBase

[RefCountedBase](/llvm/include/llvm/ADT/IntrusiveRefCntPtr.h#L76) 是一个CRTP（奇异递归模板模式）混入类，为派生类添加引用计数功能：

```cpp
template <class Derived> class RefCountedBase {
  mutable unsigned RefCount = 0;

protected:
  RefCountedBase() = default;
  RefCountedBase(const RefCountedBase &) {}
  RefCountedBase &operator=(const RefCountedBase &) = delete;

public:
  unsigned UseCount() const { return RefCount; }
  
  void Retain() const { ++RefCount; }
  
  void Release() const {
    assert(RefCount > 0 && "Reference count is already zero.");
    if (--RefCount == 0)
      delete static_cast<const Derived *>(this);
  }
};
```

关键特点：
- 使用CRTP模式，派生类将自己作为模板参数
- 引用计数是 `mutable` 的，允许在const对象上修改
- 当引用计数降为0时，对象自删除
- Debug模式下析构函数有断言检查引用计数为0

#### ThreadSafeRefCountedBase

[ThreadSafeRefCountedBase](/llvm/include/llvm/ADT/IntrusiveRefCntPtr.h#L108) 是线程安全版本，使用原子操作：

```cpp
template <class Derived> class ThreadSafeRefCountedBase {
  mutable std::atomic<int> RefCount{0};

public:
  void Retain() const { 
    RefCount.fetch_add(1, std::memory_order_relaxed); 
  }
  
  void Release() const {
    int NewRefCount = RefCount.fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (NewRefCount == 0)
      delete static_cast<const Derived *>(this);
  }
};
```

使用 `std::memory_order_acq_rel` 确保释放操作的可见性。

#### IntrusiveRefCntPtr

[IntrusiveRefCntPtr](/llvm/include/llvm/ADT/IntrusiveRefCntPtr.h) 是实际的智能指针类，管理引用计数：

```cpp
template <class T> class IntrusiveRefCntPtr {
  T *Obj = nullptr;

public:
  IntrusiveRefCntPtr() = default;
  
  IntrusiveRefCntPtr(T *Obj) : Obj(Obj) {
    if (Obj) Obj->Retain();
  }
  
  IntrusiveRefCntPtr(const IntrusiveRefCntPtr &S) : Obj(S.Obj) {
    if (Obj) Obj->Retain();
  }
  
  IntrusiveRefCntPtr(IntrusiveRefCntPtr &&S) : Obj(S.Obj) {
    S.Obj = nullptr;
  }
  
  ~IntrusiveRefCntPtr() {
    if (Obj) Obj->Release();
  }
  
  IntrusiveRefCntPtr &operator=(const IntrusiveRefCntPtr &S) {
    if (Obj) Obj->Release();
    Obj = S.Obj;
    if (Obj) Obj->Retain();
    return *this;
  }
  
  T *get() const { return Obj; }
  T &operator*() const { return *Obj; }
  T *operator->() const { return Obj; }
  
  explicit operator bool() const { return Obj != nullptr; }
};
```

### 使用模式

```cpp
// 定义可引用计数的类
class MyClass : public RefCountedBase<MyClass> {
public:
  void doSomething() { /* ... */ }
};

// 使用示例
void foo() {
  // 创建对象，引用计数为1
  IntrusiveRefCntPtr<MyClass> Ptr1(new MyClass());
  
  // 复制，引用计数增加到2
  IntrusiveRefCntPtr<MyClass> Ptr2(Ptr1);
  
  // 移动，Ptr1变为null，引用计数保持2
  IntrusiveRefCntPtr<MyClass> Ptr3(std::move(Ptr1));
  assert(Ptr1 == nullptr);
  
  // 重置，引用计数减少到1
  Ptr2.reset();
  
  // 函数返回，Ptr3析构，引用计数降到0，对象被删除
}
```

### 与LLVM类型系统的集成

`IntrusiveRefCntPtr` 可以与LLVM的 `isa<>`、`dyn_cast<>` 等类型检查工具配合使用：

```cpp
IntrusiveRefCntPtr<BaseClass> Ptr(new DerivedClass());
DerivedClass *Derived = dyn_cast<DerivedClass>(Ptr);  // 不需要 .get()
if (isa<DerivedClass>(Ptr)) { /* ... */ }
```

### 自定义Retain/Release

除了继承 `RefCountedBase`，还可以通过特化 `IntrusiveRefCntPtrInfo` 来提供自定义的引用计数管理：

```cpp
template <> struct IntrusiveRefCntPtrInfo<MyType> {
  static void retain(MyType *obj) { obj->customRetain(); }
  static void release(MyType *obj) { obj->customRelease(); }
};
```

## 指针整数对：PointerIntPair

LLVM经常需要同时存储一个指针和一个小整数。[PointerIntPair](/llvm/include/llvm/ADT/PointerIntPair.h) 利用指针的低位对齐特性，将整数编码到指针本身中，节省空间。

### 工作原理

现代系统中，由于内存对齐要求，指针的低位通常是0。例如：
- 8字节对齐的指针：低3位为0
- 4字节对齐的指针：低2位为0

`PointerIntPair` 利用这些低位来存储小整数。

### 使用示例

```cpp
#include "llvm/ADT/PointerIntPair.h"

// 存储一个指针和1位整数
PointerIntPair<void*, 1, bool> PIP;
PIP.setPointer(somePointer);
PIP.setInt(true);

void *Ptr = PIP.getPointer();
bool Flag = PIP.getInt();

// 同时设置
PIP.setPointerAndInt(anotherPointer, false);
```

### 模板参数

```cpp
template <typename PointerTy, 
          unsigned IntBits, 
          typename IntType = unsigned,
          typename PtrTraits = PointerLikeTypeTraits<PointerTy>,
          typename Info = PointerIntPairInfo<PointerTy, IntBits, PtrTraits>>
class PointerIntPair;
```

- `PointerTy`: 指针类型
- `IntBits`: 要存储的整数位数
- `IntType`: 整数类型（默认unsigned）
- `PtrTraits`: 指针特性，用于确定可用的低位数量

### 嵌套使用

由于 `PointerIntPair` 将整数放在高位，支持嵌套：

```cpp
// 存储两个bool
PointerIntPair<PointerIntPair<void*, 1, bool>, 1, bool> DoubleBool;
```

## 指针联合：PointerUnion

[PointerUnion](/llvm/include/llvm/ADT/PointerUnion.h) 实现了一个带标签的指针联合体，可以存储多种不同类型的指针，同时知道当前存储的是哪种类型。

### 使用示例

```cpp
#include "llvm/ADT/PointerUnion.h"

PointerUnion<int*, float*> PU;

// 设置为int*
int IntVal = 42;
PU = &IntVal;

// 检查类型
if (PU.is<int*>()) {
  int *IP = PU.get<int*>();
  // ...
}

// 安全转换
if (int *IP = PU.dyn_cast<int*>()) {
  // 使用IP
} else if (float *FP = PU.dyn_cast<float*>()) {
  // 使用FP
}
```

### 实现原理

`PointerUnion` 利用指针的低位存储类型标签，类似于 `PointerIntPair`。支持的类型数量取决于可用的低位数量。

## 独占所有权

LLVM在可能的情况下优先使用独占所有权，主要通过以下方式：

### std::unique_ptr

LLVM广泛使用C++11的 `std::unique_ptr` 来管理独占所有权的对象。

### llvm::OwningPtr（历史）

旧版本LLVM使用 `OwningPtr`，现在已被 `std::unique_ptr` 取代。

## 引用传递与值传递的选择原则

LLVM代码遵循以下原则来决定是使用引用还是值传递：

1. **小对象、简单类型** - 值传递
   - `llvm::ArrayRef`
   - `llvm::StringRef`
   - 小型POD类型

2. **大对象、复杂对象** - const引用传递
   - `std::string`
   - `std::vector`
   - 大型类

3. **需要转移所有权** - 右值引用/移动
   - `std::unique_ptr`
   - `llvm::IntrusiveRefCntPtr`（也可以复制）

## 非所有者引用：ArrayRef & StringRef

LLVM大量使用非所有者引用类型来避免不必要的复制：

- `ArrayRef<T>` - 数组的非所有者引用
- `StringRef` - 字符串的非所有者引用

这些类型不拥有数据，只是提供一个视图。详见[1.2.2节 ADT库详解](adt-library.md)。

## 内存池与分配器策略

LLVM在某些场景下使用内存池来提高性能：

1. **BumpPtrAllocator** - 线性分配器，用于批量分配后一起释放
2. **SpecificBumpPtrAllocator** - 特定类型的BumpPtrAllocator
3. **MallocAllocator** - 简单的malloc/free包装

这些分配器常用于AST、IR等需要创建大量小对象且生命周期相近的场景。

## 职责边界总结

| 内存管理方式 | LLVM使用场景 | 用户代码职责 |
|------------|------------|------------|
| `IntrusiveRefCntPtr` | 共享所有权的对象 | 正确管理指针生命周期，避免循环引用 |
| `std::unique_ptr` | 独占所有权的对象 | 确保不被意外复制 |
| `ArrayRef/StringRef` | 函数参数、返回值 | 确保底层数据在引用有效期内存在 |
| `PointerIntPair` | 指针+小标记 | 了解对齐限制 |
| `PointerUnion` | 多态指针 | 使用类型检查方法安全访问 |

理解这些内存管理模式对于正确编写和修改LLVM代码至关重要。
