# 1.2.5 LLVM类型系统

LLVM的类型系统是IR的核心组成部分，设计为不可变且唯一化的。本节深入分析LLVM类型系统的设计和实现。

## 类型系统的核心特性

从 [Type.h](/llvm/include/llvm/IR/Type.h#L38) 的注释可以看到两个关键特性：

1. **不可变性 (Immutability)** - 一旦创建，类型永远不会改变
2. **唯一化 (Uniquing)** - 特定类型只有一个实例存在，类型相等性比较就是指针比较

```cpp
/// The instances of the Type class are immutable: once they are created,
/// they are never changed.  Also note that only one instance of a particular
/// type is ever created.  Thus seeing if two types are equal is a matter of
/// doing a trivial pointer comparison.
```

## Type基类

[Type](/llvm/include/llvm/IR/Type.h#L46) 是所有LLVM类型的基类：

```cpp
class Type {
private:
  LLVMContext &Context;  // 类型所属的上下文
  TypeID ID : 8;         // 类型ID
  unsigned SubclassData : 24;  // 子类数据空间
  
protected:
  unsigned NumContainedTys = 0;        // 包含的类型数量
  Type * const *ContainedTys = nullptr; // 包含的类型数组
  
public:
  enum TypeID {
    // 基本类型
    HalfTyID = 0,      // 16位浮点
    BFloatTyID,        // 16位bfloat
    FloatTyID,         // 32位浮点
    DoubleTyID,        // 64位浮点
    X86_FP80TyID,      // 80位x87浮点
    FP128TyID,         // 128位浮点
    PPC_FP128TyID,     // 128位PowerPC浮点
    VoidTyID,          // void类型
    LabelTyID,         // 标签类型
    MetadataTyID,      // 元数据类型
    X86_AMXTyID,       // X86 AMX向量
    TokenTyID,         // 令牌类型
    
    // 派生类型
    IntegerTyID,        // 任意位宽整数
    ByteTyID,           // 任意位宽字节
    FunctionTyID,       // 函数类型
    PointerTyID,        // 指针类型
    StructTyID,         // 结构体类型
    ArrayTyID,          // 数组类型
    FixedVectorTyID,    // 固定宽度SIMD向量
    ScalableVectorTyID, // 可伸缩SIMD向量
    TypedPointerTyID,   // 带类型的指针（GPU目标）
    TargetExtTyID,      // 目标扩展类型
  };
};
```

## 类型分类

### 基本类型 (Primitive Types)

| 类型ID | C++类 | 说明 |
|-------|------|------|
| `VoidTyID` | `Type` | 无类型 |
| `HalfTyID` | `Type` | 16位IEEE浮点 |
| `BFloatTyID` | `Type` | 16位bfloat (7位尾数) |
| `FloatTyID` | `Type` | 32位IEEE浮点 |
| `DoubleTyID` | `Type` | 64位IEEE浮点 |
| `X86_FP80TyID` | `Type` | 80位x87扩展浮点 |
| `FP128TyID` | `Type` | 128位IEEE浮点 |
| `PPC_FP128TyID` | `Type` | 128位PowerPC双浮点 |
| `LabelTyID` | `Type` | 标签类型 |
| `MetadataTyID` | `Type` | 元数据类型 |
| `TokenTyID` | `Type` | 令牌类型 |

### 派生类型 (Derived Types)

| 类型ID | C++类 | 说明 |
|-------|------|------|
| `IntegerTyID` | `IntegerType` | 任意位宽整数 |
| `ByteTyID` | `ByteType` | 任意位宽字节 |
| `FunctionTyID` | `FunctionType` | 函数类型 |
| `PointerTyID` | `PointerType` | 指针类型 |
| `StructTyID` | `StructType` | 结构体类型 |
| `ArrayTyID` | `ArrayType` | 数组类型 |
| `FixedVectorTyID` | `FixedVectorType` | 固定宽度向量 |
| `ScalableVectorTyID` | `ScalableVectorType` | 可伸缩向量 |
| `TargetExtTyID` | `TargetExtType` | 目标扩展类型 |

## 类型唯一化 (Uniquing) 机制

类型唯一化是LLVM类型系统的核心特性，通过LLVMContext实现。

### 工作原理

1. **类型池** - LLVMContext内部维护一个类型池
2. **工厂方法** - 类型只能通过静态工厂方法创建
3. **查找与创建** - 工厂方法先查找是否已存在该类型，不存在才创建
4. **指针比较** - 相等性检查只需比较指针

### 类型获取示例

```cpp
#include "llvm/IR/Type.h"
#include "llvm/IR/LLVMContext.h"

LLVMContext &Ctx = ...;

// 获取基本类型
Type *VoidTy = Type::getVoidTy(Ctx);
Type *Int32Ty = IntegerType::get(Ctx, 32);
Type *FloatTy = Type::getFloatTy(Ctx);
Type *DoubleTy = Type::getDoubleTy(Ctx);

// 获取指针类型
PointerType *Int32PtrTy = PointerType::get(Int32Ty, 0);

// 获取函数类型
Type *ResultTy = Int32Ty;
std::vector<Type*> ParamTys = {Int32Ty, Int32Ty};
FunctionType *FuncTy = FunctionType::get(ResultTy, ParamTys, false);

// 获取结构体类型
std::vector<Type*> FieldTys = {Int32Ty, FloatTy};
StructType *StructTy = StructType::get(Ctx, FieldTys);

// 获取数组类型
ArrayType *ArrayTy = ArrayType::get(Int32Ty, 100);  // [100 x i32]

// 获取向量类型
FixedVectorType *VecTy = FixedVectorType::get(Int32Ty, 4);  // <4 x i32>
```

### 类型相等性检查

由于唯一化，类型比较非常高效：

```cpp
// 正确：直接比较指针
if (Ty1 == Ty2) { /* 类型相同 */ }

// 错误：不需要这样做
if (Ty1->getTypeID() == Ty2->getTypeID()) { /* 不够精确 */ }

// 使用isa/dyn_cast进行类型检查
if (isa<IntegerType>(Ty)) {
  IntegerType *IntTy = cast<IntegerType>(Ty);
  unsigned BitWidth = IntTy->getBitWidth();
}

if (auto *StructTy = dyn_cast<StructType>(Ty)) {
  // 处理结构体类型
}
```

## 类型检查方法

[Type](/llvm/include/llvm/IR/Type.h#L140) 提供了丰富的类型检查方法：

```cpp
// 基本类型检查
bool isVoidTy() const { return getTypeID() == VoidTyID; }
bool isHalfTy() const { return getTypeID() == HalfTyID; }
bool isFloatTy() const { return getTypeID() == FloatTyID; }
bool isDoubleTy() const { return getTypeID() == DoubleTyID; }
bool isIntegerTy() const { return getTypeID() == IntegerTyID; }
bool isPointerTy() const { return getTypeID() == PointerTyID; }
bool isStructTy() const { return getTypeID() == StructTyID; }
bool isArrayTy() const { return getTypeID() == ArrayTyID; }
bool isFunctionTy() const { return getTypeID() == FunctionTyID; }

// 复合检查
bool isFloatingPointTy() const {
  return isIEEELikeFPTy() || getTypeID() == X86_FP80TyID ||
         getTypeID() == PPC_FP128TyID;
}

bool isIEEELikeFPTy() const {
  switch (getTypeID()) {
  case DoubleTyID:
  case FloatTyID:
  case HalfTyID:
  case BFloatTyID:
  case FP128TyID:
    return true;
  default:
    return false;
  }
}

// 整数类型特有的检查
bool isIntegerTy(unsigned Bitwidth) const {
  return isIntegerTy() && cast<IntegerType>(this)->getBitWidth() == Bitwidth;
}

// 指针/标量/聚合类型
bool isPtrOrPtrVectorTy() const;
bool isScalarTy() const;
bool isAggregateType() const;
```

## 常用类型

### IntegerType

任意位宽的整数类型：

```cpp
#include "llvm/IR/DerivedTypes.h"

// 创建整数类型
IntegerType *Int8Ty = IntegerType::get(Ctx, 8);
IntegerType *Int16Ty = IntegerType::get(Ctx, 16);
IntegerType *Int32Ty = IntegerType::get(Ctx, 32);
IntegerType *Int64Ty = IntegerType::get(Ctx, 64);
IntegerType *Int1Ty = IntegerType::get(Ctx, 1);  // i1 (布尔)

// 获取位宽
unsigned BitWidth = Int32Ty->getBitWidth();  // 32
```

### PointerType

指针类型（LLVM 15+中通常是不透明指针）：

```cpp
#include "llvm/IR/DerivedTypes.h"

// 创建指针类型（地址空间0）
PointerType *PtrTy = PointerType::get(Ctx, 0);

// 创建指定地址空间的指针
PointerType *AS1PtrTy = PointerType::get(Ctx, 1);

// 获取地址空间
unsigned AS = PtrTy->getAddressSpace();  // 0
```

### FunctionType

函数类型：

```cpp
#include "llvm/IR/DerivedTypes.h"

// 返回类型 + 参数类型 + 是否可变参数
Type *RetTy = Int32Ty;
std::vector<Type*> ParamTys = {Int32Ty, Int32Ty};
bool IsVarArg = false;

FunctionType *FuncTy = FunctionType::get(RetTy, ParamTys, IsVarArg);

// 访问函数类型
Type *ResultType = FuncTy->getReturnType();
unsigned NumParams = FuncTy->getNumParams();
Type *Param0 = FuncTy->getParamType(0);
bool IsVarArg = FuncTy->isVarArg();
```

### StructType

结构体类型：

```cpp
#include "llvm/IR/DerivedTypes.h"

// 字面量结构体（匿名，按类型相等）
std::vector<Type*> FieldTys = {Int32Ty, FloatTy};
StructType *StructTy = StructType::get(Ctx, FieldTys);

// 命名结构体（按名称唯一化）
StructType *NamedStructTy = StructType::create(Ctx, "mystruct");
NamedStructTy->setBody(FieldTys);

// 访问结构体
unsigned NumElements = StructTy->getNumElements();
Type *Elem0 = StructTy->getElementType(0);
bool IsPacked = StructTy->isPacked();
bool IsLiteral = StructTy->isLiteral();
StringRef Name = StructTy->getName();  // 只有命名结构体才有
```

### ArrayType

数组类型：

```cpp
#include "llvm/IR/DerivedTypes.h"

// 数组类型：[100 x i32]
uint64_t NumElements = 100;
ArrayType *ArrayTy = ArrayType::get(Int32Ty, NumElements);

// 访问数组类型
Type *ElemTy = ArrayTy->getElementType();
uint64_t NumElems = ArrayTy->getNumElements();
```

### VectorType

向量类型（SIMD）：

```cpp
#include "llvm/IR/DerivedTypes.h"

// 固定宽度向量：<4 x i32>
unsigned NumElems = 4;
FixedVectorType *FixedVecTy = FixedVectorType::get(Int32Ty, NumElems);

// 可伸缩向量：<vscale x 4 x i32>
ScalableVectorType *ScalableVecTy = ScalableVectorType::get(Int32Ty, NumElems);

// 访问向量类型
Type *ElemTy = FixedVecTy->getElementType();
unsigned MinElems = FixedVecTy->getMinNumElements();
bool IsScalable = FixedVecTy->isScalable();  // false
```

## 类型的包含关系

某些类型包含其他类型，可以通过 `ContainedTys` 访问：

```cpp
// 函数类型包含返回类型和参数类型
FunctionType *FT = ...;
for (unsigned I = 0, E = FT->getNumContainedTypes(); I != E; ++I) {
  Type *ContainedTy = FT->getContainedType(I);
}

// 结构体类型包含字段类型
StructType *ST = ...;
for (unsigned I = 0, E = ST->getNumContainedTypes(); I != E; ++I) {
  Type *FieldTy = ST->getContainedType(I);
}

// 数组类型包含元素类型
ArrayType *AT = ...;
Type *ElemTy = AT->getContainedType(0);
```

## 类型打印

```cpp
#include "llvm/Support/raw_ostream.h"

// 打印到流
Type *Ty = ...;
Ty->print(llvm::outs());

// 打印到字符串
std::string TypeStr;
llvm::raw_string_ostream OS(TypeStr);
Ty->print(OS);
OS.flush();

// 调试打印
Ty->dump();  // 打印到stderr
```

常见类型的打印结果：

| 类型 | 打印结果 |
|-----|---------|
| void | `void` |
| i1 | `i1` |
| i32 | `i32` |
| float | `float` |
| double | `double` |
| i32* | `ptr` (LLVM 15+) |
| [100 x i32] | `[100 x i32]` |
| <4 x i32> | `<4 x i32>` |
| {i32, float} | `{ i32, float }` |
| i32 (i32, i32) | `i32 (i32, i32)` |

## 职责边界总结

| 操作 | LLVM职责 | 用户代码职责 |
|-----|---------|------------|
| 类型创建 | 通过工厂方法管理唯一化 | 只通过工厂方法获取类型 |
| 类型比较 | 确保指针相等性等价于类型相等 | 使用指针比较，不要手动比较内容 |
| 类型检查 | 提供isa/dyn_cast机制 | 使用isa/dyn_cast安全地类型转换 |
| 类型生命周期 | 由LLVMContext管理类型池 | 不需要（也不能）删除类型 |

## 总结

LLVM类型系统的关键要点：

1. **不可变** - 创建后永远不变
2. **唯一化** - 每个类型只有一个实例，指针比较即可
3. **工厂方法** - 只能通过静态工厂方法创建类型
4. **LLVMContext** - 类型属于特定的Context，由Context管理
5. **isa/dyn_cast** - 使用LLVM RTTI安全地进行类型检查和转换

理解这些特性对于正确使用和修改LLVM IR至关重要。
