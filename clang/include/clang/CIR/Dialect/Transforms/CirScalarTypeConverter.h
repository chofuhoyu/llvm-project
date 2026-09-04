//===- CirScalarTypeConverter.h - CIR scalar → MLIR builtin types *- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Maps CIR scalar types onto the MLIR builtin types that arith ops and the
// func.func signatures/bodies they appear in accept.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRSCALARTYPECONVERTER_H
#define LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRSCALARTYPECONVERTER_H

#include <optional>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

namespace cir {

/// Convert CIR scalar types to the MLIR builtin scalar types that func.func
/// bodies (and the arith ops they inline) accept. Unsupported CIR types
/// convert to a null type, so callers emit a diagnostic instead of silently
/// lowering to the wrong type. Newly supported types are added here as
/// additional conversions rather than an if/else chain.
class CirScalarTypeConverter : public mlir::TypeConverter {
public:
  CirScalarTypeConverter() {
    addConversion([](cir::IntType ty) -> std::optional<mlir::Type> {
      // Project to a signless MLIR integer (the arith convention); signedness
      // is a property of the CIR arithmetic that produced the value, not of
      // the integer type downstream consumes.
      return mlir::IntegerType::get(ty.getContext(), ty.getWidth());
    });
    addConversion([](cir::SingleType ty) -> std::optional<mlir::Type> {
      return mlir::Float32Type::get(ty.getContext());
    });
    addConversion([](cir::DoubleType ty) -> std::optional<mlir::Type> {
      return mlir::Float64Type::get(ty.getContext());
    });
    addConversion([](cir::FP16Type ty) -> std::optional<mlir::Type> {
      return mlir::Float16Type::get(ty.getContext());
    });
    addConversion([](cir::BF16Type ty) -> std::optional<mlir::Type> {
      return mlir::BFloat16Type::get(ty.getContext());
    });
  }
};

} // namespace cir

#endif // LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRSCALARTYPECONVERTER_H
