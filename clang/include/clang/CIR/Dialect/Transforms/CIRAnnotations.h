//===- CIRAnnotations.h - read cir.annotations on ops ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reading helpers for the `annotations` attribute (an array of cir.annotation
// values) that CIR ops such as cir.func and the loop ops carry.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRANNOTATIONS_H
#define LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRANNOTATIONS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "llvm/ADT/StringRef.h"

namespace cir {

/// Find a cir.annotation with the given name on an operation's annotations
/// attribute. Returns an empty attribute when not found.
inline cir::AnnotationAttr getAnnotationByName(mlir::Operation *op,
                                               llvm::StringRef name) {
  auto annotations = op->getAttrOfType<mlir::ArrayAttr>("annotations");
  if (!annotations)
    return {};
  for (mlir::Attribute attr : annotations) {
    auto ann = mlir::dyn_cast<cir::AnnotationAttr>(attr);
    if (ann && ann.getName().getValue() == name)
      return ann;
  }
  return {};
}

/// Return the first cir.annotation on an operation's annotations attribute,
/// or an empty attribute when there is none.
inline cir::AnnotationAttr getFirstAnnotation(mlir::Operation *op) {
  auto annotations = op->getAttrOfType<mlir::ArrayAttr>("annotations");
  if (!annotations || annotations.empty())
    return {};
  return mlir::dyn_cast<cir::AnnotationAttr>(annotations[0]);
}

} // namespace cir

#endif // LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRANNOTATIONS_H
