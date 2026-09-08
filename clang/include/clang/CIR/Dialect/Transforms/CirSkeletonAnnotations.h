//===- CirSkeletonAnnotations.h - skeleton annotation semantics -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Single place that owns the meaning of the skeleton annotations the manual
// path reads off CIR ops: the "skeleton.pure" mark on pure functions, the
// "skeleton.op" operator name on helper declarations, and the "skeleton.region"
// execution preference on hosts. Callers get typed enums back, not raw strings
// and guessed argument positions. Parsing and validation stay here; the magic
// strings never leave this module.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRSKELETONANNOTATIONS_H
#define LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRSKELETONANNOTATIONS_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace cir {

/// The operator a "skeleton.op" declaration stands for.
enum class SkeletonOpType { Map, Reduce };

/// The execution preference a "skeleton.region" host carries.
enum class SkeletonPreference { CPU, GPU };

/// The "CPU"/"GPU" string the Skeleton dialect's PreferenceAttr expects.
llvm::StringRef toString(SkeletonPreference preference);

/// True when `op` carries the "skeleton.pure" annotation.
bool hasSkeletonPureAnnotation(mlir::Operation *op);

/// True when `op` carries the "skeleton.op" annotation. An op without it is
/// not a skeleton operator declaration; call this before getSkeletonOpType.
bool hasSkeletonOpAnnotation(mlir::Operation *op);

/// Read the operator named by a "skeleton.op" annotation on `op`. Call only
/// when hasSkeletonOpAnnotation is true. An argument that is missing, is not a
/// string, or does not name a supported operator is reported on `op` and the
/// result fails.
mlir::FailureOr<SkeletonOpType> getSkeletonOpType(mlir::Operation *op);

/// Read the preference of a "skeleton.region" annotation on `op`. An absent
/// annotation returns CPU (an unlabeled host defaults to CPU). A present
/// annotation whose argument is missing, is not a string, or is neither "CPU"
/// nor "GPU" is reported on `op` and the result fails.
mlir::FailureOr<SkeletonPreference>
getSkeletonPreference(mlir::Operation *op);

} // namespace cir

#endif // LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRSKELETONANNOTATIONS_H
