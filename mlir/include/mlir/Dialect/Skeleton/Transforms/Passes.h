//===- Passes.h - Skeleton pass registration header ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SKELETON_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_SKELETON_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace skeleton {

#define GEN_PASS_DECL
#include "mlir/Dialect/Skeleton/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createSkeletonToLinalgPass();
std::unique_ptr<Pass> createSkeletonPreferencePartitionPass();
std::unique_ptr<Pass> createSkeletonTargetLowerPass();
std::unique_ptr<Pass> createSkeletonFinalizeMemRefToGpuPass();
std::unique_ptr<Pass> createSkeletonPatternMergePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Skeleton/Transforms/Passes.h.inc"

} // namespace skeleton
} // namespace mlir

#endif // MLIR_DIALECT_SKELETON_TRANSFORMS_PASSES_H
