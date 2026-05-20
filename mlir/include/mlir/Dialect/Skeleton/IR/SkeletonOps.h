//===- SkeletonOps.h - Skeleton dialect ops declaration ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SKELETON_IR_SKELETONOPS_H
#define MLIR_DIALECT_SKELETON_IR_SKELETONOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonAttrs.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/Skeleton/IR/SkeletonOps.h.inc"

#endif // MLIR_DIALECT_SKELETON_IR_SKELETONOPS_H
