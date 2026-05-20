//===- SkeletonDialect.td - Skeleton dialect definition ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SKELETON_DIALECT
#define SKELETON_DIALECT

include "mlir/IR/DialectBase.td"

def Skeleton_Dialect : Dialect {
  let name = "skeleton";
  let summary = "The Skeleton dialect for skeleton programming";
  let description = [{
    The Skeleton dialect captures frontend annotations (from pragmas or header
    APIs) that describe structured computation patterns. Operations in this
    dialect carry preference attributes and are lowered to Linalg for further
    optimization.
  }];

  let cppNamespace = "::mlir::skeleton";

  let extraClassDeclaration = [{
    void registerAttributes();
    void registerOperations();
  }];

  let useDefaultAttributePrinterParser = 1;
}

#endif // SKELETON_DIALECT
