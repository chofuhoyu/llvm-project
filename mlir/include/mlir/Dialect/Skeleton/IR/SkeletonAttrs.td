//===- SkeletonAttrs.td - Skeleton dialect attributes -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SKELETON_ATTRS
#define SKELETON_ATTRS

include "SkeletonDialect.td"
include "mlir/IR/AttrTypeBase.td"

// Skeleton Preference Attribute

def Skeleton_PreferenceAttr : AttrDef<Skeleton_Dialect, "Preference"> {
  let summary = "A preference attribute indicating target execution preference";
  let description = [{
    This attribute carries the execution preference (e.g., "GPU", "CPU")
    from the frontend annotation down to the compilation pipeline.
  }];
  let mnemonic = "preference";
  let parameters = (ins "StringAttr":$value);
  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
}

#endif // SKELETON_ATTRS
