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

// Skeleton Target Attribute

def Skeleton_TargetAttr : AttrDef<Skeleton_Dialect, "Target"> {
  let summary = "A target attribute marking the dispatch target of an outlined function";
  let description = [{
    This attribute is attached to outlined func.func ops by the
    skeleton-preference-partition pass to record the dispatch target
    ("CPU" or "GPU") chosen for that function.
  }];
  let mnemonic = "target";
  let parameters = (ins "StringAttr":$value);
  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
}

#endif // SKELETON_ATTRS
