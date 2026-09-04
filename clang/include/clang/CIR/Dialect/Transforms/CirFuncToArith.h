//===- CirFuncToArith.h - translate CIR function bodies to arith -*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ClangIR keeps its own op set (cir.fadd, cir.add, ...) and has no lowering to
// the arith/func dialects, so nothing upstream turns a CIR function body into
// the arith body a func.func (and, in turn, a linalg region) can accept. This
// module provides that translation: populateArithFuncBody fills a func.func
// with a straight-line arith body computed from a single-block CIR function
// body, resolving the alloca/store/load form that clang's codegen emits.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRFUNCTOARITH_H
#define LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRFUNCTOARITH_H

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class OpBuilder;
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace cir {

class FuncOp;

/// Fill \p targetFunc (a fresh func.func with an empty body) with a
/// translation of \p cirFunc's body.
///
/// \p cirFunc's body must be a single block of straight-line scalar
/// computation. The clang codegen form (parameters stored into allocas,
/// reloaded before use, results routed through a `__retval` alloca) is
/// resolved back into SSA form as it is translated; hand-written SSA bodies
/// translate as-is.
///
/// Only a subset of CIR ops is supported (constants, integer/fp
/// add/sub/mul/div/rem, fp fused multiply-add, unary negation). Anything else
/// — control flow, comparisons, selects, casts, calls, non-scalar allocas,
/// volatile or atomic memory access — is rejected with a diagnostic; the
/// translator never silently drops part of the body.
///
/// On failure \p targetFunc is left in an undefined, partially-populated
/// state; the caller is expected to erase it.
mlir::LogicalResult populateArithFuncBody(cir::FuncOp cirFunc,
                                          mlir::func::FuncOp targetFunc,
                                          mlir::OpBuilder &builder);

} // namespace cir

#endif // LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRFUNCTOARITH_H
