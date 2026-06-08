//===- SkeletonFinalizeMemRefToGpu.cpp - Finalize memref to GPU -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Skeleton/IR/SkeletonDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace mlir {
namespace skeleton {

#define GEN_PASS_DEF_SKELETONFINALIZEMEMREFTOGPU
#include "mlir/Dialect/Skeleton/Transforms/Passes.h.inc"

namespace {

struct SkeletonFinalizeMemRefToGpuPass
    : public impl::SkeletonFinalizeMemRefToGpuBase<
          SkeletonFinalizeMemRefToGpuPass> {
  using SkeletonFinalizeMemRefToGpuBase::SkeletonFinalizeMemRefToGpuBase;

  void runOnOperation() override {
    auto module = dyn_cast<ModuleOp>(getOperation());
    if (!module) {
      signalPassFailure();
      return;
    }

    bool hasGpuTarget = false;
    module.walk([&](func::FuncOp func) {
      if (auto attr = func->getAttrOfType<StringAttr>("skeleton.target"))
        if (attr.getValue() == "GPU")
          hasGpuTarget = true;
    });
    if (!hasGpuTarget)
      return;

    // Convert host-side memref and GPU lifecycle ops to LLVM IR.
    LLVMTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());

    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    populateGpuToLLVMConversionPatterns(typeConverter, patterns);

    LLVMConversionTarget target(getContext());

    // Required by gpu-to-llvm patterns: async type conversion and legality.
    populateAsyncStructuralTypeConversionsAndLegality(
        typeConverter, patterns, target);

    // Legal: func, scf, arith, cf stay for the GPU pipeline.
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<cf::ControlFlowDialect>();

    // Legal: gpu.alloc and gpu.dealloc — sync versions lowered by
    // GPU pipeline (gpu-to-llvm, #191661).
    // Illegal: gpu.memcpy — convert before expand-strided-metadata.
    target.addLegalDialect<gpu::GPUDialect>();
    target.addIllegalOp<gpu::MemcpyOp>();

    // Legal: memref load/store used inside GPU kernels.
    target.addLegalOp<memref::LoadOp, memref::StoreOp>();

    // Illegal: host-side memref lifecycle ops.
    target.addIllegalOp<memref::AllocOp, memref::DeallocOp, memref::CopyOp,
                        memref::CastOp, memref::GetGlobalOp,
                        memref::GlobalOp>();

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createSkeletonFinalizeMemRefToGpuPass() {
  return std::make_unique<SkeletonFinalizeMemRefToGpuPass>();
}

} // namespace skeleton
} // namespace mlir
