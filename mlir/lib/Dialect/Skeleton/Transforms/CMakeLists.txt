add_mlir_dialect_library(MLIRSkeletonTransforms
  SkeletonToLinalg.cpp
  SkeletonPreferencePartition.cpp
  SkeletonPatternMerge.cpp
  SkeletonTargetLower.cpp
  SkeletonFinalizeMemRefToGpu.cpp

  DEPENDS
  MLIRSkeletonPassIncGen
  MLIRSkeletonOpsIncGen
  MLIRSkeletonAttrsIncGen

  LINK_LIBS PUBLIC
  MLIRSkeleton
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRFuncDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRMemRefDialect
  MLIRGPUDialect
  MLIRLLVMDialect
  MLIRGPUToGPURuntimeTransforms
  MLIRMemRefToLLVM
  MLIRLLVMCommonConversion
  MLIRAsyncToLLVM
  MLIRArithDialect
  MLIRSCFDialect
  MLIRControlFlowDialect
  MLIRTransformUtils
  MLIRAnalysis
  MLIRPass
  )
