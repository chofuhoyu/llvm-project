add_mlir_dialect_library(MLIRSkeletonTransforms
  SkeletonToLinalg.cpp
  SkeletonPatternMerge.cpp
  SkeletonTargetLower.cpp
  SkeletonOutline.cpp
  SkeletonLowerDispatch.cpp

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
