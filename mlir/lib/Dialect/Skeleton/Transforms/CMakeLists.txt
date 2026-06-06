add_mlir_dialect_library(MLIRSkeletonTransforms
  SkeletonToLinalg.cpp
  SkeletonPreferencePartition.cpp
  SkeletonTargetLower.cpp

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
  MLIRTransformUtils
  MLIRAnalysis
  MLIRPass
  )
