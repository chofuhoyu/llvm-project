add_mlir_dialect_library(MLIRSkeletonTransforms
  SkeletonToLinalg.cpp
  SkeletonPreferencePartition.cpp

  DEPENDS
  MLIRSkeletonPassIncGen
  MLIRSkeletonOpsIncGen
  MLIRSkeletonAttrsIncGen

  LINK_LIBS PUBLIC
  MLIRSkeleton
  MLIRLinalgDialect
  MLIRFuncDialect
  MLIRTransformUtils
  MLIRAnalysis
  )
