add_mlir_dialect_library(MLIRSkeleton
  SkeletonDialect.cpp
  SkeletonOps.cpp

  DEPENDS
  MLIRSkeletonOpsIncGen
  MLIRSkeletonAttrsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  )
