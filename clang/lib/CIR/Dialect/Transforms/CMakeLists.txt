add_subdirectory(TargetLowering)

add_clang_library(MLIRCIRTransforms
  CallConvLoweringPass.cpp
  CIRCanonicalize.cpp
  CIRSimplify.cpp
  CIRTransformUtils.cpp
  CXXABILowering.cpp
  EHABILowering.cpp
  TargetLowering.cpp
  FlattenCFG.cpp
  HoistAllocas.cpp
  LoweringPrepare.cpp
  GotoSolver.cpp
  IdiomRecognizer.cpp
  LibOpt.cpp
  CirSkeletonSplit.cpp
  CirCallAnalysis.cpp
  CirCallLowering.cpp
  CirSkeletonAnnotations.cpp
  CirFuncToArith.cpp
  CirStripCIRAttrs.cpp

  DEPENDS
  MLIRCIRPassIncGen

  LINK_LIBS PUBLIC
  clangAST
  clangBasic

  LLVMABI
  MLIRABI
  MLIRAnalysis
  MLIRBufferizationDialect
  MLIRLinalgDialect
  MLIRControlFlowDialect
  MLIRDLTIDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRTransformUtils

  MLIRCIR
  MLIRCIRInterfaces
  MLIRCIRTargetLowering
  MLIROpenACCDialect
  MLIROpenMPDialect
  MLIRSkeleton
)
