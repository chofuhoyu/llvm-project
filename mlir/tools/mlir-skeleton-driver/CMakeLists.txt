set(LLVM_LINK_COMPONENTS
  Core
  Object
  Passes
  Support
  nativecodegen
  native
  )

add_mlir_tool(mlir-skeleton-driver
  mlir-skeleton-driver.cpp

  EXPORT_SYMBOLS
  )
llvm_update_compile_flags(mlir-skeleton-driver)
mlir_target_link_libraries(mlir-skeleton-driver PRIVATE
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRSupport
  MLIRToLLVMIRTranslationRegistration
  MLIRSkeletonTransforms
  )
target_link_libraries(mlir-skeleton-driver PRIVATE
  MLIRExecutionEngine
  MLIRSkeletonRuntime
  )

mlir_check_all_link_libraries(mlir-skeleton-driver)
