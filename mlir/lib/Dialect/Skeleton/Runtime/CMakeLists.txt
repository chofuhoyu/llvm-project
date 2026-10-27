add_mlir_library(MLIRSkeletonRuntime
  SkeletonRuntime.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Skeleton/Runtime

  # Linked into generated program artifacts, which do not link MLIR.
  STANDALONE
  EXCLUDE_FROM_LIBMLIR
  )

target_link_libraries(MLIRSkeletonRuntime PUBLIC ${CMAKE_DL_LIBS})
