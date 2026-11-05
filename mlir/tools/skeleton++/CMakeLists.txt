# skeleton++ is a shell script rather than a tool: it only sequences the three
# existing ones (the C++ front end, cir-opt, and the driver), and keeping it as
# a script means the pipeline is readable and can be edited without a rebuild.
# The paths are substituted here so the script works from the build tree.

set(SKELETON_CLANG "${LLVM_BINARY_DIR}/bin/clang++")
set(SKELETON_CIR_OPT "${LLVM_BINARY_DIR}/bin/cir-opt")
set(SKELETON_DRIVER "${LLVM_BINARY_DIR}/bin/mlir-skeleton-driver")
set(SKELETON_RUNTIME_LIB
    "${LLVM_LIBRARY_OUTPUT_INTDIR}/libMLIRSkeletonRuntime.a")
set(SKELETON_MLIR_INCLUDE "${MLIR_MAIN_SRC_DIR}/include")

configure_file(skeleton++.sh.in "${LLVM_RUNTIME_OUTPUT_INTDIR}/skeleton++"
               @ONLY)
file(CHMOD "${LLVM_RUNTIME_OUTPUT_INTDIR}/skeleton++"
     PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ
                 GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

install(PROGRAMS "${LLVM_RUNTIME_OUTPUT_INTDIR}/skeleton++"
        DESTINATION "${CMAKE_INSTALL_BINDIR}")
