add_mlir_dialect(SkeletonOps skeleton)
add_mlir_doc(SkeletonOps SkeletonDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS SkeletonAttrs.td)
mlir_tablegen(SkeletonAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(SkeletonAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRSkeletonAttrsIncGen)
