set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/CombAddOptimize.td)
mlir_tablegen(CombAddOptimize.h.inc --gen-pass-decls -name ReduceCombAddPass)
add_public_tablegen_target(MLIRCombAddOptimizeIncGen)
