get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
  FIRAnalysis
  FIRCodeGen
  FIRDialect
  FIRTransforms
  ${dialect_libs}
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  MLIRStandardToLLVM
  MLIRTransforms
  MLIRAffineToStandard
  MLIRAnalysis
  MLIRSCFToStandard
  MLIREDSC
  MLIRParser
  MLIRStandardToLLVM
  MLIRSupport
  MLIRVectorToLLVM
)

add_flang_unittest(FlangOptimizerTests
  Basic.cpp
)
target_link_libraries(FlangOptimizerTests
  PRIVATE
  ${LIBS})

