get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FortranLower
  Bridge.cpp
  CallInterface.cpp
  CharacterExpr.cpp
  CharacterRuntime.cpp
  Coarray.cpp
  ComplexExpr.cpp
  ConvertType.cpp
  DoLoopHelper.cpp
  IntrinsicCall.cpp
  IO.cpp
  Mangler.cpp
  OpenACC.cpp
  OpenMP.cpp
  PFTBuilder.cpp
  SymbolMap.cpp

  DEPENDS
  FIRDialect
  FIRSupport
  FIRTransforms
  ${dialect_libs}

  LINK_LIBS
  FIRDialect
  FIRSupport
  FIRTransforms
  ${dialect_libs}
  FortranCommon
  FortranParser
  FortranEvaluate
  FortranSemantics
  MLIRAffineToStandard
  MLIRLLVMIR
  MLIRSCFToStandard
  MLIRStandard

  LINK_COMPONENTS
  Support
)
