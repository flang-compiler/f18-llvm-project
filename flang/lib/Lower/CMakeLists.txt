
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FortranLower
  Allocatable.cpp
  Bridge.cpp
  CallInterface.cpp
  Coarray.cpp
  ComponentPath.cpp
  ConvertExpr.cpp
  ConvertType.cpp
  ConvertVariable.cpp
  DumpEvaluateExpr.cpp
  HostAssociations.cpp
  IntrinsicCall.cpp
  IO.cpp
  IterationSpace.cpp
  Mangler.cpp
  OpenACC.cpp
  OpenMP.cpp
  PFTBuilder.cpp
  Runtime.cpp
  SymbolMap.cpp
  VectorSubscripts.cpp

  DEPENDS
  FIRBuilder
  FIRDialect
  FIRSupport
  FIRTransforms
  ${dialect_libs}
  omp_gen
  acc_gen

  LINK_LIBS
  FIRBuilder
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
