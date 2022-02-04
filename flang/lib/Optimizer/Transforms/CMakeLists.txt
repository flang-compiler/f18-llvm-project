
add_flang_library(FIRTransforms
  AbstractResult.cpp
  AffineDemotion.cpp
  AffinePromotion.cpp
  AnnotateConstant.cpp
  ArrayValueCopy.cpp
  CharacterConversion.cpp
  ControlFlowConverter.cpp
  CSE.cpp
  ExternalNameConversion.cpp
  FirLoopResultOpt.cpp
  MemoryAllocation.cpp
  MemDataFlowOpt.cpp
  MemToReg.cpp
  RewriteLoop.cpp
  SimplifyRegionLite.cpp

  DEPENDS
  FIRAnalysis
  FIRBuilder
  FIRDialect
  FIRSupport
  FIROptTransformsPassIncGen

  LINK_LIBS
  FIRAnalysis
  FIRBuilder
  FIRDialect
  MLIRAffineToStandard
  MLIRLLVMIR
  MLIRTransformUtils
  MLIROpenACC
  MLIROpenMP
  FIRSupport
)
