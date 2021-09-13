
add_flang_library(FIRTransforms
  AbstractResult.cpp
  AffineDemotion.cpp
  AffinePromotion.cpp
  ArrayValueCopy.cpp
  CharacterConversion.cpp
  ControlFlowConverter.cpp
  CSE.cpp
  ExternalNameConversion.cpp
  FirLoopResultOpt.cpp
  MemDataFlowOpt.cpp
  MemToReg.cpp
  RewriteLoop.cpp

  DEPENDS
  FIRAnalysis
  FIRBuilder
  FIRDialect
  FIRSupport
  FIROptTransformsPassIncGen
  RewritePatternsIncGen

  LINK_LIBS
  FIRAnalysis
  FIRBuilder
  FIRDialect
  MLIRAffineToStandard
  FIRSupport
)
