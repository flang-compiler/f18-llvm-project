
add_flang_library(FIRTransforms
  AbstractResult.cpp
  AffineDemotion.cpp
  AffinePromotion.cpp
  ArrayValueCopy.cpp
  CharacterConversion.cpp
  ControlFlowConverter.cpp
  CSE.cpp
  FirLoopResultOpt.cpp
  MemDataFlowOpt.cpp
  MemToReg.cpp
  OpenACCDataOperandConversion.cpp
  RewriteLoop.cpp

  DEPENDS
  FIRAnalysis
  FIRDialect
  FIRSupport
  FIROptTransformsPassIncGen
  RewritePatternsIncGen

  LINK_LIBS
  FIRAnalysis
  FIRCodeGen
  FIRDialect
  MLIRAffineToStandard
  MLIROpenACCToLLVMIRTranslation
  FIRSupport
)
