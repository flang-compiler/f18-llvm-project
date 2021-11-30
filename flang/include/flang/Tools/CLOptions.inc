//===-- CLOptions.inc -- command line options -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file defines some shared command-line options that can be used when
// debugging the test tools. This file must be included into the tool.

#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"

#define DisableOption(DOName, DOOption, DODescription) \
  static llvm::cl::opt<bool> disable##DOName("disable-" DOOption, \
      llvm::cl::desc("disable " DODescription " pass"), llvm::cl::init(false), \
      llvm::cl::Hidden)

namespace {
// Optimizer Passes
DisableOption(FirCse, "fir-cse", "CSE for FIR dialect");
DisableOption(CfgConversion, "cfg-conversion", "disable FIR to CFG pass");

// CodeGen Passes
#if !defined(FLANG_EXCLUDE_CODEGEN)
DisableOption(CodeGenRewrite, "codegen-rewrite", "rewrite FIR for codegen");
DisableOption(TargetRewrite, "target-rewrite", "rewrite FIR for target");
DisableOption(FirToLlvmIr, "fir-to-llvmir", "FIR to LLVM-IR dialect");
DisableOption(LlvmIrToLlvm, "llvm", "conversion to LLVM");
#endif

// Generic for adding a pass to the pass manager if it is not disabled.
template <typename F>
void addPassConditionally(
    mlir::PassManager &pm, llvm::cl::opt<bool> &disabled, F ctor) {
  if (!disabled)
    pm.addPass(ctor());
}

template <typename OP, typename F>
void addNestedPassConditionally(
    mlir::PassManager &pm, llvm::cl::opt<bool> &disabled, F ctor) {
  if (!disabled)
    pm.addNestedPass<OP>(ctor());
}

} // namespace

namespace fir {

static void defaultFlangInlinerOptPipeline(mlir::OpPassManager &pm) {
  mlir::GreedyRewriteConfig config;
  config.enableRegionSimplification = false;
  pm.addPass(mlir::createCanonicalizerPass(config));
}

inline void addCfgConversionPass(mlir::PassManager &pm) {
  addNestedPassConditionally<mlir::FuncOp>(
      pm, disableCfgConversion, fir::createFirToCfgPass);
}

inline void addCSE(mlir::PassManager &pm) {
  addNestedPassConditionally<mlir::FuncOp>(
      pm, disableFirCse, fir::createCSEPass);
}

/// Create a basic pass pipeline for simplyfing the FIR dialect
///
/// \param pm - MLIR pass manager that will hold the pipeline definition
inline void createBasicFirPassPipeline(mlir::PassManager &pm)
{
  // simplify the IR
  fir::addCSE(pm);
  pm.addNestedPass<mlir::FuncOp>(fir::createArrayValueCopyPass());
  pm.addNestedPass<mlir::FuncOp>(fir::createCharacterConversionPass());
  pm.addPass(mlir::createCanonicalizerPass());
  fir::addCSE(pm);
  // The default inliner pass adds the canonicalizer pass with the default
  // configuration. Create the inliner pass with tco config.
  llvm::StringMap<mlir::OpPassManager> pipelines;
  pm.addPass(
      mlir::createInlinerPass(pipelines, defaultFlangInlinerOptPipeline));
  pm.addPass(mlir::createCSEPass());

  // convert control flow to CFG form
  fir::addCfgConversionPass(pm);
  pm.addNestedPass<mlir::FuncOp>(fir::createControlFlowLoweringPass());
  pm.addPass(mlir::createLowerToCFGPass());

  mlir::GreedyRewriteConfig config;
  config.enableRegionSimplification = false;
  pm.addPass(mlir::createCanonicalizerPass(config));
  pm.addPass(fir::createSimplifyRegionLitePass());
  fir::addCSE(pm);
}


#if !defined(FLANG_EXCLUDE_CODEGEN)
inline void addCodeGenRewritePass(mlir::PassManager &pm) {
  addPassConditionally(
      pm, disableCodeGenRewrite, fir::createFirCodeGenRewritePass);
}

inline void addTargetRewritePass(mlir::PassManager &pm) {
  addPassConditionally(pm, disableTargetRewrite, []() {
    return fir::createFirTargetRewritePass(fir::TargetRewriteOptions{});
  });
}

inline void addFIRToLLVMPass(mlir::PassManager &pm) {
  addPassConditionally(pm, disableFirToLlvmIr, fir::createFIRToLLVMPass);
}

inline void addLLVMDialectToLLVMPass(
    mlir::PassManager &pm, llvm::raw_ostream &output) {
  addPassConditionally(pm, disableLlvmIrToLlvm,
      [&]() { return fir::createLLVMDialectToLLVMPass(output); });
}

inline void addCodeGenPasses(mlir::PassManager &pm)
{
  pm.addNestedPass<mlir::FuncOp>(fir::createAbstractResultOptPass());
  fir::addCodeGenRewritePass(pm);
  fir::addTargetRewritePass(pm);
  fir::addFIRToLLVMPass(pm);
}

#undef FLANG_EXCLUDE_CODEGEN
#endif

} // namespace fir
