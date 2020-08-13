//===- tco.cpp - Tilikum Crossing Opt ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is to be like LLVM's opt program, only for FIR.  Such a program is
// required for roundtrip testing, etc.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/OptPasses.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string> outputFilename("o",
                                           cl::desc("Specify output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<bool> emitFir("emit-fir",
                             cl::desc("Parse and pretty-print the input"),
                             cl::init(false));

static cl::opt<std::string> targetTriple("target",
                                         cl::desc("specify a target triple"),
                                         cl::init("native"));

static void printModuleBody(mlir::ModuleOp mod, raw_ostream &output) {
  for (auto &op : mod.getBody()->without_terminator())
    output << op << '\n';
}

// compile a .fir file
static int compileFIR(const mlir::PassPipelineCLParser &passPipeline) {
  // check that there is a file to load
  ErrorOr<std::unique_ptr<MemoryBuffer>> fileOrErr =
      MemoryBuffer::getFileOrSTDIN(inputFilename);

  if (std::error_code EC = fileOrErr.getError()) {
    errs() << "Could not open file: " << EC.message() << '\n';
    return 1;
  }

  // load the file into a module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  auto context = std::make_unique<mlir::MLIRContext>();
  auto owningRef = mlir::parseSourceFile(sourceMgr, context.get());

  if (!owningRef) {
    errs() << "Error can't load file " << inputFilename << '\n';
    return 2;
  }
  if (mlir::failed(owningRef->verify())) {
    errs() << "Error verifying FIR module\n";
    return 4;
  }

  std::error_code ec;
  ToolOutputFile out(outputFilename, ec, sys::fs::OF_None);

  // run passes
  llvm::Triple triple(fir::determineTargetTriple(targetTriple));
  fir::NameUniquer uniquer;
  fir::KindMapping kindMap{context.get()};
  fir::setTargetTriple(*owningRef, triple);
  fir::setNameUniquer(*owningRef, uniquer);
  fir::setKindMapping(*owningRef, kindMap);
  mlir::PassManager pm{context.get()};
  mlir::applyPassManagerCLOptions(pm);
  if (emitFir) {
    // parse the input and pretty-print it back out
    // -emit-fir intentionally disables all the passes
  } else if (passPipeline.hasAnyOccurrences()) {
    passPipeline.addToPipeline(pm);
  } else {
    // simplify the IR
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(fir::createCSEPass());
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::createCSEPass());

    // convert fir dialect to affine
    pm.addPass(fir::createPromoteToAffinePass());

    // convert control flow to CFG form
    pm.addPass(fir::createFirToCfgPass());
    pm.addPass(fir::createControlFlowLoweringPass());
    pm.addPass(mlir::createLowerToCFGPass());

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(fir::createCSEPass());

    // pm.addPass(fir::createMemToRegPass());
    pm.addPass(fir::createFirCodeGenRewritePass());
    pm.addPass(fir::createFIRToLLVMPass(uniquer));
    pm.addPass(fir::createLLVMDialectToLLVMPass(out.os()));
  }

  // run the pass manager
  if (mlir::succeeded(pm.run(*owningRef))) {
    // passes ran successfully, so keep the output
    if (emitFir || passPipeline.hasAnyOccurrences())
      printModuleBody(*owningRef, out.os());
    out.keep();
    return 0;
  }

  // pass manager failed
  printModuleBody(*owningRef, errs());
  errs() << "\n\nFAILED: " << inputFilename << '\n';
  return 8;
}

int main(int argc, char **argv) {
  fir::registerFIR();
  fir::registerFIRPasses();
  fir::registerOptPasses();

  [[maybe_unused]] InitLLVM y(argc, argv);
  llvm::InitializeAllTargets();
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipe("", "Compiler passes to run");
  cl::ParseCommandLineOptions(argc, argv, "Tilikum Crossing Optimizer\n");
  return compileFIR(passPipe);
}
