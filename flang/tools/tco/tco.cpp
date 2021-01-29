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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/ToolUtilities.h"
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

// list of program return codes
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

static cl::opt<bool>
    splitInputFile("split-input-file",
                   cl::desc("Split the input file into pieces and process each "
                            "chunk independently"),
                   cl::init(false));

static cl::opt<bool>
    verifyDiagnostics("verify-diagnostics",
                      cl::desc("Check that emitted diagnostics match "
                               "expected-* lines on the corresponding line"),
                      cl::init(false));

static void printModuleBody(mlir::ModuleOp mod, raw_ostream &output) {
  for (auto &op : mod.getBody()->without_terminator())
    output << op << '\n';
}

static mlir::LogicalResult
performActions(raw_ostream &os, SourceMgr &sourceMgr,
               const mlir::PassPipelineCLParser &passPipeline, bool emitFir,
               mlir::MLIRContext *context) {
  // Disable multi-threading when parsing the input file. This removes the
  // unnecessary/costly context synchronization when parsing.
  bool wasThreadingEnabled = context->isMultithreadingEnabled();
  context->disableMultithreading();

  // Parse the input file and reset the context threading state.
  mlir::OwningModuleRef module(parseSourceFile(sourceMgr, context));
  context->enableMultithreading(wasThreadingEnabled);
  if (!module)
    return mlir::failure();

  if (mlir::failed(module->verify())) {
    errs() << "Error verifying FIR module\n";
    return mlir::failure();
  }

  // run passes
  llvm::Triple triple(fir::determineTargetTriple(targetTriple));
  fir::NameUniquer uniquer;
  fir::KindMapping kindMap{context};
  fir::setTargetTriple(*module, triple);
  fir::setNameUniquer(*module, uniquer);
  fir::setKindMapping(*module, kindMap);
  mlir::PassManager pm(context, mlir::OpPassManager::Nesting::Implicit);
  pm.enableVerifier(/*verifyPasses=*/true);
  mlir::applyPassManagerCLOptions(pm);
  if (emitFir) {
    // parse the input and pretty-print it back out
    // -emit-fir intentionally disables all the passes
  } else if (passPipeline.hasAnyOccurrences()) {
    passPipeline.addToPipeline(pm, [&](const Twine &msg) {
      mlir::emitError(mlir::UnknownLoc::get(context)) << msg;
      return mlir::failure();
    });
  } else {
    // simplify the IR
    pm.addNestedPass<mlir::FuncOp>(fir::createArrayValueCopyPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::FuncOp>(fir::createCSEPass());
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::createCSEPass());

    // convert control flow to CFG form
    pm.addNestedPass<mlir::FuncOp>(fir::createFirToCfgPass());
    pm.addNestedPass<mlir::FuncOp>(fir::createControlFlowLoweringPass());
    pm.addPass(mlir::createLowerToCFGPass());

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::FuncOp>(fir::createCSEPass());

    // pm.addPass(fir::createMemToRegPass());
    pm.addPass(fir::createFirCodeGenRewritePass());
    pm.addPass(fir::createFirTargetRewritePass());
    pm.addPass(fir::createFIRToLLVMPass(uniquer));
    pm.addPass(fir::createLLVMDialectToLLVMPass(os));
  }

  // run the pass manager
  if (mlir::succeeded(pm.run(*module))) {
    // passes ran successfully, so keep the output
    if (emitFir || passPipeline.hasAnyOccurrences())
      printModuleBody(*module, os);
    return mlir::success();
  }

  return mlir::failure();
}

static mlir::LogicalResult
processBuffer(raw_ostream &os, std::unique_ptr<MemoryBuffer> ownedBuffer,
              const mlir::PassPipelineCLParser &passPipeline, bool emitFir,
              bool verifyDiagnostics) {
  // Tell sourceMgr about this buffer, which is what the parser will pick up.
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), SMLoc());

  mlir::MLIRContext context;
  fir::registerAndLoadDialects(context);
  context.printOpOnDiagnostic(!verifyDiagnostics);

  if (!verifyDiagnostics) {
    mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    return performActions(os, sourceMgr, passPipeline, emitFir, &context);
  }

  mlir::SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr,
                                                            &context);

  // Do any processing requested by command line flags.  We don't care whether
  // these actions succeed or fail, we only care what diagnostics they produce
  // and whether they match our expectations.
  performActions(os, sourceMgr, passPipeline, emitFir, &context);

  // Verify the diagnostic handler to make sure that each of the diagnostics
  // matched.
  return sourceMgrHandler.verify();
}

// compile a .fir file
static mlir::LogicalResult
compileFIR(const mlir::PassPipelineCLParser &passPipeline) {
  // check that there is a file to load
  ErrorOr<std::unique_ptr<MemoryBuffer>> fileOrErr =
      MemoryBuffer::getFileOrSTDIN(inputFilename);

  if (std::error_code EC = fileOrErr.getError()) {
    errs() << "Could not open file: " << EC.message() << '\n';
    return mlir::failure();
  }

  mlir::MLIRContext context;
  fir::registerAndLoadDialects(context);

  std::error_code ec;
  ToolOutputFile out(outputFilename, ec, sys::fs::OF_None);

  if (splitInputFile)
    return mlir::splitAndProcessBuffer(
        std::move(*fileOrErr),
        [&](std::unique_ptr<MemoryBuffer> chunkBuffer, raw_ostream &os) {
          return processBuffer(out.os(), std::move(chunkBuffer), passPipeline,
                               emitFir, verifyDiagnostics);
        },
        out.os());

  if (failed(processBuffer(out.os(), std::move(*fileOrErr), passPipeline,
                           emitFir, verifyDiagnostics))) {
    errs() << "\n\nFAILED: " << inputFilename << '\n';
    return mlir::failure();
  }

  // Keep the output file if the invocation was successful.
  out.keep();
  return mlir::success();
}

int main(int argc, char **argv) {
  fir::registerFIRPasses();
  fir::registerOptPasses();

  [[maybe_unused]] InitLLVM y(argc, argv);
  InitializeAllTargets();
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipe("", "Compiler passes to run");
  cl::ParseCommandLineOptions(argc, argv, "Tilikum Crossing Optimizer\n");
  return mlir::failed(compileFIR(passPipe));
}
