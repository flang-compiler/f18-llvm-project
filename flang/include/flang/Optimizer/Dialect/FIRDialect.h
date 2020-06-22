//===-- Optimizer/Dialect/FIRDialect.h -- FIR dialect -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPTIMIZER_DIALECT_FIRDIALECT_H
#define OPTIMIZER_DIALECT_FIRDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Dialect/Affine/Passes.h"

namespace fir {

/// FIR dialect
class FIROpsDialect final : public mlir::Dialect {
public:
  explicit FIROpsDialect(mlir::MLIRContext *ctx);
  virtual ~FIROpsDialect();

  static llvm::StringRef getDialectNamespace() { return "fir"; }

  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  void printType(mlir::Type ty, mlir::DialectAsmPrinter &p) const override;

  mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser,
                                 mlir::Type type) const override;
  void printAttribute(mlir::Attribute attr,
                      mlir::DialectAsmPrinter &p) const override;
};

/// Register the dialect with MLIR
inline void registerFIR() {
  // we want to register exactly once
  [[maybe_unused]] static bool init_once = [] {
    mlir::registerDialect<mlir::AffineDialect>();
    mlir::registerDialect<mlir::LLVM::LLVMDialect>();
    mlir::registerDialect<mlir::scf::SCFDialect>();
    mlir::registerDialect<mlir::StandardOpsDialect>();
    mlir::registerDialect<mlir::vector::VectorDialect>();
    mlir::registerDialect<FIROpsDialect>();
    return true;
  }();
}

/// Register the standard passes we use. This comes from registerAllPasses(),
/// but is a smaller set since we aren't using many of the passes found there.
inline void registerGeneralPasses() {
using mlir::Pass;
#define GEN_PASS_REGISTRATION_Canonicalizer
#define GEN_PASS_REGISTRATION_CSE
#define GEN_PASS_REGISTRATION_AffineLoopFusion
#define GEN_PASS_REGISTRATION_LoopInvariantCodeMotion
#define GEN_PASS_REGISTRATION_LoopCoalescing
#define GEN_PASS_REGISTRATION_StripDebugInfo
#define GEN_PASS_REGISTRATION_PrintOpStats
#define GEN_PASS_REGISTRATION_Inliner
#define GEN_PASS_REGISTRATION_SCCP
#define GEN_PASS_REGISTRATION_MemRefDataFlowOpt
#define GEN_PASS_REGISTRATION_SymbolDCE
#define GEN_PASS_REGISTRATION_LocationSnapshot
#define GEN_PASS_REGISTRATION_PipelineDataTransfer
#include "mlir/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION_AffineVectorize
#define GEN_PASS_REGISTRATION_AffineLoopUnroll
#define GEN_PASS_REGISTRATION_AffineLoopUnrollAndJam
#define GEN_PASS_REGISTRATION_SimplifyAffineStructures
#define GEN_PASS_REGISTRATION_AffineLoopInvariantCodeMotion
#define GEN_PASS_REGISTRATION_AffineLoopTiling
#define GEN_PASS_REGISTRATION_AffineDataCopyGeneration
#include "mlir/Dialect/Affine/Passes.h.inc"

#define GEN_PASS_REGISTRATION_ConvertAffineToStandard
#include "mlir/Conversion/Passes.h.inc"
}

inline void registerFIRPasses() { registerGeneralPasses(); }

} // namespace fir

#endif // OPTIMIZER_DIALECT_FIRDIALECT_H
