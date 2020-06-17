//===-- AffinePromotion.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/CommandLine.h"
#define DEBUG_TYPE "flang-affine-promotion"

/// disable FIR to affine dialect conversion
static llvm::cl::opt<bool>
    disableAffinePromo("disable-affine-promotion",
                       llvm::cl::desc("disable FIR to Affine pass"),
                       llvm::cl::init(true));

using namespace fir;

namespace {
class AffineFunctionAnalysis;
class AffineLoopAnalysis;

class AffineLoopAnalysis {
public:
  AffineLoopAnalysis(fir::LoopOp op, AffineFunctionAnalysis &afa)
      : legality(analyzeLoop(op, afa)) {}
  bool canPromoteToAffine() { return legality; }
  Optional<int64_t> step;
  friend AffineFunctionAnalysis;

private:
  bool legality;
  struct MemoryLoadAnalysis {};
  DenseMap<Operation *, MemoryLoadAnalysis> loadAnalysis;
  AffineLoopAnalysis(bool forcedLegality) : legality(forcedLegality) {}
  bool analyzeBody(fir::LoopOp, AffineFunctionAnalysis &);
  bool analyzeLoop(fir::LoopOp loopOperation,
                   AffineFunctionAnalysis &functionAnalysis) {
    LLVM_DEBUG(llvm::dbgs() << "AffinLoopAnalysis: \n"; loopOperation.dump(););
    return analyzeStep(loopOperation.step()) &&
           analyzeMemoryAccess(loopOperation) &&
           analyzeBody(loopOperation, functionAnalysis);
  }
  bool analyzeStep(const mlir::Value stepValue) {
    auto stepDefinition = stepValue.getDefiningOp<ConstantOp>();
    if (stepDefinition) {
      if (auto stepAttr = stepDefinition.getValue().dyn_cast<IntegerAttr>()) {
        step = stepAttr.getInt();
        return true;
      } else {
        LLVM_DEBUG(llvm::dbgs() << "AffineLoopAnalysis: cannot promote loop, "
                                   "step not integer\n";
                   stepAttr.print(llvm::dbgs()););
        return false;
      }
    } else {
      LLVM_DEBUG(
          llvm::dbgs()
              << "AffineLoopAnalysis: cannot promote loop, step not constant\n";
          if (stepValue.getDefiningOp()) {
            stepValue.getDefiningOp()->print(llvm::dbgs());
          });
      return false;
    }
  }
  bool analyzeLoad(fir::LoadOp);
  bool analyzeStore(fir::StoreOp);
  bool analyzeMemoryAccess(fir::LoopOp loopOperation) {
    for (auto loadOp : loopOperation.getOps<fir::LoadOp>()) {
      if (!analyzeLoad(loadOp))
        return false;
    }
    for (auto storeOp : loopOperation.getOps<fir::StoreOp>()) {
      if (!analyzeStore(storeOp))
        return false;
    }
    return true;
  }
};

/// builds analysis for all loop operations within a function
class AffineFunctionAnalysis {
public:
  AffineFunctionAnalysis(mlir::FuncOp funcOp) {
    for (fir::LoopOp op : funcOp.getOps<fir::LoopOp>()) {
      loopAnalysisMap.try_emplace(op, op, *this);
    }
  }
  AffineLoopAnalysis getChildLoopAnalysis(fir::LoopOp op) const {
    auto it = loopAnalysisMap.find_as(op);
    if (it == loopAnalysisMap.end()) {
      LLVM_DEBUG(llvm::dbgs() << "AffineFunctionAnalysis: not computed for:\n";
                 op.dump(););
      op.emitError(
          "error in fetching loop analysis in AffineFunctionAnalysis\n");
      return AffineLoopAnalysis(false);
    } else {
      return it->getSecond();
    }
  }
  friend AffineLoopAnalysis;

private:
  llvm::DenseMap<mlir::Operation *, AffineLoopAnalysis> loopAnalysisMap;
};
bool analyzeCoordinate(mlir::Value coordinate) {
  if (auto blockArg = coordinate.dyn_cast<mlir::BlockArgument>()) {
    if (isa<fir::LoopOp>(blockArg.getOwner()->getParentOp())) {
      return true;
    } else {
      llvm::dbgs() << "AffineLoopAnalysis: array coordinate is not a "
                      "loop induction variable (owner not loopOp)\n";
      return false;
    }
  } else {
    llvm::dbgs() << "AffineLoopAnalysis: array coordinate is not a loop "
                    "induction variable (not a block argument)\n";
    return false;
  }
}
bool AffineLoopAnalysis::analyzeLoad(fir::LoadOp loadOp) {
  bool canPromote = true;
  if (auto acoOp = loadOp.memref().getDefiningOp<ArrayCoorOp>()) {
    for (auto coordinate : acoOp.coor()) {
      canPromote = canPromote && analyzeCoordinate(coordinate);
    }
    if (auto genDim = acoOp.dims().getDefiningOp<GenDimsOp>()) {
    } else {
      LLVM_DEBUG(llvm::dbgs() << "AffineLoopAnalysis: cannot promote loop, "
                                 "dims in ArrayCoorOp not from GenDimsOp\n";);
      canPromote = false;
    }
  } else {
    LLVM_DEBUG(llvm::dbgs() << "AffineLoopAnalysis: cannot promote loop, "
                               "loadOp uses non ArrayCoorOp\n";);
    canPromote = false;
  }
  return canPromote;
}
bool AffineLoopAnalysis::analyzeStore(fir::StoreOp storeOp) { return true; }

bool AffineLoopAnalysis::analyzeBody(fir::LoopOp loopOperation,
                                     AffineFunctionAnalysis &functionAnalysis) {
  for (auto loopOp : loopOperation.getOps<fir::LoopOp>()) {
    auto analysis = functionAnalysis.loopAnalysisMap
                        .try_emplace(loopOp, loopOp, functionAnalysis)
                        .first->getSecond();
    if (!analysis.canPromoteToAffine())
      return false;
  }
  return true;
}

mlir::AffineMap createArrayIndexAffineMap(unsigned dimensions,
                                          MLIRContext *context) {
  auto index = mlir::getAffineConstantExpr(0, context);
  auto extent = mlir::getAffineConstantExpr(1, context);
  for (unsigned i = 0; i < dimensions; ++i) {
    mlir::AffineExpr idx = mlir::getAffineDimExpr(i, context),
                     lowerBound = mlir::getAffineSymbolExpr(i * 3, context),
                     upperBound = mlir::getAffineSymbolExpr(i * 3 + 1, context),
                     stride = mlir::getAffineSymbolExpr(i * 3 + 2, context),
                     currentPart = (idx - lowerBound) * extent;
    index = currentPart + index;
    extent = (upperBound - lowerBound + 1) * stride * extent;
  }
  return mlir::AffineMap::get(dimensions, dimensions * 3, index);
}

/// Convert `fir.loop` to `affine.for`
class AffineLoopConversion : public mlir::OpRewritePattern<fir::LoopOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  AffineLoopConversion(mlir::MLIRContext *context, AffineFunctionAnalysis &afa)
      : OpRewritePattern(context), functionAnalysis(afa) {}

  mlir::LogicalResult
  matchAndRewrite(fir::LoopOp loop,
                  mlir::PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "AffineLoopConversion: rewriting loop:\n";
               loop.dump(););
    auto loopAnalysis = functionAnalysis.getChildLoopAnalysis(loop);
    if (loopAnalysis.step.getValue() <= 0) {
      LLVM_DEBUG(llvm::dbgs()
                     << "AffineLoopAnalysis: cannot promote loop for now, "
                        "step not postive\n";);
      return failure();
    }
    auto &loopOps = loop.getBody()->getOperations();

    auto affineFor = rewriter.create<mlir::AffineForOp>(
        loop.getLoc(), ValueRange(loop.lowerBound()),
        AffineMap::getMultiDimIdentityMap(1, loop.getContext()),
        ValueRange(loop.upperBound()),
        AffineMap::getMultiDimIdentityMap(1, loop.getContext()),
        loopAnalysis.step.getValue());

    rewriter.startRootUpdate(affineFor.getOperation());
    affineFor.getBody()->getOperations().splice(affineFor.getBody()->begin(),
                                                loopOps, loopOps.begin(),
                                                --loopOps.end());
    rewriter.finalizeRootUpdate(affineFor.getOperation());

    for (auto &bodyOp : affineFor.getBody()->getOperations()) {
      if (isa<fir::LoadOp>(bodyOp)) {
        if (failed(rewriteLoad(cast<fir::LoadOp>(bodyOp), rewriter))) {
          return failure();
        }
      }
    }

    rewriter.startRootUpdate(loop.getOperation());
    loop.getInductionVar().replaceAllUsesWith(affineFor.getInductionVar());
    rewriter.finalizeRootUpdate(loop.getOperation());

    rewriter.replaceOp(loop, affineFor.getOperation()->getResults());

    LLVM_DEBUG(llvm::dbgs() << "AffineLoopConversion: loop rewriten to:\n";
               affineFor.dump(););
    return success();
  }

private:
  mlir::LogicalResult rewriteLoad(fir::LoadOp loadOp,
                                  mlir::PatternRewriter &rewriter) const {
    auto acoOp = loadOp.memref().getDefiningOp<ArrayCoorOp>();
    auto genDim = acoOp.dims().getDefiningOp<GenDimsOp>();
    rewriter.setInsertionPoint(loadOp);
    auto affineMap = createArrayIndexAffineMap(acoOp.coor().size(), loadOp.getContext());
    SmallVector<mlir::Value, 4> indexArgs;
    indexArgs.append(acoOp.coor().begin(), acoOp.coor().end());
    indexArgs.append(genDim.triples().begin(), genDim.triples().end());

    auto newIndex = rewriter.create<mlir::AffineApplyOp>(acoOp.getLoc(), affineMap, indexArgs);
    auto arrayElementType = acoOp.ref().getType().dyn_cast<ReferenceType>().getEleTy().dyn_cast<SequenceType>().getEleTy();
    auto newType = mlir::MemRefType::get({-1}, arrayElementType);
    auto newArray = rewriter.create<fir::ConvertOp>(acoOp.getLoc(), newType, acoOp.ref());
    auto newLoad = rewriter.create<mlir::AffineLoadOp>(loadOp.getLoc(), newArray.getResult(), newIndex.getResult());

    rewriter.replaceOp(loadOp, newLoad.getResult());
    return success();
  }
  AffineFunctionAnalysis &functionAnalysis;
};

/// Promote fir.loop and fir.where to affine.for and affine.if, in the cases
/// where such a promotion is possible.
class AffineDialectPromotion
    : public AffineDialectPromotionBase<AffineDialectPromotion> {
public:
  void runOnFunction() override {
    if (disableAffinePromo)
      return;

    auto *context = &getContext();
    auto function = getFunction();
    auto functionAnalysis = AffineFunctionAnalysis(function);
    mlir::OwningRewritePatternList patterns;
    patterns.insert<AffineLoopConversion>(context, functionAnalysis);
    mlir::ConversionTarget target = *context;
    target.addLegalDialect<mlir::AffineDialect, FIROpsDialect,
                           mlir::scf::SCFDialect, mlir::StandardOpsDialect>();
    target.addDynamicallyLegalOp<LoopOp>([&functionAnalysis](fir::LoopOp op) {
      return !(functionAnalysis.getChildLoopAnalysis(op).canPromoteToAffine());
    });
    LLVM_DEBUG(llvm::dbgs()
                   << "AffineDialectPromotion: running promotion on: \n";
               function.print(llvm::dbgs()););
    // apply the patterns
    if (mlir::failed(mlir::applyPartialConversion(function, target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "error in converting to affine dialect\n");
      signalPassFailure();
    }
  }
};
} // namespace

/// Convert FIR loop constructs to the Affine dialect
std::unique_ptr<mlir::Pass> fir::createPromoteToAffinePass() {
  return std::make_unique<AffineDialectPromotion>();
}
