//===- OpenMPToLLVM.cpp - conversion from OpenMP to LLVM dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"

#include "../PassDetail.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

using namespace mlir;

namespace {
/// A pattern that converts the region arguments in a single-region OpenMP
/// operation to the LLVM dialect. The body of the region is not modified and is
/// expected to either be processed by the conversion infrastructure or already
/// contain ops compatible with LLVM dialect types.
template <typename OpType>
struct RegionOpConversion : public ConvertOpToLLVMPattern<OpType> {
  using ConvertOpToLLVMPattern<OpType>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(OpType curOp, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.create<OpType>(
        curOp.getLoc(), TypeRange(), adaptor.getOperands(), curOp->getAttrs());
    rewriter.inlineRegionBefore(curOp.region(), newOp.region(),
                                newOp.region().end());
    if (failed(rewriter.convertRegionTypes(&newOp.region(),
                                           *this->getTypeConverter())))
      return failure();

    rewriter.eraseOp(curOp);
    return success();
  }
};

template <typename T>
struct RegionLessOpConversion : public ConvertOpToLLVMPattern<T> {
  using ConvertOpToLLVMPattern<T>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(T curOp, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = nullptr;
    TypeConverter *converter = ConvertToLLVMPattern::getTypeConverter();
    if (curOp.getType())
      resType = converter->convertType(curOp.getType());
    rewriter.replaceOpWithNewOp<T>(curOp,
                                   resType ? TypeRange(resType) : TypeRange(),
                                   adaptor.getOperands(), curOp->getAttrs());
    return success();
  }
};
} // namespace

void mlir::addDynamicallyLegalOperations(ConversionTarget &target,
                                         LLVMTypeConverter &typeConverter) {
  target.addDynamicallyLegalOp<mlir::omp::ParallelOp, mlir::omp::WsLoopOp,
                               mlir::omp::MasterOp>(
      [&](Operation *op) { return typeConverter.isLegal(&op->getRegion(0)); });
  target.addDynamicallyLegalOp<mlir::omp::ThreadprivateOp>(
      [&](Operation *op) {
          return typeConverter.isLegal(op->getOperandTypes());
      });
}

void mlir::populateOpenMPToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                  RewritePatternSet &patterns) {
  patterns.add<RegionOpConversion<omp::MasterOp>,
               RegionOpConversion<omp::ParallelOp>,
               RegionOpConversion<omp::WsLoopOp>,
               RegionLessOpConversion<omp::ThreadprivateOp>>(converter);
}

namespace {
struct ConvertOpenMPToLLVMPass
    : public ConvertOpenMPToLLVMBase<ConvertOpenMPToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertOpenMPToLLVMPass::runOnOperation() {
  auto module = getOperation();

  // Convert to OpenMP operations with LLVM IR dialect
  RewritePatternSet patterns(&getContext());
  LLVMTypeConverter converter(&getContext());
  mlir::arith::populateArithmeticToLLVMConversionPatterns(converter, patterns);
  populateMemRefToLLVMConversionPatterns(converter, patterns);
  populateStdToLLVMConversionPatterns(converter, patterns);
  populateOpenMPToLLVMConversionPatterns(converter, patterns);

  LLVMConversionTarget target(getContext());
  target.addLegalOp<omp::TerminatorOp, omp::TaskyieldOp, omp::FlushOp,
                    omp::BarrierOp, omp::TaskwaitOp>();
  addDynamicallyLegalOperations(target, converter);
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertOpenMPToLLVMPass() {
  return std::make_unique<ConvertOpenMPToLLVMPass>();
}
