//===- OpenACCDataOperandConversion.cpp -- convert OpenACC data operand -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Conversion/OpenACCToLLVM/ConvertOpenACCToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "flang-openacc-conversion"
#include "../CodeGen/TypeConverter.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

template <typename Op>
class LegalizeDataOpForLLVMTranslation : public OpConversionPattern<Op> {
public:
  explicit LegalizeDataOpForLLVMTranslation(TypeConverter &converter,
                                            MLIRContext *ctx)
      : OpConversionPattern<Op>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(Op op, llvm::ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &builder) const override {
    Location loc = op.getLoc();
    fir::LLVMTypeConverter &converter = *static_cast<fir::LLVMTypeConverter *>(
        OpConversionPattern<Op>::getTypeConverter());
    unsigned numDataOperand = op.getNumDataOperands();

    // Keep the non data operands without modification.
    auto nonDataOperands =
        operands.take_front(operands.size() - numDataOperand);
    llvm::SmallVector<mlir::Value> convertedOperands;
    convertedOperands.append(nonDataOperands.begin(), nonDataOperands.end());

    // Go over the data operand and legalize them for translation.
    for (unsigned idx = 0; idx < numDataOperand; ++idx) {
      mlir::Value originalDataOperand = op.getDataOperand(idx);

      //
      if (auto refTy =
              originalDataOperand.getType().dyn_cast<fir::ReferenceType>()) {
        if (refTy.getEleTy().isa<fir::SequenceType>() ||
            fir::isa_std_type(refTy.getEleTy())) {
          // Basic arrays and scalars are passed as llvm.ptr to the translation.
          // Code to compute the size will be generated during the translation.
          auto convertedType = converter.convertType(refTy);
          auto convertedValue = builder.create<fir::CastOp>(
              loc, convertedType, originalDataOperand);
          convertedOperands.push_back(convertedValue);
        } else if (auto boxTy = refTy.getEleTy().dyn_cast<fir::BoxType>()) {
          // BoxType needs more work to extract the correct information.
          return builder.notifyMatchFailure(
              op, "fir.box type currently not supported");

          // When a BoxType is encountered we need to push the descriptor and
          // the actual data.

          auto convertedType =
              converter.convertType(refTy).cast<mlir::LLVM::LLVMPointerType>();
          auto convertedValue =
              builder
                  .create<fir::CastOp>(loc, convertedType, originalDataOperand)
                  .res();

          // TODO: currently this code is not doing what it should just so dummy
          // code to create and populate the data descriptor with info.
          auto fortranDescriptor =
              builder.create<LLVM::LoadOp>(loc, convertedValue);
          LLVM::LLVMStructType fortranDescrType =
              convertedType.getElementType().dyn_cast<LLVM::LLVMStructType>();
          auto descrElementPtr = builder.create<LLVM::ExtractValueOp>(
              loc, fortranDescrType, fortranDescriptor,
              builder.getI32ArrayAttr(0));

          auto descr =
              DataDescriptor::undef(builder, loc, fortranDescrType.getBody()[0],
                                    fortranDescrType.getBody()[0]);
          descr.setBasePointer(builder, loc, descrElementPtr);
          descr.setPointer(builder, loc, descrElementPtr);
          // Compute size?
          // descr.setSize(builder, loc, descrElementSize);

          convertedOperands.push_back(descr);
        }
      } else {
        // Type not supported.
        return builder.notifyMatchFailure(op, "unsupported type");
      }
    }

    builder.replaceOpWithNewOp<Op>(op, TypeRange(), convertedOperands,
                                   op.getOperation()->getAttrs());

    return success();
  }
};
} // namespace

namespace {
class ConvertOpenACCToLLVMPass
    : public fir::OpenACCFirDataConversionBase<ConvertOpenACCToLLVMPass> {
public:
  mlir::ModuleOp getModule() { return getOperation(); }
  void runOnOperation() override;
};
} // namespace

void ConvertOpenACCToLLVMPass::runOnOperation() {
  auto op = getOperation();
  auto *context = &getContext();

  // Convert to OpenACC operations with LLVM IR dialect
  RewritePatternSet patterns(context);
  fir::LLVMTypeConverter converter{getModule()};

  // patterns.add<LegalizeDataOpForLLVMTranslation<acc::DataOp>>(converter,
  // context);
  patterns.add<LegalizeDataOpForLLVMTranslation<acc::EnterDataOp>>(converter,
                                                                   context);
  patterns.add<LegalizeDataOpForLLVMTranslation<acc::ExitDataOp>>(converter,
                                                                  context);
  // patterns.add<LegalizeDataOpForLLVMTranslation<acc::ParallelOp>>(converter,
  // context);
  // patterns.add<LegalizeDataOpForLLVMTranslation<acc::UpdateOp>>(converter,
  // context);

  ConversionTarget target(*context);
  target.addLegalDialect<fir::FIROpsDialect, LLVM::LLVMDialect>();

  auto allDataOperandsAreConverted = [](ValueRange operands) {
    for (mlir::Value operand : operands) {
      if (!DataDescriptor::isValid(operand) &&
          !operand.getType().isa<LLVM::LLVMPointerType>())
        return false;
    }
    return true;
  };

  // target.addDynamicallyLegalOp<acc::DataOp>(
  //     [allDataOperandsAreConverted](acc::DataOp op) {
  //       return allDataOperandsAreConverted(op.copyOperands()) &&
  //              allDataOperandsAreConverted(op.copyinOperands()) &&
  //              allDataOperandsAreConverted(op.copyinReadonlyOperands()) &&
  //              allDataOperandsAreConverted(op.copyoutOperands()) &&
  //              allDataOperandsAreConverted(op.copyoutZeroOperands()) &&
  //              allDataOperandsAreConverted(op.createOperands()) &&
  //              allDataOperandsAreConverted(op.createZeroOperands()) &&
  //              allDataOperandsAreConverted(op.noCreateOperands()) &&
  //              allDataOperandsAreConverted(op.presentOperands()) &&
  //              allDataOperandsAreConverted(op.deviceptrOperands()) &&
  //              allDataOperandsAreConverted(op.attachOperands());
  //     });

  target.addDynamicallyLegalOp<acc::EnterDataOp>(
      [allDataOperandsAreConverted](acc::EnterDataOp op) {
        return allDataOperandsAreConverted(op.copyinOperands()) &&
               allDataOperandsAreConverted(op.createOperands()) &&
               allDataOperandsAreConverted(op.createZeroOperands()) &&
               allDataOperandsAreConverted(op.attachOperands());
      });

  target.addDynamicallyLegalOp<acc::ExitDataOp>(
      [allDataOperandsAreConverted](acc::ExitDataOp op) {
        return allDataOperandsAreConverted(op.copyoutOperands()) &&
               allDataOperandsAreConverted(op.deleteOperands()) &&
               allDataOperandsAreConverted(op.detachOperands());
      });

  // target.addDynamicallyLegalOp<acc::UpdateOp>(
  //     [allDataOperandsAreConverted](acc::UpdateOp op) {
  //       return allDataOperandsAreConverted(op.hostOperands()) &&
  //              allDataOperandsAreConverted(op.deviceOperands());
  //     });

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> fir::createOpenACCDataOperandConversionPass() {
  return std::make_unique<ConvertOpenACCToLLVMPass>();
}
