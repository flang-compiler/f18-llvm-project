//===-- BoxedProcedure.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/LowLevelIntrinsics.h"
#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "flang-procedure-pointer"

using namespace fir;

namespace {
/// Options to the procedure pointer pass.
struct BoxedProcedureOptions {
  // Lower the boxproc abstraction to function pointers and thunks where
  // required.
  bool useThunks = true;
};

/// This type converter rewrites all `!fir.boxproc<Func>` types to `Func` types.
class BoxprocTypeRewriter : public mlir::TypeConverter {
public:
  using mlir::TypeConverter::convertType;

  BoxprocTypeRewriter() {
    addConversion([](mlir::Type ty) { return ty; });
    addConversion([](BoxProcType boxproc) { return boxproc.getEleTy(); });
    addConversion([&](mlir::TupleType tupTy) {
      llvm::SmallVector<mlir::Type> memTys;
      for (auto ty : tupTy.getTypes())
        memTys.push_back(convertType(ty));
      return mlir::TupleType::get(tupTy.getContext(), memTys);
    });
    addArgumentMaterialization(materializeProcedure);
    addSourceMaterialization(materializeProcedure);
    addTargetMaterialization(materializeProcedure);
  }

  static mlir::Value materializeProcedure(mlir::OpBuilder &builder,
                                          BoxProcType type,
                                          mlir::ValueRange inputs,
                                          mlir::Location loc) {
    assert(inputs.size() == 1);
    return builder.create<ConvertOp>(loc, unwrapRefType(type.getEleTy()),
                                     inputs[0]);
  }
};

/// Rewrite all `fir.emboxproc` ops to either `fir.convert` or a thunk as
/// required.
class EmboxprocConversion : public mlir::OpRewritePattern<EmboxProcOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  EmboxprocConversion(mlir::MLIRContext *ctx) : OpRewritePattern(ctx) {}

  mlir::LogicalResult
  matchAndRewrite(EmboxProcOp embox,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Type toTy = embox.getType().cast<BoxProcType>().getEleTy();
    if (embox.host()) {
      // Create the thunk.
      auto module = embox->getParentOfType<mlir::ModuleOp>();
      FirOpBuilder builder(rewriter, getKindMapping(module));
      auto loc = embox.getLoc();
      mlir::Type i8Ty = builder.getI8Type();
      mlir::Type i8Ptr = builder.getRefType(i8Ty);
      mlir::Type buffTy = SequenceType::get({32}, i8Ty);
      auto buffer = builder.create<AllocaOp>(loc, buffTy);
      mlir::Value closure = builder.createConvert(loc, i8Ptr, embox.host());
      mlir::Value tramp = builder.createConvert(loc, i8Ptr, buffer);
      mlir::Value func = builder.createConvert(loc, i8Ptr, embox.func());
      builder.create<fir::CallOp>(
          loc, factory::getLlvmInitTrampoline(builder),
          llvm::ArrayRef<mlir::Value>{tramp, func, closure});
      auto adjustCall = builder.create<fir::CallOp>(
          loc, factory::getLlvmAdjustTrampoline(builder),
          llvm::ArrayRef<mlir::Value>{tramp});
      rewriter.replaceOpWithNewOp<ConvertOp>(embox, toTy,
                                             adjustCall.getResult(0));
    } else {
      // Just forward the function as a pointer.
      rewriter.replaceOpWithNewOp<ConvertOp>(embox, toTy, embox.func());
    }
    return mlir::success();
  }
};

class FuncConversion : public mlir::OpRewritePattern<mlir::FuncOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  FuncConversion(mlir::MLIRContext *ctx, BoxprocTypeRewriter &tc)
      : OpRewritePattern(ctx), typeConverter(tc) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::FuncOp func,
                  mlir::PatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type> inTys;
    llvm::SmallVector<mlir::Type> resTys;
    rewriter.startRootUpdate(func);
    mlir::FunctionType funcTy = func.getType();
    for (auto ty : funcTy.getInputs())
      inTys.push_back(typeConverter.convertType(ty));
    for (auto ty : funcTy.getResults())
      resTys.push_back(typeConverter.convertType(ty));
    setTypeAndArguments(func, rewriter.getFunctionType(inTys, resTys));
    rewriter.finalizeRootUpdate(func);
    return mlir::success();
  }

  // We have to set the type on the FuncOp but we also have to set the types on
  // the block arguments to type check.
  void setTypeAndArguments(mlir::FuncOp func,
                           mlir::FunctionType toFuncTy) const {
    if (!func.empty()) {
      for (auto e : llvm::enumerate(toFuncTy.getInputs())) {
        unsigned i = e.index();
        auto &block = func.front();
        block.insertArgument(i, e.value());
        block.getArgument(i + 1).replaceAllUsesWith(block.getArgument(i));
        block.eraseArgument(i + 1);
      }
    }
    func.setType(toFuncTy);
  }

  BoxprocTypeRewriter &typeConverter;
};

class UndefConversion : public mlir::OpRewritePattern<UndefOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  UndefConversion(mlir::MLIRContext *ctx, BoxprocTypeRewriter &tc)
      : OpRewritePattern(ctx), typeConverter(tc) {}

  mlir::LogicalResult
  matchAndRewrite(UndefOp undef,
                  mlir::PatternRewriter &rewriter) const override {
    auto tupTy = undef.getType().cast<mlir::TupleType>();
    auto newTupTy = typeConverter.convertType(tupTy);
    rewriter.replaceOpWithNewOp<fir::UndefOp>(undef, newTupTy);
    return mlir::success();
  }

  BoxprocTypeRewriter &typeConverter;
};

class InsertValueConversion : public mlir::OpRewritePattern<InsertValueOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  InsertValueConversion(mlir::MLIRContext *ctx, BoxprocTypeRewriter &tc)
      : OpRewritePattern(ctx), typeConverter(tc) {}

  mlir::LogicalResult
  matchAndRewrite(InsertValueOp ins,
                  mlir::PatternRewriter &rewriter) const override {
    auto tupTy = ins.getType().cast<mlir::TupleType>();
    auto newTupTy = typeConverter.convertType(tupTy);
    rewriter.replaceOpWithNewOp<InsertValueOp>(ins, newTupTy, ins.adt(),
                                               ins.val(), ins.coor());
    return mlir::success();
  }

  BoxprocTypeRewriter &typeConverter;
};

class ExtractValueConversion : public mlir::OpRewritePattern<ExtractValueOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  ExtractValueConversion(mlir::MLIRContext *ctx, BoxprocTypeRewriter &tc)
      : OpRewritePattern(ctx), typeConverter(tc) {}

  mlir::LogicalResult
  matchAndRewrite(ExtractValueOp ext,
                  mlir::PatternRewriter &rewriter) const override {
    auto boxTy = ext.getType().cast<BoxProcType>();
    auto newTy = typeConverter.convertType(boxTy);
    rewriter.replaceOpWithNewOp<ExtractValueOp>(ext, newTy, ext.adt(),
                                                ext.coor());
    return mlir::success();
  }

  BoxprocTypeRewriter &typeConverter;
};

/// Rewrite all `fir.box_addr` ops on values of type `!fir.boxproc` or function
/// type to be `fir.convert` ops.
class BoxaddrConversion : public mlir::OpRewritePattern<BoxAddrOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  BoxaddrConversion(mlir::MLIRContext *ctx) : OpRewritePattern(ctx) {}

  mlir::LogicalResult
  matchAndRewrite(BoxAddrOp addr,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ConvertOp>(addr, addr.getType(), addr.val());
    return mlir::success();
  }
};

/// A `boxproc` is an abstraction for a Fortran procedure reference. Typically,
/// Fortran procedures can be referenced directly through a function pointer.
/// However, Fortran has one-level dynamic scoping between a host procedure and
/// its internal procedures. This allows internal procedures to directly access
/// and modify the state of the host procedure's variables.
///
/// There are any number of possible implementations possible.
///
/// The implementation used here is to convert `boxproc` values to function
/// pointers everywhere. If a `boxproc` value includes a frame pointer to the
/// host procedure's data, then a thunk will be created at runtime to capture
/// the frame pointer during execution. In LLVM IR, the frame pointer is
/// designated with the `nest` attribute. The thunk's address will then be used
/// as the call target instead of the original function's address directly.
class BoxedProcedurePass
    : public BoxedProcedurePassBase<BoxedProcedurePass> {
public:
  BoxedProcedurePass() { options = {true}; }
  BoxedProcedurePass(bool useThunks) { options = {useThunks}; }

  inline mlir::ModuleOp getModule() { return getOperation(); }

  inline static bool isBoxProc(mlir::Type ty) { return ty.isa<BoxProcType>(); };

  // Functions returning a CHARACTER may use a tuple type of
  // `tuple<boxproc<() -> ()>, i64>`. These values must be type converted.
  inline static bool isTupledBoxProc(mlir::Type ty) {
    if (auto tupTy = ty.dyn_cast<mlir::TupleType>())
      if (mlir::Type subTy = tupTy.getType(0))
        return subTy.isa<fir::BoxProcType>();
    return false;
  }

  inline static bool usesBoxProc(mlir::Type ty) {
    return isBoxProc(ty) || isTupledBoxProc(ty);
  }

  void runOnOperation() override final {
    if (options.useThunks) {
      auto *context = &getContext();
      mlir::OwningRewritePatternList pattern(context);
      mlir::ConversionTarget target(*context);
      BoxprocTypeRewriter typeConverter;
      target.addLegalDialect<FIROpsDialect, mlir::StandardOpsDialect,
                             mlir::arith::ArithmeticDialect>();
      target.addDynamicallyLegalOp<BoxAddrOp>([&](BoxAddrOp addr) {
        mlir::Type ty = addr.val().getType();
        return !(ty.isa<BoxProcType>() || ty.isa<mlir::FunctionType>());
      });
      target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp func) {
        mlir::FunctionType ty = func.getType();
        return !(llvm::any_of(ty.getInputs(), usesBoxProc) ||
                 llvm::any_of(ty.getResults(), usesBoxProc));
      });
      target.addDynamicallyLegalOp<UndefOp>(
          [&](UndefOp undef) { return !isTupledBoxProc(undef.getType()); });
      target.addDynamicallyLegalOp<InsertValueOp>(
          [&](InsertValueOp ins) { return !isTupledBoxProc(ins.getType()); });
      target.addDynamicallyLegalOp<ExtractValueOp>(
          [&](ExtractValueOp ext) { return !isBoxProc(ext.getType()); });
      target.addIllegalOp<EmboxProcOp>();
      pattern.insert<EmboxprocConversion, BoxaddrConversion>(context);
      pattern.insert<FuncConversion, UndefConversion, InsertValueConversion,
                     ExtractValueConversion>(context, typeConverter);
      if (mlir::failed(mlir::applyPartialConversion(getModule(), target,
                                                    std::move(pattern))))
        signalPassFailure();
    }
    // TODO: any alternative implementation. Note: currently, the default code
    // gen will not be able to handle boxproc and will give an error.
  }

private:
  BoxedProcedureOptions options;
};
} // namespace

std::unique_ptr<mlir::Pass> fir::createBoxedProcedurePass() {
  return std::make_unique<BoxedProcedurePass>();
}

std::unique_ptr<mlir::Pass> fir::createBoxedProcedurePass(bool useThunks) {
  return std::make_unique<BoxedProcedurePass>(useThunks);
}
