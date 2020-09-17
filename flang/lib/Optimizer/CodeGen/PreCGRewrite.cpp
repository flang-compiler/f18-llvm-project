//===-- PreCGRewrite.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "Target.h"
#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include <memory>

//===----------------------------------------------------------------------===//
// Codegen rewrite: rewriting of subgraphs of ops
//===----------------------------------------------------------------------===//

using namespace fir;

#define DEBUG_TYPE "flang-codegen-rewrite"

static void populateShape(llvm::SmallVectorImpl<mlir::Value> &vec,
                          ShapeOp shape) {
  vec.append(shape.extents().begin(), shape.extents().end());
}

// Operands of fir.shape_shift split into two vectors.
static void populateShapeAndShift(llvm::SmallVectorImpl<mlir::Value> &shapeVec,
                                  llvm::SmallVectorImpl<mlir::Value> &shiftVec,
                                  ShapeShiftOp shift) {
  auto endIter = shift.pairs().end();
  for (auto i = shift.pairs().begin(); i != endIter;) {
    shiftVec.push_back(*i++);
    shapeVec.push_back(*i++);
  }
}

namespace {

/// Convert fir.embox to the extended form where necessary.
class EmboxConversion : public mlir::OpRewritePattern<EmboxOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(EmboxOp embox,
                  mlir::PatternRewriter &rewriter) const override {
    auto shapeVal = embox.getShape();
    // If the embox does not include a shape, then do not convert it
    if (shapeVal)
      return rewriteDynamicShape(embox, rewriter, shapeVal);
    if (auto boxTy = embox.getType().dyn_cast<BoxType>())
      if (auto seqTy = boxTy.getEleTy().dyn_cast<SequenceType>())
        if (seqTy.hasConstantShape())
          return rewriteStaticShape(embox, rewriter, seqTy);
    return mlir::failure();
  }

  mlir::LogicalResult rewriteStaticShape(EmboxOp embox,
                                         mlir::PatternRewriter &rewriter,
                                         SequenceType seqTy) const {
    auto loc = embox.getLoc();
    llvm::SmallVector<mlir::Value, 8> shapeOpers;
    auto idxTy = rewriter.getIndexType();
    for (auto ext : seqTy.getShape()) {
      auto iAttr = rewriter.getIndexAttr(ext);
      auto extVal = rewriter.create<mlir::ConstantOp>(loc, idxTy, iAttr);
      shapeOpers.push_back(extVal);
    }
    mlir::NamedAttrList attrs;
    auto rank = seqTy.getDimension();
    auto rankAttr = rewriter.getIntegerAttr(idxTy, rank);
    attrs.push_back(rewriter.getNamedAttr(XEmboxOp::rankAttrName(), rankAttr));
    auto zeroAttr = rewriter.getIntegerAttr(idxTy, 0);
    attrs.push_back(
        rewriter.getNamedAttr(XEmboxOp::lenParamAttrName(), zeroAttr));
    auto shapeAttr = rewriter.getIntegerAttr(idxTy, shapeOpers.size());
    attrs.push_back(
        rewriter.getNamedAttr(XEmboxOp::shapeAttrName(), shapeAttr));
    attrs.push_back(rewriter.getNamedAttr(XEmboxOp::shiftAttrName(), zeroAttr));
    attrs.push_back(rewriter.getNamedAttr(XEmboxOp::sliceAttrName(), zeroAttr));
    auto xbox = rewriter.create<XEmboxOp>(loc, embox.getType(), embox.memref(),
                                          shapeOpers, llvm::None, llvm::None,
                                          llvm::None, attrs);
    LLVM_DEBUG(llvm::dbgs() << "rewriting " << embox << " to " << xbox << '\n');
    rewriter.replaceOp(embox, xbox.getOperation()->getResults());
    return mlir::success();
  }

  mlir::LogicalResult rewriteDynamicShape(EmboxOp embox,
                                          mlir::PatternRewriter &rewriter,
                                          mlir::Value shapeVal) const {
    auto loc = embox.getLoc();
    auto shapeOp = dyn_cast<ShapeOp>(shapeVal.getDefiningOp());
    llvm::SmallVector<mlir::Value, 8> shapeOpers;
    llvm::SmallVector<mlir::Value, 8> shiftOpers;
    if (shapeOp) {
      populateShape(shapeOpers, shapeOp);
    } else {
      auto shiftOp = dyn_cast<ShapeShiftOp>(shapeVal.getDefiningOp());
      assert(shiftOp && "shape is neither fir.shape nor fir.shape_shift");
      populateShapeAndShift(shapeOpers, shiftOpers, shiftOp);
    }
    mlir::NamedAttrList attrs;
    auto idxTy = rewriter.getIndexType();
    auto rank = shapeOp.getType().cast<ShapeType>().getRank();
    auto rankAttr = rewriter.getIntegerAttr(idxTy, rank);
    attrs.push_back(rewriter.getNamedAttr(XEmboxOp::rankAttrName(), rankAttr));
    auto lenParamSize = embox.lenParams().size();
    auto lenParamAttr = rewriter.getIntegerAttr(idxTy, lenParamSize);
    attrs.push_back(
        rewriter.getNamedAttr(XEmboxOp::lenParamAttrName(), lenParamAttr));
    auto shapeAttr = rewriter.getIntegerAttr(idxTy, shapeOpers.size());
    attrs.push_back(
        rewriter.getNamedAttr(XEmboxOp::shapeAttrName(), shapeAttr));
    auto shiftAttr = rewriter.getIntegerAttr(idxTy, shiftOpers.size());
    attrs.push_back(
        rewriter.getNamedAttr(XEmboxOp::shiftAttrName(), shiftAttr));
    llvm::SmallVector<mlir::Value, 8> sliceOpers;
    if (auto s = embox.getSlice())
      if (auto sliceOp = dyn_cast_or_null<SliceOp>(s.getDefiningOp()))
        sliceOpers.append(sliceOp.triples().begin(), sliceOp.triples().end());
    auto sliceAttr = rewriter.getIntegerAttr(idxTy, sliceOpers.size());
    attrs.push_back(
        rewriter.getNamedAttr(XEmboxOp::sliceAttrName(), sliceAttr));
    auto xbox = rewriter.create<XEmboxOp>(loc, embox.getType(), embox.memref(),
                                          shapeOpers, shiftOpers, sliceOpers,
                                          embox.lenParams(), attrs);
    LLVM_DEBUG(llvm::dbgs() << "rewriting " << embox << " to " << xbox << '\n');
    rewriter.replaceOp(embox, xbox.getOperation()->getResults());
    return mlir::success();
  }
};

/// Convert all fir.array_coor to the extended form.
class ArrayCoorConversion : public mlir::OpRewritePattern<ArrayCoorOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ArrayCoorOp arrCoor,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = arrCoor.getLoc();
    auto shapeVal = arrCoor.shape();
    auto shapeOp = dyn_cast<ShapeOp>(shapeVal.getDefiningOp());
    llvm::SmallVector<mlir::Value, 8> shapeOpers;
    llvm::SmallVector<mlir::Value, 8> shiftOpers;
    if (shapeOp) {
      populateShape(shapeOpers, shapeOp);
    } else {
      auto shiftOp = dyn_cast<ShapeShiftOp>(shapeVal.getDefiningOp());
      if (shiftOp)
        populateShapeAndShift(shapeOpers, shiftOpers, shiftOp);
    }
    mlir::NamedAttrList attrs;
    auto idxTy = rewriter.getIndexType();
    auto rank = shapeOp.getType().cast<ShapeType>().getRank();
    auto rankAttr = rewriter.getIntegerAttr(idxTy, rank);
    attrs.push_back(
        rewriter.getNamedAttr(XArrayCoorOp::rankAttrName(), rankAttr));
    auto lenParamSize = arrCoor.lenParams().size();
    auto lenParamAttr = rewriter.getIntegerAttr(idxTy, lenParamSize);
    attrs.push_back(
        rewriter.getNamedAttr(XArrayCoorOp::lenParamAttrName(), lenParamAttr));
    auto indexSize = arrCoor.indices().size();
    auto idxAttr = rewriter.getIntegerAttr(idxTy, indexSize);
    attrs.push_back(
        rewriter.getNamedAttr(XArrayCoorOp::indexAttrName(), idxAttr));
    auto shapeSize = shapeOp.getNumOperands();
    auto dimAttr = rewriter.getIntegerAttr(idxTy, shapeSize);
    attrs.push_back(
        rewriter.getNamedAttr(XArrayCoorOp::shapeAttrName(), dimAttr));
    llvm::SmallVector<mlir::Value, 8> sliceOpers;
    if (auto s = arrCoor.slice())
      if (auto sliceOp = dyn_cast_or_null<SliceOp>(s.getDefiningOp()))
        sliceOpers.append(sliceOp.triples().begin(), sliceOp.triples().end());
    auto sliceAttr = rewriter.getIntegerAttr(idxTy, sliceOpers.size());
    attrs.push_back(
        rewriter.getNamedAttr(XArrayCoorOp::sliceAttrName(), sliceAttr));
    auto xArrCoor = rewriter.create<XArrayCoorOp>(
        loc, arrCoor.getType(), arrCoor.memref(), shapeOpers, shiftOpers,
        sliceOpers, arrCoor.indices(), arrCoor.lenParams(), attrs);
    LLVM_DEBUG(llvm::dbgs()
               << "rewriting " << arrCoor << " to " << xArrCoor << '\n');
    rewriter.replaceOp(arrCoor, xArrCoor.getOperation()->getResults());
    return mlir::success();
  }
};

/// Convert FIR structured control flow ops to CFG ops.
class CodeGenRewrite : public CodeGenRewriteBase<CodeGenRewrite> {
public:
  void runOnFunction() override final {
    auto &context = getContext();
    mlir::OwningRewritePatternList patterns;
    patterns.insert<EmboxConversion, ArrayCoorConversion>(&context);
    mlir::ConversionTarget target(context);
    target.addLegalDialect<FIROpsDialect, mlir::StandardOpsDialect>();
    target.addIllegalOp<ArrayCoorOp>();
    target.addDynamicallyLegalOp<EmboxOp>([](EmboxOp embox) {
      return !(embox.getShape() ||
               embox.getType().cast<BoxType>().getEleTy().isa<SequenceType>());
    });

    // Do the conversions.
    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(&context),
                      "error in running the pre-codegen conversions");
      signalPassFailure();
    }

    // Erase any residual.
    simplifyRegion(getFunction().getBody());
  }

  // Clean up the region.
  void simplifyRegion(mlir::Region &region) {
    for (auto &block : region.getBlocks())
      for (auto &op : block.getOperations()) {
        if (op.getNumRegions() != 0)
          for (auto &reg : op.getRegions())
            simplifyRegion(reg);
        maybeEraseOp(&op);
      }

    for (auto *op : opsToErase)
      op->erase();
    opsToErase.clear();
  }

  void maybeEraseOp(mlir::Operation *op) {
    if (!op)
      return;

    // Erase any embox that was replaced.
    if (auto embox = dyn_cast<EmboxOp>(op))
      if (embox.getShape()) {
        assert(op->use_empty());
        opsToErase.push_back(op);
      }

    // Erase all fir.array_coor.
    if (isa<ArrayCoorOp>(op)) {
      assert(op->use_empty());
      opsToErase.push_back(op);
    }

    // Erase all fir.shape, fir.shape_shift, and fir.slice ops.
    if (isa<ShapeOp>(op)) {
      assert(op->use_empty());
      opsToErase.push_back(op);
    }
    if (isa<ShapeShiftOp>(op)) {
      assert(op->use_empty());
      opsToErase.push_back(op);
    }
    if (isa<SliceOp>(op)) {
      assert(op->use_empty());
      opsToErase.push_back(op);
    }
  }

private:
  std::vector<mlir::Operation *> opsToErase;
};

} // namespace

/// Convert FIR's structured control flow ops to CFG ops.  This conversion
/// enables the `createLowerToCFGPass` to transform these to CFG form.
std::unique_ptr<mlir::Pass> fir::createFirCodeGenRewritePass() {
  return std::make_unique<CodeGenRewrite>();
}

//===----------------------------------------------------------------------===//
// Target rewrite: reriting of ops to make target-specific lowerings manifest.
//===----------------------------------------------------------------------===//

#undef DEBUG_TYPE
#define DEBUG_TYPE "flang-target-rewrite"

namespace {

/// Fixups for updating a FuncOp's arguments and return values.
struct FixupTy {
  // clang-format off
  enum class Codes {
    ArgumentAsLoad, ArgumentType, CharPair, ReturnAsStore, ReturnType,
    Split, Trailing
  };
  // clang-format on

  FixupTy(Codes code, std::size_t index, std::size_t second = 0)
      : code{code}, index{index}, second{second} {}
  FixupTy(Codes code, std::size_t index,
          std::function<void(mlir::FuncOp)> &&finalizer)
      : code{code}, index{index}, finalizer{finalizer} {}
  FixupTy(Codes code, std::size_t index, std::size_t second,
          std::function<void(mlir::FuncOp)> &&finalizer)
      : code{code}, index{index}, second{second}, finalizer{finalizer} {}

  Codes code;
  std::size_t index;
  std::size_t second{};
  llvm::Optional<std::function<void(mlir::FuncOp)>> finalizer{};
}; // namespace

/// Target-specific rewriting of the IR. This is a prerequisite pass to code
/// generation that traverses the IR and modifies types and operations to a
/// form that appropriate for the specific target. LLVM IR has specific idioms
/// that are used for distinct target processor and ABI combinations.
class TargetRewrite : public TargetRewriteBase<TargetRewrite> {
public:
  TargetRewrite(const TargetRewriteOptions &options) {
    noCharacterConversion = options.noCharacterConversion;
    noComplexConversion = options.noComplexConversion;
  }

  void runOnOperation() override final {
    auto &context = getContext();
    mlir::OpBuilder rewriter(&context);
    auto mod = getModule();
    auto specifics = CodeGenSpecifics::get(getOperation().getContext(),
                                           *getTargetTriple(getOperation()),
                                           *getKindMapping(getOperation()));
    setMembers(specifics.get(), &rewriter);

    // Perform type conversion on signatures and call sites.
    if (mlir::failed(convertTypes(mod))) {
      mlir::emitError(mlir::UnknownLoc::get(&context),
                      "error in converting types to target abi");
      signalPassFailure();
    }

    // Convert ops in target-specific patterns.
    mod.walk([&](mlir::Operation *op) {
      if (auto call = dyn_cast<fir::CallOp>(op)) {
        if (!hasPortableSignature(call.getFunctionType()))
          convertCallOp(call);
      } else if (auto dispatch = dyn_cast<DispatchOp>(op)) {
        if (!hasPortableSignature(dispatch.getFunctionType()))
          convertCallOp(dispatch);
      } else if (auto addr = dyn_cast<AddrOfOp>(op)) {
        if (addr.getType().isa<mlir::FunctionType>() &&
            !hasPortableSignature(addr.getType()))
          convertAddrOp(addr);
      }
    });

    clearMembers();
  }

  mlir::ModuleOp getModule() { return getOperation(); }

  template <typename A, typename B, typename C>
  std::function<mlir::Value(mlir::Operation *)>
  rewriteCallComplexResultType(A ty, B &newResTys, B &newInTys, C &newOpers) {
    auto m = specifics->complexReturnType(ty.getElementType());
    // Currently targets mandate COMPLEX is a single aggregate or packed
    // scalar, included the sret case.
    assert(m.size() == 1 && "target lowering of complex return not supported");
    auto resTy = std::get<mlir::Type>(m[0]);
    auto attr = std::get<CodeGenSpecifics::Attributes>(m[0]);
    auto loc = mlir::UnknownLoc::get(resTy.getContext());
    if (attr.isSRet()) {
      assert(isa_ref_type(resTy));
      mlir::Value stack =
          rewriter->create<fir::AllocaOp>(loc, dyn_cast_ptrEleTy(resTy));
      newInTys.push_back(resTy);
      newOpers.push_back(stack);
      return [=](mlir::Operation *) -> mlir::Value {
        auto memTy = ReferenceType::get(ty);
        auto cast = rewriter->create<ConvertOp>(loc, memTy, stack);
        return rewriter->create<fir::LoadOp>(loc, cast);
      };
    }
    newResTys.push_back(resTy);
    return [=](mlir::Operation *call) -> mlir::Value {
      auto mem = rewriter->create<fir::AllocaOp>(loc, resTy);
      rewriter->create<fir::StoreOp>(loc, call->getResult(0), mem);
      auto memTy = ReferenceType::get(ty);
      auto cast = rewriter->create<ConvertOp>(loc, memTy, mem);
      return rewriter->create<fir::LoadOp>(loc, cast);
    };
  }

  template <typename A, typename B, typename C>
  void rewriteCallComplexInputType(A ty, mlir::Value oper, B &newInTys,
                                   C &newOpers) {
    auto m = specifics->complexArgumentType(ty.getElementType());
    auto *ctx = ty.getContext();
    auto loc = mlir::UnknownLoc::get(ctx);
    if (m.size() == 1) {
      // COMPLEX is a single aggregate
      auto resTy = std::get<mlir::Type>(m[0]);
      auto attr = std::get<CodeGenSpecifics::Attributes>(m[0]);
      auto oldRefTy = ReferenceType::get(ty);
      if (attr.isByVal()) {
        auto mem = rewriter->create<fir::AllocaOp>(loc, ty);
        rewriter->create<fir::StoreOp>(loc, oper, mem);
        newOpers.push_back(rewriter->create<ConvertOp>(loc, resTy, mem));
      } else {
        auto mem = rewriter->create<fir::AllocaOp>(loc, resTy);
        auto cast = rewriter->create<ConvertOp>(loc, oldRefTy, mem);
        rewriter->create<fir::StoreOp>(loc, oper, cast);
        newOpers.push_back(rewriter->create<fir::LoadOp>(loc, mem));
      }
      newInTys.push_back(resTy);
    } else {
      assert(m.size() == 2);
      // COMPLEX is split into 2 separate arguments
      auto iTy = rewriter->getIntegerType(32);
      for (auto e : llvm::enumerate(m)) {
        auto &tup = e.value();
        auto ty = std::get<mlir::Type>(tup);
        auto index = e.index();
        mlir::Value idx = rewriter->create<mlir::ConstantOp>(
            loc, iTy, mlir::IntegerAttr::get(iTy, index));
        auto val = rewriter->create<ExtractValueOp>(loc, ty, oper, idx);
        newInTys.push_back(ty);
        newOpers.push_back(val);
      }
    }
  }

  // Convert fir.call and fir.dispatch Ops.
  template <typename A>
  void convertCallOp(A callOp) {
    auto fnTy = callOp.getFunctionType();
    auto loc = callOp.getLoc();
    rewriter->setInsertionPoint(callOp);
    llvm::SmallVector<mlir::Type, 8> newResTys;
    llvm::SmallVector<mlir::Type, 8> newInTys;
    llvm::SmallVector<mlir::Value, 8> newOpers;
    // FIXME: if the call is indirect, the first argument must still be the
    // function to call.
    llvm::Optional<std::function<mlir::Value(mlir::Operation *)>> wrap;
    if (fnTy.getResults().size() == 1) {
      mlir::Type ty = fnTy.getResult(0);
      llvm::TypeSwitch<mlir::Type>(ty)
          .template Case<fir::ComplexType>([&](fir::ComplexType cmplx) {
            wrap = rewriteCallComplexResultType(cmplx, newResTys, newInTys,
                                                newOpers);
          })
          .template Case<mlir::ComplexType>([&](mlir::ComplexType cmplx) {
            wrap = rewriteCallComplexResultType(cmplx, newResTys, newInTys,
                                                newOpers);
          })
          .Default([&](mlir::Type ty) { newResTys.push_back(ty); });
    } else if (fnTy.getResults().size() > 1) {
      // If the function is returning more than 1 result, do not perform any
      // target-specific lowering. (FIXME?) This may need to be revisited.
      newResTys.insert(newResTys.end(), fnTy.getResults().begin(),
                       fnTy.getResults().end());
    }
    llvm::SmallVector<mlir::Type, 8> trailingInTys;
    llvm::SmallVector<mlir::Value, 8> trailingOpers;
    for (auto e :
         llvm::enumerate(llvm::zip(fnTy.getInputs(), callOp.getOperands()))) {
      mlir::Type ty = std::get<0>(e.value());
      mlir::Value oper = std::get<1>(e.value());
      unsigned index = e.index();
      llvm::TypeSwitch<mlir::Type>(ty)
          .template Case<BoxCharType>([&](BoxCharType boxTy) {
            bool sret;
            if constexpr (std::is_same_v<std::decay_t<A>, fir::CallOp>) {
              sret = callOp.callee() &&
                     functionArgIsSRet(index,
                                       getModule().lookupSymbol<mlir::FuncOp>(
                                           *callOp.callee()));
            } else {
              // TODO: dispatch case; how do we put arguments on a call?
              sret = false;
              llvm_unreachable("not implemented");
            }
            auto m = specifics->boxcharArgumentType(boxTy.getEleTy(), sret);
            auto unbox =
                rewriter->create<UnboxCharOp>(loc, std::get<mlir::Type>(m[0]),
                                              std::get<mlir::Type>(m[1]), oper);
            // unboxed CHARACTER arguments
            for (auto e : llvm::enumerate(m)) {
              unsigned idx = e.index();
              auto attr = std::get<CodeGenSpecifics::Attributes>(e.value());
              auto argTy = std::get<mlir::Type>(e.value());
              if (attr.isAppend()) {
                trailingInTys.push_back(argTy);
                trailingOpers.push_back(unbox.getResult(idx));
              } else {
                newInTys.push_back(argTy);
                newOpers.push_back(unbox.getResult(idx));
              }
            }
          })
          .template Case<fir::ComplexType>([&](fir::ComplexType cmplx) {
            rewriteCallComplexInputType(cmplx, oper, newInTys, newOpers);
          })
          .template Case<mlir::ComplexType>([&](mlir::ComplexType cmplx) {
            rewriteCallComplexInputType(cmplx, oper, newInTys, newOpers);
          })
          .Default([&](mlir::Type ty) {
            newInTys.push_back(ty);
            newOpers.push_back(oper);
          });
    }
    newInTys.insert(newInTys.end(), trailingInTys.begin(), trailingInTys.end());
    newOpers.insert(newOpers.end(), trailingOpers.begin(), trailingOpers.end());
    if constexpr (std::is_same_v<std::decay_t<A>, fir::CallOp>) {
      assert(callOp.callee().hasValue() && "indirect call not implemented");
      auto newCall = rewriter->create<A>(loc, callOp.callee().getValue(),
                                         newResTys, newOpers);
      LLVM_DEBUG(llvm::dbgs() << "replacing call with " << newCall << '\n');
      if (wrap.hasValue())
        replaceOp(callOp, (*wrap)(newCall.getOperation()));
      else
        replaceOp(callOp, newCall.getResults());
    } else {
      // A is fir::DispatchOp
      llvm_unreachable("not implemented"); // TODO
    }
  }

  // Result type fixup for fir::ComplexType and mlir::ComplexType
  template <typename A, typename B>
  void lowerComplexSignatureRes(A cmplx, B &newResTys, B &newInTys) {
    if (noComplexConversion) {
      newResTys.push_back(cmplx);
    } else {
      for (auto &tup : specifics->complexReturnType(cmplx.getElementType())) {
        auto argTy = std::get<mlir::Type>(tup);
        if (std::get<CodeGenSpecifics::Attributes>(tup).isSRet())
          newInTys.push_back(argTy);
        else
          newResTys.push_back(argTy);
      }
    }
  }

  // Argument type fixup for fir::ComplexType and mlir::ComplexType
  template <typename A, typename B>
  void lowerComplexSignatureArg(A cmplx, B &newInTys) {
    if (noComplexConversion)
      newInTys.push_back(cmplx);
    else
      for (auto &tup : specifics->complexArgumentType(cmplx.getElementType()))
        newInTys.push_back(std::get<mlir::Type>(tup));
  }

  /// Taking the address of a function. Modify the signature as needed.
  void convertAddrOp(AddrOfOp addrOp) {
    auto addrTy = addrOp.getType().cast<mlir::FunctionType>();
    llvm::SmallVector<mlir::Type, 8> newResTys;
    llvm::SmallVector<mlir::Type, 8> newInTys;
    for (mlir::Type ty : addrTy.getResults()) {
      llvm::TypeSwitch<mlir::Type>(ty)
          .Case<fir::ComplexType>([&](fir::ComplexType ty) {
            lowerComplexSignatureRes(ty, newResTys, newInTys);
          })
          .Case<mlir::ComplexType>([&](mlir::ComplexType ty) {
            lowerComplexSignatureRes(ty, newResTys, newInTys);
          })
          .Default([&](mlir::Type ty) { newResTys.push_back(ty); });
    }
    llvm::SmallVector<mlir::Type, 8> trailingInTys;
    for (mlir::Type ty : addrTy.getInputs()) {
      llvm::TypeSwitch<mlir::Type>(ty)
          .Case<BoxCharType>([&](BoxCharType box) {
            if (noCharacterConversion) {
              newInTys.push_back(box);
            } else {
              for (auto &tup : specifics->boxcharArgumentType(box.getEleTy())) {
                auto attr = std::get<CodeGenSpecifics::Attributes>(tup);
                auto argTy = std::get<mlir::Type>(tup);
                auto &vec = attr.isAppend() ? trailingInTys : newInTys;
                vec.push_back(argTy);
              }
            }
          })
          .Case<fir::ComplexType>([&](fir::ComplexType ty) {
            lowerComplexSignatureArg(ty, newInTys);
          })
          .Case<mlir::ComplexType>([&](mlir::ComplexType ty) {
            lowerComplexSignatureArg(ty, newInTys);
          })
          .Default([&](mlir::Type ty) { newInTys.push_back(ty); });
    }
    // append trailing input types
    newInTys.insert(newInTys.end(), trailingInTys.begin(), trailingInTys.end());
    // replace this op with a new one with the updated signature
    auto newTy = rewriter->getFunctionType(newInTys, newResTys);
    auto newOp =
        rewriter->create<AddrOfOp>(addrOp.getLoc(), newTy, addrOp.symbol());
    replaceOp(addrOp, newOp.getOperation()->getResults());
  }

  /// Convert the type signatures on all the functions present in the module.
  /// As the type signature is being changed, this must also update the
  /// function itself to use any new arguments, etc.
  mlir::LogicalResult convertTypes(mlir::ModuleOp mod) {
    for (auto fn : mod.getOps<mlir::FuncOp>())
      convertSignature(fn);
    return mlir::success();
  }

  /// If the signature does not need any special target-specific converions,
  /// then it is considered portable for any target, and this function will
  /// return `true`. Otherwise, the signature is not portable and `false` is
  /// returned.
  bool hasPortableSignature(mlir::Type signature) {
    assert(signature.isa<mlir::FunctionType>());
    auto func = signature.dyn_cast<mlir::FunctionType>();
    for (auto ty : func.getResults())
      if ((ty.isa<BoxCharType>() && !noCharacterConversion) ||
          (isa_complex(ty) && !noComplexConversion)) {
        LLVM_DEBUG(llvm::dbgs() << "rewrite " << signature << " for target\n");
        return false;
      }
    for (auto ty : func.getInputs())
      if ((ty.isa<BoxCharType>() && !noCharacterConversion) ||
          (isa_complex(ty) && !noComplexConversion)) {
        LLVM_DEBUG(llvm::dbgs() << "rewrite " << signature << " for target\n");
        return false;
      }
    return true;
  }

  /// Rewrite the signatures and body of the `FuncOp`s in the module for
  /// the immediately subsequent target code gen.
  void convertSignature(mlir::FuncOp func) {
    auto funcTy = func.getType().cast<mlir::FunctionType>();
    if (hasPortableSignature(funcTy))
      return;
    llvm::SmallVector<mlir::Type, 8> newResTys;
    llvm::SmallVector<mlir::Type, 8> newInTys;
    llvm::SmallVector<FixupTy, 8> fixups;

    // Convert return value(s)
    for (auto ty : funcTy.getResults())
      llvm::TypeSwitch<mlir::Type>(ty)
          .Case<fir::ComplexType>([&](fir::ComplexType cmplx) {
            if (noComplexConversion)
              newResTys.push_back(cmplx);
            else
              doComplexReturn(func, cmplx, newResTys, newInTys, fixups);
          })
          .Case<mlir::ComplexType>([&](mlir::ComplexType cmplx) {
            if (noComplexConversion)
              newResTys.push_back(cmplx);
            else
              doComplexReturn(func, cmplx, newResTys, newInTys, fixups);
          })
          .Default([&](mlir::Type ty) { newResTys.push_back(ty); });

    // Convert arguments
    llvm::SmallVector<mlir::Type, 8> trailingTys;
    for (auto e : llvm::enumerate(funcTy.getInputs())) {
      auto ty = e.value();
      unsigned index = e.index();
      llvm::TypeSwitch<mlir::Type>(ty)
          .Case<BoxCharType>([&](BoxCharType boxTy) {
            if (noCharacterConversion) {
              newInTys.push_back(boxTy);
            } else {
              // Convert a CHARACTER argument type. This can involve separating
              // the pointer and the LEN into two arguments and moving the LEN
              // argument to the end of the arg list.
              bool sret = functionArgIsSRet(index, func);
              for (auto e : llvm::enumerate(specifics->boxcharArgumentType(
                       boxTy.getEleTy(), sret))) {
                auto &tup = e.value();
                auto index = e.index();
                auto attr = std::get<CodeGenSpecifics::Attributes>(tup);
                auto argTy = std::get<mlir::Type>(tup);
                if (attr.isAppend()) {
                  trailingTys.push_back(argTy);
                } else {
                  if (sret) {
                    fixups.emplace_back(FixupTy::Codes::CharPair,
                                        newInTys.size(), index);
                  } else {
                    fixups.emplace_back(FixupTy::Codes::Trailing,
                                        newInTys.size(), trailingTys.size());
                  }
                  newInTys.push_back(argTy);
                }
              }
            }
          })
          .Case<fir::ComplexType>([&](fir::ComplexType cmplx) {
            if (noComplexConversion)
              newInTys.push_back(cmplx);
            else
              doComplexArg(func, cmplx, newInTys, fixups);
          })
          .Case<mlir::ComplexType>([&](mlir::ComplexType cmplx) {
            if (noComplexConversion)
              newInTys.push_back(cmplx);
            else
              doComplexArg(func, cmplx, newInTys, fixups);
          })
          .Default([&](mlir::Type ty) { newInTys.push_back(ty); });
    }

    if (!func.empty()) {
      // If the function has a body, then apply the fixups to the arguments and
      // return ops as required. These fixups are done in place.
      auto loc = func.getLoc();
      const auto fixupSize = fixups.size();
      const auto oldArgTys = func.getType().getInputs();
      int offset = 0;
      for (std::remove_const_t<decltype(fixupSize)> i = 0; i < fixupSize; ++i) {
        const auto &fixup = fixups[i];
        switch (fixup.code) {
        case FixupTy::Codes::ArgumentAsLoad: {
          // Argument was pass-by-value, but is now pass-by-reference and
          // possibly with a different element type.
          auto newArg =
              func.front().insertArgument(fixup.index, newInTys[fixup.index]);
          rewriter->setInsertionPointToStart(&func.front());
          auto oldArgTy = ReferenceType::get(oldArgTys[fixup.index - offset]);
          auto cast = rewriter->create<ConvertOp>(loc, oldArgTy, newArg);
          auto load = rewriter->create<fir::LoadOp>(loc, cast);
          func.getArgument(fixup.index + 1).replaceAllUsesWith(load);
          func.front().eraseArgument(fixup.index + 1);
        } break;
        case FixupTy::Codes::ArgumentType: {
          // Argument is pass-by-value, but its type is likely been modified to
          // suit the target ABI convention.
          auto newArg =
              func.front().insertArgument(fixup.index, newInTys[fixup.index]);
          rewriter->setInsertionPointToStart(&func.front());
          auto mem =
              rewriter->create<fir::AllocaOp>(loc, newInTys[fixup.index]);
          rewriter->create<fir::StoreOp>(loc, newArg, mem);
          auto oldArgTy = ReferenceType::get(oldArgTys[fixup.index - offset]);
          auto cast = rewriter->create<ConvertOp>(loc, oldArgTy, mem);
          mlir::Value load = rewriter->create<fir::LoadOp>(loc, cast);
          func.getArgument(fixup.index + 1).replaceAllUsesWith(load);
          func.front().eraseArgument(fixup.index + 1);
          LLVM_DEBUG(llvm::dbgs()
                     << "old argument: " << oldArgTy.getEleTy()
                     << ", repl: " << load << ", new argument: "
                     << func.getArgument(fixup.index).getType() << '\n');
        } break;
        case FixupTy::Codes::CharPair: {
          // The FIR boxchar argument has been split into a pair of distinct
          // arguments that are in juxtaposition to each other.
          auto newArg =
              func.front().insertArgument(fixup.index, newInTys[fixup.index]);
          if (fixup.second == 1) {
            rewriter->setInsertionPointToStart(&func.front());
            auto boxTy = oldArgTys[fixup.index - offset - fixup.second];
            auto box = rewriter->create<EmboxCharOp>(
                loc, boxTy, func.front().getArgument(fixup.index - 1), newArg);
            func.getArgument(fixup.index + 1).replaceAllUsesWith(box);
            func.front().eraseArgument(fixup.index + 1);
            offset++;
          }
        } break;
        case FixupTy::Codes::ReturnAsStore: {
          // The value being returned is now being returned in memory (callee
          // stack space) through a hidden reference argument.
          auto newArg =
              func.front().insertArgument(fixup.index, newInTys[fixup.index]);
          offset++;
          func.walk([&](mlir::ReturnOp ret) {
            rewriter->setInsertionPoint(ret);
            auto oldOper = ret.getOperand(0);
            auto oldOperTy = ReferenceType::get(oldOper.getType());
            auto cast = rewriter->create<ConvertOp>(loc, oldOperTy, newArg);
            rewriter->create<fir::StoreOp>(loc, oldOper, cast);
            rewriter->create<mlir::ReturnOp>(loc);
            ret.erase();
          });
        } break;
        case FixupTy::Codes::ReturnType: {
          // The function is still returning a value, but its type has likely
          // changed to suit the target ABI convention.
          func.walk([&](mlir::ReturnOp ret) {
            rewriter->setInsertionPoint(ret);
            auto oldOper = ret.getOperand(0);
            auto oldOperTy = ReferenceType::get(oldOper.getType());
            auto mem =
                rewriter->create<fir::AllocaOp>(loc, newResTys[fixup.index]);
            auto cast = rewriter->create<ConvertOp>(loc, oldOperTy, mem);
            rewriter->create<fir::StoreOp>(loc, oldOper, cast);
            mlir::Value load = rewriter->create<fir::LoadOp>(loc, mem);
            rewriter->create<mlir::ReturnOp>(loc, load);
            ret.erase();
          });
        } break;
        case FixupTy::Codes::Split: {
          // The FIR argument has been split into a pair of distinct arguments
          // that are in juxtaposition to each other. (For COMPLEX value.)
          auto newArg =
              func.front().insertArgument(fixup.index, newInTys[fixup.index]);
          if (fixup.second == 1) {
            rewriter->setInsertionPointToStart(&func.front());
            auto cplxTy = oldArgTys[fixup.index - offset - fixup.second];
            auto undef = rewriter->create<UndefOp>(loc, cplxTy);
            auto iTy = rewriter->getIntegerType(32);
            mlir::Value zero = rewriter->create<mlir::ConstantOp>(
                loc, iTy, mlir::IntegerAttr::get(iTy, 0));
            mlir::Value one = rewriter->create<mlir::ConstantOp>(
                loc, iTy, mlir::IntegerAttr::get(iTy, 1));
            auto cplx1 = rewriter->create<InsertValueOp>(
                loc, cplxTy, undef, func.front().getArgument(fixup.index - 1),
                zero);
            auto cplx = rewriter->create<InsertValueOp>(loc, cplxTy, cplx1,
                                                        newArg, one);
            func.getArgument(fixup.index + 1).replaceAllUsesWith(cplx);
            func.front().eraseArgument(fixup.index + 1);
            offset++;
          }
        } break;
        case FixupTy::Codes::Trailing: {
          // The FIR argument has been split into a pair of distinct arguments.
          // The first part of the pair appears in the original argument
          // position. The second part of the pair is appended after all the
          // original arguments. (Boxchar arguments.)
          auto newBufArg =
              func.front().insertArgument(fixup.index, newInTys[fixup.index]);
          auto newLenArg = func.front().addArgument(trailingTys[fixup.second]);
          auto boxTy = oldArgTys[fixup.index - offset];
          rewriter->setInsertionPointToStart(&func.front());
          auto box =
              rewriter->create<EmboxCharOp>(loc, boxTy, newBufArg, newLenArg);
          func.getArgument(fixup.index + 1).replaceAllUsesWith(box);
          func.front().eraseArgument(fixup.index + 1);
        } break;
        }
      }
    }

    // Set the new type and finalize the arguments, etc.
    newInTys.insert(newInTys.end(), trailingTys.begin(), trailingTys.end());
    auto newFuncTy =
        mlir::FunctionType::get(newInTys, newResTys, func.getContext());
    LLVM_DEBUG(llvm::dbgs() << "new func: " << newFuncTy << '\n');
    func.setType(newFuncTy);

    for (auto &fixup : fixups)
      if (fixup.finalizer)
        (*fixup.finalizer)(func);
  }

  inline bool functionArgIsSRet(unsigned index, mlir::FuncOp func) {
    if (auto attr = func.getArgAttrOfType<mlir::BoolAttr>(index, "llvm.sret"))
      return attr.getValue();
    return false;
  }

  /// Convert a complex return value. This can involve converting the return
  /// value to a "hidden" first argument or packing the complex into a wide
  /// GPR.
  template <typename A, typename B, typename C>
  void doComplexReturn(mlir::FuncOp func, A cmplx, B &newResTys, B &newInTys,
                       C &fixups) {
    if (noComplexConversion) {
      newResTys.push_back(cmplx);
      return;
    }
    auto m = specifics->complexReturnType(cmplx.getElementType());
    assert(m.size() == 1);
    auto &tup = m[0];
    auto attr = std::get<CodeGenSpecifics::Attributes>(tup);
    auto argTy = std::get<mlir::Type>(tup);
    if (attr.isSRet()) {
      bool argNo = newInTys.size();
      fixups.emplace_back(
          FixupTy::Codes::ReturnAsStore, argNo, [=](mlir::FuncOp func) {
            func.setArgAttr(argNo, "llvm.sret", rewriter->getBoolAttr(true));
          });
      newInTys.push_back(argTy);
      return;
    }
    fixups.emplace_back(FixupTy::Codes::ReturnType, newResTys.size());
    newResTys.push_back(argTy);
  }

  /// Convert a complex argument value. This can involve storing the value to
  /// a temporary memory location or factoring the value into two distinct
  /// arguments.
  template <typename A, typename B, typename C>
  void doComplexArg(mlir::FuncOp func, A cmplx, B &newInTys, C &fixups) {
    if (noComplexConversion) {
      newInTys.push_back(cmplx);
      return;
    }
    auto m = specifics->complexArgumentType(cmplx.getElementType());
    const auto fixupCode =
        m.size() > 1 ? FixupTy::Codes::Split : FixupTy::Codes::ArgumentType;
    for (auto e : llvm::enumerate(m)) {
      auto &tup = e.value();
      auto index = e.index();
      auto attr = std::get<CodeGenSpecifics::Attributes>(tup);
      auto argTy = std::get<mlir::Type>(tup);
      auto argNo = newInTys.size();
      if (attr.isByVal()) {
        if (auto align = attr.getAlignment())
          fixups.emplace_back(
              FixupTy::Codes::ArgumentAsLoad, argNo, [=](mlir::FuncOp func) {
                func.setArgAttr(argNo, "llvm.byval",
                                rewriter->getBoolAttr(true));
                func.setArgAttr(argNo, "llvm.align",
                                rewriter->getIntegerAttr(
                                    rewriter->getIntegerType(32), align));
              });
        else
          fixups.emplace_back(FixupTy::Codes::ArgumentAsLoad, newInTys.size(),
                              [=](mlir::FuncOp func) {
                                func.setArgAttr(argNo, "llvm.byval",
                                                rewriter->getBoolAttr(true));
                              });
      } else {
        if (auto align = attr.getAlignment())
          fixups.emplace_back(fixupCode, argNo, index, [=](mlir::FuncOp func) {
            func.setArgAttr(
                argNo, "llvm.align",
                rewriter->getIntegerAttr(rewriter->getIntegerType(32), align));
          });
        else
          fixups.emplace_back(fixupCode, argNo, index);
      }
      newInTys.push_back(argTy);
    }
  }

private:
  // Replace `op` and remove it.
  void replaceOp(mlir::Operation *op, mlir::ValueRange newValues) {
    op->replaceAllUsesWith(newValues);
    op->dropAllReferences();
    op->erase();
  }

  inline void setMembers(CodeGenSpecifics *s, mlir::OpBuilder *r) {
    specifics = s;
    rewriter = r;
  }

  inline void clearMembers() { setMembers(nullptr, nullptr); }

  CodeGenSpecifics *specifics{};
  mlir::OpBuilder *rewriter;
}; // namespace
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
fir::createFirTargetRewritePass(const TargetRewriteOptions &options) {
  return std::make_unique<TargetRewrite>(options);
}
