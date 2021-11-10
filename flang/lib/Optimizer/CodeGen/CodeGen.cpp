//===-- CodeGen.cpp -- bridge to lower to LLVM ----------------------------===//
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

#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "CGOps.h"
#include "DescriptorModel.h"
#include "PassDetail.h"
#include "Target.h"
#include "flang/Lower/Todo.h" // remove when TODO's are done
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "flang/Optimizer/Support/TypeCode.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Config/abi-breaking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "flang-codegen"

//===----------------------------------------------------------------------===//
/// \file
///
/// The Tilikum bridge performs the conversion of operations from both the FIR
/// and standard dialects to the LLVM-IR dialect.
///
/// Some FIR operations may be lowered to other dialects, such as standard, but
/// some FIR operations will pass through to the Tilikum bridge.  This may be
/// necessary to preserve the semantics of the Fortran program.
//===----------------------------------------------------------------------===//

using namespace llvm;

using OperandTy = ArrayRef<mlir::Value>;

namespace fir {
/// return true if all `Value`s in `operands` are `ConstantOp`s
bool allConstants(OperandTy operands) {
  for (auto opnd : operands) {
    if (auto *defop = opnd.getDefiningOp())
      if (isa<mlir::LLVM::ConstantOp>(defop) ||
          isa<mlir::arith::ConstantOp>(defop))
        continue;
    return false;
  }
  return true;
}
} // namespace fir

// FIXME: This should really be recovered from the specified target.
static constexpr unsigned defaultAlign = 8;

// fir::LLVMTypeConverter for converting to LLVM IR dialect types.
#include "TypeConverter.h"

// Instantiate static data member of the type converter.
StringMap<mlir::Type> fir::LLVMTypeConverter::identStructCache;

inline mlir::Type getVoidPtrType(mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(context, 8));
}

static mlir::LLVM::ConstantOp
genConstantIndex(mlir::Location loc, mlir::Type ity,
                 mlir::ConversionPatternRewriter &rewriter,
                 std::int64_t offset) {
  auto cattr = rewriter.getI64IntegerAttr(offset);
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, ity, cattr);
}

namespace {
/// FIR conversion pattern template
template <typename FromOp>
class FIROpConversion : public mlir::OpConversionPattern<FromOp> {
public:
  explicit FIROpConversion(mlir::MLIRContext *ctx,
                           fir::LLVMTypeConverter &lowering)
      : mlir::OpConversionPattern<FromOp>(lowering, ctx, 1) {}

protected:
  mlir::Type convertType(mlir::Type ty) const {
    return lowerTy().convertType(ty);
  }
  mlir::Type voidPtrTy() const {
    return getVoidPtrType(&lowerTy().getContext());
  }

  mlir::LLVM::ConstantOp
  genConstantOffset(mlir::Location loc,
                    mlir::ConversionPatternRewriter &rewriter,
                    int offset) const {
    auto ity = lowerTy().offsetType();
    auto cattr = rewriter.getI32IntegerAttr(offset);
    return rewriter.create<mlir::LLVM::ConstantOp>(loc, ity, cattr);
  }

  /// Perform an extension or truncation as needed on an integer value. Lowering
  /// to the specific target may involve some sign-extending or truncation of
  /// values, particularly to fit them from abstract box types to the
  /// appropriate reified structures.
  mlir::Value integerCast(mlir::Location loc,
                          mlir::ConversionPatternRewriter &rewriter,
                          mlir::Type ty, mlir::Value val) const {
    auto valTy = val.getType();
    // If the value was not yet lowered, lower its type so that it can
    // be used in getPrimitiveTypeSizeInBits.
    if (!valTy.isa<mlir::IntegerType>())
      valTy = convertType(valTy);
    auto toSize = mlir::LLVM::getPrimitiveTypeSizeInBits(ty);
    auto fromSize = mlir::LLVM::getPrimitiveTypeSizeInBits(valTy);
    if (toSize < fromSize)
      return rewriter.create<mlir::LLVM::TruncOp>(loc, ty, val);
    if (toSize > fromSize)
      return rewriter.create<mlir::LLVM::SExtOp>(loc, ty, val);
    return val;
  }

  /// Method to construct code sequence to get the rank from a box.
  mlir::Value getRankFromBox(mlir::Location loc, mlir::Value box,
                             mlir::Type resultTy,
                             mlir::ConversionPatternRewriter &rewriter) const {
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c3 = genConstantOffset(loc, rewriter, 3);
    auto pty = mlir::LLVM::LLVMPointerType::get(resultTy);
    auto p = rewriter.create<mlir::LLVM::GEPOp>(loc, pty,
                                                mlir::ValueRange{box, c0, c3});
    return rewriter.create<mlir::LLVM::LoadOp>(loc, resultTy, p);
  }

  /// Method to construct code sequence to get the triple for dimension `dim`
  /// from a box.
  SmallVector<mlir::Value, 3>
  getDimsFromBox(mlir::Location loc, ArrayRef<mlir::Type> retTys,
                 mlir::Value box, mlir::Value dim,
                 mlir::ConversionPatternRewriter &rewriter) const {
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c7 = genConstantOffset(loc, rewriter, 7);
    auto l0 = loadFromOffset(loc, box, c0, c7, dim, 0, retTys[0], rewriter);
    auto l1 = loadFromOffset(loc, box, c0, c7, dim, 1, retTys[1], rewriter);
    auto l2 = loadFromOffset(loc, box, c0, c7, dim, 2, retTys[2], rewriter);
    return {l0.getResult(), l1.getResult(), l2.getResult()};
  }

  mlir::LLVM::LoadOp
  loadFromOffset(mlir::Location loc, mlir::Value a, mlir::LLVM::ConstantOp c0,
                 mlir::LLVM::ConstantOp c7, mlir::Value dim, int off,
                 mlir::Type ty,
                 mlir::ConversionPatternRewriter &rewriter) const {
    auto pty = mlir::LLVM::LLVMPointerType::get(ty);
    auto c = genConstantOffset(loc, rewriter, off);
    auto p = genGEP(loc, pty, rewriter, a, c0, c7, dim, c);
    return rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
  }

  mlir::Value
  loadStrideFromBox(mlir::Location loc, mlir::Value box, unsigned dim,
                    mlir::ConversionPatternRewriter &rewriter) const {
    auto idxTy = lowerTy().indexType();
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c7 = genConstantOffset(loc, rewriter, 7);
    auto dimValue = genConstantIndex(loc, idxTy, rewriter, dim);
    return loadFromOffset(loc, box, c0, c7, dimValue, 2, idxTy, rewriter);
  }

  /// Read base address from a fir.box. Returned address has type ty.
  mlir::Value
  loadBaseAddrFromBox(mlir::Location loc, mlir::Type ty, mlir::Value box,
                      mlir::ConversionPatternRewriter &rewriter) const {
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto pty = mlir::LLVM::LLVMPointerType::get(ty);
    auto p = genGEP(loc, pty, rewriter, box, c0, c0);
    // load the pointer from the buffer
    return rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
  }

  mlir::Value
  loadElementSizeFromBox(mlir::Location loc, mlir::Type ty, mlir::Value box,
                         mlir::ConversionPatternRewriter &rewriter) const {
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c1 = genConstantOffset(loc, rewriter, 1);
    auto pty = mlir::LLVM::LLVMPointerType::get(ty);
    auto p = genGEP(loc, pty, rewriter, box, c0, c1);
    return rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
  }

  // Get the element type given an LLVM type that is of the form
  // [llvm.ptr](llvm.array|llvm.struct)+ and the provided indexes.
  static mlir::Type getBoxEleTy(mlir::Type type,
                                llvm::ArrayRef<unsigned> indexes) {
    if (auto t = type.dyn_cast<mlir::LLVM::LLVMPointerType>())
      type = t.getElementType();
    for (auto i : indexes) {
      if (auto t = type.dyn_cast<mlir::LLVM::LLVMStructType>()) {
        assert(!t.isOpaque() && i < t.getBody().size());
        type = t.getBody()[i];
      } else if (auto t = type.dyn_cast<mlir::LLVM::LLVMArrayType>()) {
        type = t.getElementType();
      } else if (auto t = type.dyn_cast<mlir::VectorType>()) {
        type = t.getElementType();
      } else {
        fir::emitFatalError(mlir::UnknownLoc::get(type.getContext()),
                            "request for invalid box element type");
      }
    }
    return type;
  }

  // Return LLVM type of the base address given the LLVM type
  // of the related descriptor (lowered fir.box type).
  static mlir::Type getBaseAddrTypeFromBox(mlir::Type type) {
    return getBoxEleTy(type, {0});
  }

  template <typename... ARGS>
  mlir::LLVM::GEPOp genGEP(mlir::Location loc, mlir::Type ty,
                           mlir::ConversionPatternRewriter &rewriter,
                           mlir::Value base, ARGS... args) const {
    SmallVector<mlir::Value> cv{args...};
    return rewriter.create<mlir::LLVM::GEPOp>(loc, ty, base, cv);
  }

  fir::LLVMTypeConverter &lowerTy() const {
    return *static_cast<fir::LLVMTypeConverter *>(this->getTypeConverter());
  }
};

namespace {
/// Helper function for generating the LLVM IR that computes the size
/// in bytes for a derived type.
mlir::Value computeDerivedTypeSize(mlir::Location loc, mlir::Type ptrTy,
                                   mlir::Type idxTy,
                                   mlir::ConversionPatternRewriter &rewriter) {
  auto nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, ptrTy);
  mlir::Value one = genConstantIndex(loc, idxTy, rewriter, 1);
  llvm::SmallVector<mlir::Value> args{nullPtr, one};
  auto gep = rewriter.create<mlir::LLVM::GEPOp>(loc, ptrTy, args);
  return rewriter.create<mlir::LLVM::PtrToIntOp>(loc, idxTy, gep);
}
} // namespace

/// FIR conversion pattern template
template <typename FromOp>
class FIROpAndTypeConversion : public FIROpConversion<FromOp> {
public:
  using FIROpConversion<FromOp>::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(FromOp op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Type ty = this->convertType(op.getType());
    return doRewrite(op, ty, operands, rewriter);
  }

  virtual mlir::LogicalResult
  doRewrite(FromOp addr, mlir::Type ty, OperandTy operands,
            mlir::ConversionPatternRewriter &rewriter) const {
    llvm_unreachable("derived class must override");
  }
};
} // namespace

static Block *createBlock(mlir::ConversionPatternRewriter &rewriter,
                          mlir::Block *insertBefore) {
  assert(insertBefore && "expected valid insertion block");
  return rewriter.createBlock(insertBefore->getParent(),
                              mlir::Region::iterator(insertBefore));
}

/// Create an LLVM dialect global
static void createGlobal(mlir::Location loc, mlir::ModuleOp mod, StringRef name,
                         mlir::Type type,
                         mlir::ConversionPatternRewriter &rewriter) {
  if (mod.lookupSymbol<mlir::LLVM::GlobalOp>(name))
    return;
  mlir::OpBuilder modBuilder(mod.getBodyRegion());
  modBuilder.create<mlir::LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                          mlir::LLVM::Linkage::Weak, name,
                                          mlir::Attribute{});
}

namespace {
/// Lower `fir.address_of` operation to `llvm.address_of` operation.
struct AddrOfOpConversion : public FIROpConversion<fir::AddrOfOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::AddrOfOp addr, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ty = convertType(addr.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(
        addr, ty, addr.symbol().getRootReference().getValue());
    return success();
  }
};
} // namespace

/// Lookup the function to compute the memory size of this parametric derived
/// type. The size of the object may depend on the LEN type parameters of the
/// derived type.
static mlir::LLVM::LLVMFuncOp
getDependentTypeMemSizeFn(fir::RecordType recTy, fir::AllocaOp op,
                          mlir::ConversionPatternRewriter &rewriter) {
  auto module = op->getParentOfType<mlir::ModuleOp>();
  std::string name = recTy.getName().str() + "P.mem.size";
  if (auto memSizeFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name))
    return memSizeFunc;
  TODO(op.getLoc(), "did not find allocation function");
}

namespace {
/// convert to LLVM IR dialect `alloca`
struct AllocaOpConversion : public FIROpConversion<fir::AllocaOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::AllocaOp alloc, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = alloc.getLoc();
    auto ity = lowerTy().indexType();
    unsigned i = 0;
    auto size = genConstantIndex(loc, ity, rewriter, 1).getResult();
    auto ty = convertType(alloc.getType());
    auto resultTy = ty;
    if (alloc.hasLenParams()) {
      unsigned end = alloc.numLenParams();
      llvm::SmallVector<mlir::Value> lenParams;
      for (; i < end; ++i)
        lenParams.push_back(operands[i]);
      auto scalarType = fir::unwrapSequenceType(alloc.getInType());
      if (auto chrTy = scalarType.dyn_cast<fir::CharacterType>()) {
        auto rawCharTy = fir::CharacterType::getUnknownLen(chrTy.getContext(),
                                                           chrTy.getFKind());
        ty = mlir::LLVM::LLVMPointerType::get(convertType(rawCharTy));
        assert(end == 1);
        size = integerCast(loc, rewriter, ity, lenParams[0]);
      } else if (auto recTy = scalarType.dyn_cast<fir::RecordType>()) {
        auto memSizeFn = getDependentTypeMemSizeFn(recTy, alloc, rewriter);
        auto attr = rewriter.getNamedAttr("callee",
                                          mlir::SymbolRefAttr::get(memSizeFn));
        auto call = rewriter.create<mlir::LLVM::CallOp>(
            loc, ity, lenParams, llvm::ArrayRef<mlir::NamedAttribute>{attr});
        size = call.getResult(0);
        ty = mlir::LLVM::LLVMPointerType::get(
            mlir::IntegerType::get(alloc.getContext(), 8));
      } else {
        return emitError(loc, "unexpected type ")
               << scalarType << " with type parameters";
      }
    }
    if (alloc.hasShapeOperands()) {
      auto allocEleTy = fir::unwrapRefType(alloc.getType());
      // Scale the size by constant factors encoded in the array type.
      if (auto seqTy = allocEleTy.dyn_cast<fir::SequenceType>()) {
        fir::SequenceType::Extent constSize = 1;
        for (auto extent : seqTy.getShape())
          if (extent != fir::SequenceType::getUnknownExtent())
            constSize *= extent;
        auto constVal =
            genConstantIndex(loc, ity, rewriter, constSize).getResult();
        size = rewriter.create<mlir::LLVM::MulOp>(loc, ity, size, constVal);
      }
      unsigned end = operands.size();
      for (; i < end; ++i)
        size = rewriter.create<mlir::LLVM::MulOp>(
            loc, ity, size, integerCast(loc, rewriter, ity, operands[i]));
    }
    if (ty == resultTy) {
      // Do not emit the bitcast if ty and resultTy are the same.
      rewriter.replaceOpWithNewOp<mlir::LLVM::AllocaOp>(alloc, ty, size,
                                                        alloc->getAttrs());
    } else {
      auto al = rewriter.create<mlir::LLVM::AllocaOp>(loc, ty, size,
                                                      alloc->getAttrs());
      rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(alloc, resultTy, al);
    }
    return success();
  }
};
} // namespace

/// Return the LLVMFuncOp corresponding to the standard malloc call.
static mlir::LLVM::LLVMFuncOp
getMalloc(fir::AllocMemOp op, mlir::ConversionPatternRewriter &rewriter) {
  auto module = op->getParentOfType<mlir::ModuleOp>();
  if (auto mallocFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("malloc"))
    return mallocFunc;
  mlir::OpBuilder moduleBuilder(
      op->getParentOfType<mlir::ModuleOp>().getBodyRegion());
  auto indexType = mlir::IntegerType::get(op.getContext(), 64);
  return moduleBuilder.create<mlir::LLVM::LLVMFuncOp>(
      rewriter.getUnknownLoc(), "malloc",
      mlir::LLVM::LLVMFunctionType::get(getVoidPtrType(op.getContext()),
                                        indexType,
                                        /*isVarArg=*/false));
}

namespace {
/// convert to `call` to the runtime to `malloc` memory
struct AllocMemOpConversion : public FIROpConversion<fir::AllocMemOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::AllocMemOp heap, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ty = convertType(heap.getType());
    auto mallocFunc = getMalloc(heap, rewriter);
    auto loc = heap.getLoc();
    auto ity = lowerTy().indexType();
    if (auto recTy =
            fir::unwrapSequenceType(heap.getType()).dyn_cast<fir::RecordType>())
      if (recTy.getNumLenParams() != 0)
        TODO(loc, "fir.allocmem of derived type with length parameters");
    auto size = genTypeSizeInBytes(loc, ity, rewriter, ty);
    for (auto opnd : operands)
      size = rewriter.create<mlir::LLVM::MulOp>(
          loc, ity, size, integerCast(loc, rewriter, ity, opnd));
    heap->setAttr("callee", mlir::SymbolRefAttr::get(mallocFunc));
    auto malloc = rewriter.create<mlir::LLVM::CallOp>(
        loc, getVoidPtrType(heap.getContext()), size, heap->getAttrs());
    rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(heap, ty,
                                                       malloc.getResult(0));
    return success();
  }

  // Compute the (allocation) size of the allocmem type in bytes.
  mlir::Value genTypeSizeInBytes(mlir::Location loc, mlir::Type idxTy,
                                 mlir::ConversionPatternRewriter &rewriter,
                                 mlir::Type llTy) const {
    // Use the primitive size, if available.
    auto ptrTy = llTy.dyn_cast<mlir::LLVM::LLVMPointerType>();
    if (auto size =
            mlir::LLVM::getPrimitiveTypeSizeInBits(ptrTy.getElementType()))
      return genConstantIndex(loc, idxTy, rewriter, size / 8);

    // Otherwise, generate the GEP trick in LLVM IR to compute the size.
    return computeDerivedTypeSize(loc, ptrTy, idxTy, rewriter);
  }
};
} // namespace

/// obtain the free() function
static mlir::LLVM::LLVMFuncOp
getFree(fir::FreeMemOp op, mlir::ConversionPatternRewriter &rewriter) {
  auto module = op->getParentOfType<mlir::ModuleOp>();
  if (auto freeFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("free"))
    return freeFunc;
  mlir::OpBuilder moduleBuilder(module.getBodyRegion());
  auto voidType = mlir::LLVM::LLVMVoidType::get(op.getContext());
  return moduleBuilder.create<mlir::LLVM::LLVMFuncOp>(
      rewriter.getUnknownLoc(), "free",
      mlir::LLVM::LLVMFunctionType::get(voidType,
                                        getVoidPtrType(op.getContext()),
                                        /*isVarArg=*/false));
}

namespace {
/// lower a freemem instruction into a call to free()
struct FreeMemOpConversion : public FIROpConversion<fir::FreeMemOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::FreeMemOp freemem, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto freeFunc = getFree(freemem, rewriter);
    auto loc = freemem.getLoc();
    auto bitcast = rewriter.create<mlir::LLVM::BitcastOp>(
        freemem.getLoc(), voidPtrTy(), operands[0]);
    freemem->setAttr("callee", mlir::SymbolRefAttr::get(freeFunc));
    rewriter.create<mlir::LLVM::CallOp>(
        loc, mlir::TypeRange{}, mlir::ValueRange{bitcast}, freemem->getAttrs());
    rewriter.eraseOp(freemem);
    return success();
  }
};

/// convert to returning the first element of the box (any flavor)
struct BoxAddrOpConversion : public FIROpConversion<fir::BoxAddrOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxAddrOp boxaddr, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto a = operands[0];
    auto loc = boxaddr.getLoc();
    auto ty = convertType(boxaddr.getType());
    if (auto argty = boxaddr.val().getType().dyn_cast<fir::BoxType>()) {
      rewriter.replaceOp(boxaddr, loadBaseAddrFromBox(loc, ty, a, rewriter));
    } else {
      auto c0attr = rewriter.getI32IntegerAttr(0);
      auto c0 = mlir::ArrayAttr::get(boxaddr.getContext(), c0attr);
      rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(boxaddr, ty, a,
                                                              c0);
    }
    return success();
  }
};

/// convert to an extractvalue for the 2nd part of the boxchar
struct BoxCharLenOpConversion : public FIROpConversion<fir::BoxCharLenOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxCharLenOp boxchar, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto a = operands[0];
    auto ty = convertType(boxchar.getType());
    auto *ctx = boxchar.getContext();
    auto c1 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(1));
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(boxchar, ty, a, c1);
    return success();
  }
};

/// convert to a triple set of GEPs and loads
struct BoxDimsOpConversion : public FIROpConversion<fir::BoxDimsOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxDimsOp boxdims, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    SmallVector<mlir::Type, 3> resultTypes = {
        convertType(boxdims.getResult(0).getType()),
        convertType(boxdims.getResult(1).getType()),
        convertType(boxdims.getResult(2).getType()),
    };
    auto results = getDimsFromBox(boxdims.getLoc(), resultTypes, operands[0],
                                  operands[1], rewriter);
    rewriter.replaceOp(boxdims, results);
    return success();
  }
};

struct BoxEleSizeOpConversion : public FIROpConversion<fir::BoxEleSizeOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxEleSizeOp boxelesz, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto a = operands[0];
    auto loc = boxelesz.getLoc();
    auto ty = convertType(boxelesz.getType());
    rewriter.replaceOp(boxelesz, loadElementSizeFromBox(loc, ty, a, rewriter));
    return success();
  }
};

struct BoxIsAllocOpConversion : public FIROpConversion<fir::BoxIsAllocOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxIsAllocOp boxisalloc, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto a = operands[0];
    auto loc = boxisalloc.getLoc();
    auto ity = lowerTy().offsetType();
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c5 = genConstantOffset(loc, rewriter, 5);
    auto ty = convertType(boxisalloc.getType());
    auto p = genGEP(loc, ty, rewriter, a, c0, c5);
    auto ld = rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
    auto ab = genConstantOffset(loc, rewriter, 2);
    auto bit = rewriter.create<mlir::LLVM::AndOp>(loc, ity, ld, ab);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        boxisalloc, mlir::LLVM::ICmpPredicate::ne, bit, c0);
    return success();
  }
};

struct BoxIsArrayOpConversion : public FIROpConversion<fir::BoxIsArrayOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxIsArrayOp boxisarray, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto a = operands[0];
    auto loc = boxisarray.getLoc();
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c3 = genConstantOffset(loc, rewriter, 3);
    auto ty = convertType(boxisarray.getType());
    auto p = genGEP(loc, ty, rewriter, a, c0, c3);
    auto ld = rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        boxisarray, mlir::LLVM::ICmpPredicate::ne, ld, c0);
    return success();
  }
};

struct BoxIsPtrOpConversion : public FIROpConversion<fir::BoxIsPtrOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxIsPtrOp boxisptr, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto a = operands[0];
    auto loc = boxisptr.getLoc();
    auto ty = convertType(boxisptr.getType());
    auto ity = lowerTy().offsetType();
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c5 = genConstantOffset(loc, rewriter, 5);
    auto p = rewriter.create<mlir::LLVM::GEPOp>(loc, ty,
                                                mlir::ValueRange{a, c0, c5});
    auto ld = rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
    auto ab = genConstantOffset(loc, rewriter, 1);
    auto bit = rewriter.create<mlir::LLVM::AndOp>(loc, ity, ld, ab);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        boxisptr, mlir::LLVM::ICmpPredicate::ne, bit, c0);
    return success();
  }
};

struct BoxProcHostOpConversion : public FIROpConversion<fir::BoxProcHostOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxProcHostOp boxprochost, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto a = operands[0];
    auto ty = convertType(boxprochost.getType());
    auto c1 = mlir::ArrayAttr::get(boxprochost.getContext(),
                                   rewriter.getI32IntegerAttr(1));
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(boxprochost, ty, a,
                                                            c1);
    return success();
  }
};

struct BoxRankOpConversion : public FIROpConversion<fir::BoxRankOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxRankOp boxrank, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto a = operands[0];
    auto loc = boxrank.getLoc();
    auto ty = convertType(boxrank.getType());
    auto result = getRankFromBox(loc, a, ty, rewriter);
    rewriter.replaceOp(boxrank, result);
    return success();
  }
};

struct BoxTypeDescOpConversion : public FIROpConversion<fir::BoxTypeDescOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxTypeDescOp boxtypedesc, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto a = operands[0];
    auto loc = boxtypedesc.getLoc();
    auto ty = convertType(boxtypedesc.getType());
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c4 = genConstantOffset(loc, rewriter, 4);
    auto pty = mlir::LLVM::LLVMPointerType::get(ty);
    auto p = rewriter.create<mlir::LLVM::GEPOp>(loc, pty,
                                                mlir::ValueRange{a, c0, c4});
    auto ld = rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
    auto i8ptr = mlir::LLVM::LLVMPointerType::get(
        mlir::IntegerType::get(boxtypedesc.getContext(), 8));
    rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(boxtypedesc, i8ptr, ld);
    return success();
  }
};

struct StringLitOpConversion : public FIROpConversion<fir::StringLitOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::StringLitOp constop, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ty = convertType(constop.getType());
    auto attr = constop.getValue();
    if (attr.isa<mlir::StringAttr>()) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(constop, ty, attr);
    } else {
      // FIXME: As of llvm head 85bf221f204eafc1142a064f1650ffa9d9e03dad, using
      // a dense elements attr causes llvm ir verification error:
      //   %0 = llvm.mlir.constant(dense<[234, 456]> : vector<2xi16>) :
      //   !llvm.array<2 x i16> creates a vector type constant instead of an
      //   array, later conflicting with its usage. It is likely an MLIR bug
      //   that should be investigated. In the meantime, do a dumb chain of
      //   inserts.
      auto arr = attr.cast<mlir::ArrayAttr>();
      auto charTy = constop.getType().cast<fir::CharacterType>();
      auto bits = lowerTy().characterBitsize(charTy);
      auto intTy = rewriter.getIntegerType(bits);
      auto loc = constop.getLoc();

      mlir::Value cst = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
      for (auto a : llvm::enumerate(arr.getValue())) {
        // convert each character to a precise bitsize
        auto elemAttr = mlir::IntegerAttr::get(
            intTy,
            a.value().cast<mlir::IntegerAttr>().getValue().sextOrTrunc(bits));
        auto elemCst =
            rewriter.create<mlir::LLVM::ConstantOp>(loc, intTy, elemAttr);
        auto index = mlir::ArrayAttr::get(
            constop.getContext(), rewriter.getI32IntegerAttr(a.index()));
        cst = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, cst, elemCst,
                                                         index);
      }
      rewriter.replaceOp(constop, cst);
    }
    return success();
  }
};

/// direct call LLVM function
struct CallOpConversion : public FIROpConversion<fir::CallOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::CallOp call, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    SmallVector<mlir::Type> resultTys;
    for (auto r : call.getResults())
      resultTys.push_back(convertType(r.getType()));
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(call, resultTys, operands,
                                                    call->getAttrs());
    return success();
  }
};

/// Compare complex values
///
/// Per 10.1, the only comparisons available are .EQ. (oeq) and .NE. (une).
///
/// For completeness, all other comparison are done on the real component only.
struct CmpcOpConversion : public FIROpConversion<fir::CmpcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::CmpcOp cmp, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *ctxt = cmp.getContext();
    auto kind = cmp.lhs().getType().cast<fir::ComplexType>().getFKind();
    auto ty = convertType(fir::RealType::get(ctxt, kind));
    auto resTy = convertType(cmp.getType());
    auto loc = cmp.getLoc();
    auto pos0 = mlir::ArrayAttr::get(ctxt, rewriter.getI32IntegerAttr(0));
    SmallVector<mlir::Value, 2> rp{
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, operands[0], pos0),
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, operands[1],
                                                    pos0)};
    auto rcp =
        rewriter.create<mlir::LLVM::FCmpOp>(loc, resTy, rp, cmp->getAttrs());
    auto pos1 = mlir::ArrayAttr::get(ctxt, rewriter.getI32IntegerAttr(1));
    SmallVector<mlir::Value, 2> ip{
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, operands[0], pos1),
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, operands[1],
                                                    pos1)};
    auto icp =
        rewriter.create<mlir::LLVM::FCmpOp>(loc, resTy, ip, cmp->getAttrs());
    SmallVector<mlir::Value, 2> cp{rcp, icp};
    switch (cmp.getPredicate()) {
    case mlir::arith::CmpFPredicate::OEQ: // .EQ.
      rewriter.replaceOpWithNewOp<mlir::LLVM::AndOp>(cmp, resTy, cp);
      break;
    case mlir::arith::CmpFPredicate::UNE: // .NE.
      rewriter.replaceOpWithNewOp<mlir::LLVM::OrOp>(cmp, resTy, cp);
      break;
    default:
      rewriter.replaceOp(cmp, rcp.getResult());
      break;
    }
    return success();
  }
};

struct ConstcOpConversion : public FIROpConversion<fir::ConstcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::ConstcOp conc, OperandTy,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = conc.getLoc();
    auto *ctx = conc.getContext();
    auto ty = convertType(conc.getType());
    auto ct = conc.getType().cast<fir::ComplexType>();
    auto ety = lowerTy().convertComplexPartType(ct.getFKind());
    auto ri = mlir::FloatAttr::get(ety, getValue(conc.getReal()));
    auto rp = rewriter.create<mlir::LLVM::ConstantOp>(loc, ety, ri);
    auto ii = mlir::FloatAttr::get(ety, getValue(conc.getImaginary()));
    auto ip = rewriter.create<mlir::LLVM::ConstantOp>(loc, ety, ii);
    auto c0 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(0));
    auto c1 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(1));
    auto r = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    auto rr = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, r, rp, c0);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(conc, ty, rr, ip,
                                                           c1);
    return success();
  }

  inline APFloat getValue(mlir::Attribute attr) const {
    return attr.cast<fir::RealAttr>().getValue();
  }
};

static mlir::Type getComplexEleTy(mlir::Type complex) {
  if (auto cc = complex.dyn_cast<mlir::ComplexType>())
    return cc.getElementType();
  return complex.cast<fir::ComplexType>().getElementType();
}

/// convert value of from-type to value of to-type
struct ConvertOpConversion : public FIROpConversion<fir::ConvertOp> {
  using FIROpConversion::FIROpConversion;

  static bool isFloatingPointTy(mlir::Type ty) {
    return ty.isa<mlir::FloatType>();
  }

  mlir::LogicalResult
  matchAndRewrite(fir::ConvertOp convert, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto fromTy = convertType(convert.value().getType());
    auto toTy = convertType(convert.res().getType());
    auto &op0 = operands[0];
    if (fromTy == toTy) {
      rewriter.replaceOp(convert, op0);
      return success();
    }
    auto loc = convert.getLoc();
    auto convertFpToFp = [&](mlir::Value val, unsigned fromBits,
                             unsigned toBits, mlir::Type toTy) -> mlir::Value {
      if (fromBits == toBits) {
        // TODO: Converting between two floating-point representations with the
        // same bitwidth is not allowed for now.
        mlir::emitError(loc,
                        "cannot implicitly convert between two floating-point "
                        "representations of the same bitwidth");
        return {};
      }
      if (fromBits > toBits)
        return rewriter.create<mlir::LLVM::FPTruncOp>(loc, toTy, val);
      return rewriter.create<mlir::LLVM::FPExtOp>(loc, toTy, val);
    };
    if (fir::isa_complex(convert.value().getType()) &&
        fir::isa_complex(convert.res().getType())) {
      // Special case: handle the conversion of a complex such that both the
      // real and imaginary parts are converted together.
      auto zero = mlir::ArrayAttr::get(convert.getContext(),
                                       rewriter.getI32IntegerAttr(0));
      auto one = mlir::ArrayAttr::get(convert.getContext(),
                                      rewriter.getI32IntegerAttr(1));
      auto ty = convertType(getComplexEleTy(convert.value().getType()));
      auto rp = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, op0, zero);
      auto ip = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, op0, one);
      auto nt = convertType(getComplexEleTy(convert.res().getType()));
      auto fromBits = mlir::LLVM::getPrimitiveTypeSizeInBits(ty);
      auto toBits = mlir::LLVM::getPrimitiveTypeSizeInBits(nt);
      auto rc = convertFpToFp(rp, fromBits, toBits, nt);
      auto ic = convertFpToFp(ip, fromBits, toBits, nt);
      auto un = rewriter.create<mlir::LLVM::UndefOp>(loc, toTy);
      auto i1 =
          rewriter.create<mlir::LLVM::InsertValueOp>(loc, toTy, un, rc, zero);
      rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(convert, toTy, i1,
                                                             ic, one);
      return mlir::success();
    }
    if (isFloatingPointTy(fromTy)) {
      if (isFloatingPointTy(toTy)) {
        auto fromBits = mlir::LLVM::getPrimitiveTypeSizeInBits(fromTy);
        auto toBits = mlir::LLVM::getPrimitiveTypeSizeInBits(toTy);
        auto v = convertFpToFp(op0, fromBits, toBits, toTy);
        rewriter.replaceOp(convert, v);
        return mlir::success();
      }
      if (toTy.isa<mlir::IntegerType>()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::FPToSIOp>(convert, toTy, op0);
        return mlir::success();
      }
    } else if (fromTy.isa<mlir::IntegerType>()) {
      if (toTy.isa<mlir::IntegerType>()) {
        auto fromBits = mlir::LLVM::getPrimitiveTypeSizeInBits(fromTy);
        auto toBits = mlir::LLVM::getPrimitiveTypeSizeInBits(toTy);
        assert(fromBits != toBits);
        if (fromBits > toBits) {
          rewriter.replaceOpWithNewOp<mlir::LLVM::TruncOp>(convert, toTy, op0);
          return mlir::success();
        }
        rewriter.replaceOpWithNewOp<mlir::LLVM::SExtOp>(convert, toTy, op0);
        return mlir::success();
      }
      if (isFloatingPointTy(toTy)) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::SIToFPOp>(convert, toTy, op0);
        return mlir::success();
      }
      if (toTy.isa<mlir::LLVM::LLVMPointerType>()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(convert, toTy, op0);
        return mlir::success();
      }
    } else if (fromTy.isa<mlir::LLVM::LLVMPointerType>()) {
      if (toTy.isa<mlir::IntegerType>()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::PtrToIntOp>(convert, toTy, op0);
        return mlir::success();
      }
      if (toTy.isa<mlir::LLVM::LLVMPointerType>()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(convert, toTy, op0);
        return mlir::success();
      }
    }
    return emitError(loc) << "cannot convert " << fromTy << " to " << toTy;
  }
};

/// virtual call to a method in a dispatch table
struct DispatchOpConversion : public FIROpConversion<fir::DispatchOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::DispatchOp dispatch, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ty = convertType(dispatch.getFunctionType());
    // get the table, lookup the method, fetch the func-ptr
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(dispatch, ty, operands,
                                                    None);
    TODO(dispatch.getLoc(), "fir.dispatch codegen");
    return success();
  }
};

/// dispatch table for a Fortran derived type
struct DispatchTableOpConversion
    : public FIROpConversion<fir::DispatchTableOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::DispatchTableOp dispTab, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    TODO(dispTab.getLoc(), "fir.dispatch_table codegen");
    return success();
  }
};

/// entry in a dispatch table; binds a method-name to a function
struct DTEntryOpConversion : public FIROpConversion<fir::DTEntryOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::DTEntryOp dtEnt, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    TODO(dtEnt.getLoc(), "fir.dt_entry codegen");
    return success();
  }
};

/// create a CHARACTER box
struct EmboxCharOpConversion : public FIROpConversion<fir::EmboxCharOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::EmboxCharOp emboxChar, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto a = operands[0];
    auto b1 = operands[1];
    auto loc = emboxChar.getLoc();
    auto *ctx = emboxChar.getContext();
    auto ty = convertType(emboxChar.getType());
    auto c0 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(0));
    auto c1 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(1));
    auto un = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    auto lenTy = ty.cast<mlir::LLVM::LLVMStructType>().getBody()[1];
    auto b = integerCast(loc, rewriter, lenTy, b1);
    auto r = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, un, a, c0);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(emboxChar, ty, r, b,
                                                           c1);
    return success();
  }
};

// Common base class for lowering of embox to descriptor creation.
template <typename OP>
struct EmboxCommonConversion : public FIROpConversion<OP> {
  using FIROpConversion<OP>::FIROpConversion;

  /// Generate an alloca of size `size` and cast it to type `toTy`
  mlir::LLVM::AllocaOp
  genAllocaWithType(mlir::Location loc, mlir::Type toTy, unsigned alignment,
                    mlir::ConversionPatternRewriter &rewriter) const {
    auto thisPt = rewriter.saveInsertionPoint();
    auto *thisBlock = rewriter.getInsertionBlock();
    auto op = thisBlock->getParentOp();
    // Order to find the Op in whose entry block the alloca should be inserted.
    // The parent Op if it is an LLVM Function Op.
    // The ancestor OpenMP Op which is outlineable.
    // The ancestor LLVM Function Op.
    if (auto iface =
            thisBlock->getParent()
                ->getParentOfType<mlir::omp::OutlineableOpenMPOpInterface>()) {
      rewriter.setInsertionPointToStart(iface.getAllocaBlock());
    } else {
      auto func = mlir::isa<mlir::LLVM::LLVMFuncOp>(op)
                      ? mlir::cast<mlir::LLVM::LLVMFuncOp>(op)
                      : op->getParentOfType<mlir::LLVM::LLVMFuncOp>();
      rewriter.setInsertionPointToStart(&func.front());
    }
    auto sz = this->genConstantOffset(loc, rewriter, 1);
    auto al = rewriter.create<mlir::LLVM::AllocaOp>(loc, toTy, sz, alignment);
    rewriter.restoreInsertionPoint(thisPt);
    return al;
  }

  int getCFIAttr(fir::BoxType boxTy) const {
    auto eleTy = boxTy.getEleTy();
    if (eleTy.isa<fir::PointerType>())
      return CFI_attribute_pointer;
    if (eleTy.isa<fir::HeapType>())
      return CFI_attribute_allocatable;
    return CFI_attribute_other;
  }

  static fir::RecordType unwrapIfDerived(fir::BoxType boxTy) {
    return fir::unwrapSequenceType(fir::dyn_cast_ptrOrBoxEleTy(boxTy))
        .template dyn_cast<fir::RecordType>();
  }
  static bool isDerivedTypeWithLenParams(fir::BoxType boxTy) {
    auto recTy = unwrapIfDerived(boxTy);
    return recTy && recTy.getNumLenParams() > 0;
  }
  static bool isDerivedType(fir::BoxType boxTy) {
    return static_cast<bool>(unwrapIfDerived(boxTy));
  }

  // Get the element size and CFI type code of the boxed value.
  std::tuple<mlir::Value, mlir::Value> getSizeAndTypeCode(
      mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
      mlir::Type boxEleTy, mlir::ValueRange lenParams = {}) const {
    auto doInteger =
        [&](unsigned width) -> std::tuple<mlir::Value, mlir::Value> {
      int typeCode = fir::integerBitsToTypeCode(width);
      return {this->genConstantOffset(loc, rewriter, width / 8),
              this->genConstantOffset(loc, rewriter, typeCode)};
    };
    auto doLogical =
        [&](unsigned width) -> std::tuple<mlir::Value, mlir::Value> {
      int typeCode = fir::logicalBitsToTypeCode(width);
      return {this->genConstantOffset(loc, rewriter, width / 8),
              this->genConstantOffset(loc, rewriter, typeCode)};
    };
    auto doFloat = [&](unsigned width) -> std::tuple<mlir::Value, mlir::Value> {
      int typeCode = fir::realBitsToTypeCode(width);
      return {this->genConstantOffset(loc, rewriter, width / 8),
              this->genConstantOffset(loc, rewriter, typeCode)};
    };
    auto doComplex =
        [&](unsigned width) -> std::tuple<mlir::Value, mlir::Value> {
      auto typeCode = fir::complexBitsToTypeCode(width);
      return {this->genConstantOffset(loc, rewriter, width / 8 * 2),
              this->genConstantOffset(loc, rewriter, typeCode)};
    };
    auto doCharacter =
        [&](unsigned width,
            mlir::Value len) -> std::tuple<mlir::Value, mlir::Value> {
      auto typeCode = fir::characterBitsToTypeCode(width);
      auto typeCodeVal = this->genConstantOffset(loc, rewriter, typeCode);
      if (width == 8)
        return {len, typeCodeVal};
      auto byteWidth = this->genConstantOffset(loc, rewriter, width / 8);
      auto i64Ty = mlir::IntegerType::get(&this->lowerTy().getContext(), 64);
      auto size =
          rewriter.create<mlir::LLVM::MulOp>(loc, i64Ty, byteWidth, len);
      return {size, typeCodeVal};
    };
    auto getKindMap = [&]() -> fir::KindMapping & {
      return this->lowerTy().getKindMap();
    };

    if (auto eleTy = fir::dyn_cast_ptrEleTy(boxEleTy))
      boxEleTy = eleTy;
    if (fir::isa_integer(boxEleTy)) {
      if (auto ty = boxEleTy.dyn_cast<mlir::IntegerType>())
        return doInteger(ty.getWidth());
      auto ty = boxEleTy.cast<fir::IntegerType>();
      return doInteger(getKindMap().getIntegerBitsize(ty.getFKind()));
    }
    if (fir::isa_real(boxEleTy)) {
      if (auto ty = boxEleTy.dyn_cast<mlir::FloatType>())
        return doFloat(ty.getWidth());
      auto ty = boxEleTy.cast<fir::RealType>();
      return doFloat(getKindMap().getRealBitsize(ty.getFKind()));
    }
    if (fir::isa_complex(boxEleTy)) {
      if (auto ty = boxEleTy.dyn_cast<mlir::ComplexType>())
        return doComplex(
            ty.getElementType().cast<mlir::FloatType>().getWidth());
      auto ty = boxEleTy.cast<fir::ComplexType>();
      return doComplex(getKindMap().getRealBitsize(ty.getFKind()));
    }
    if (auto ty = boxEleTy.dyn_cast<fir::CharacterType>()) {
      auto charWidth = getKindMap().getCharacterBitsize(ty.getFKind());
      if (ty.getLen() != fir::CharacterType::unknownLen()) {
        auto len = this->genConstantOffset(loc, rewriter, ty.getLen());
        return doCharacter(charWidth, len);
      }
      assert(!lenParams.empty());
      return doCharacter(charWidth, lenParams.back());
    }
    if (auto ty = boxEleTy.dyn_cast<fir::LogicalType>())
      return doLogical(getKindMap().getLogicalBitsize(ty.getFKind()));
    if (auto seqTy = boxEleTy.dyn_cast<fir::SequenceType>())
      return getSizeAndTypeCode(loc, rewriter, seqTy.getEleTy(), lenParams);
    if (boxEleTy.isa<fir::RecordType>()) {
      auto ptrTy = mlir::LLVM::LLVMPointerType::get(
          this->lowerTy().convertType(boxEleTy));
      auto nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, ptrTy);
      auto one =
          genConstantIndex(loc, this->lowerTy().offsetType(), rewriter, 1);
      auto gep = rewriter.create<mlir::LLVM::GEPOp>(
          loc, ptrTy, mlir::ValueRange{nullPtr, one});
      auto eleSize = rewriter.create<mlir::LLVM::PtrToIntOp>(
          loc, this->lowerTy().indexType(), gep);
      return {eleSize,
              this->genConstantOffset(loc, rewriter, fir::derivedToTypeCode())};
    }
    if (fir::isa_ref_type(boxEleTy)) {
      // FIXME: use the target pointer size rather than sizeof(void*)
      return {this->genConstantOffset(loc, rewriter, sizeof(void *)),
              this->genConstantOffset(loc, rewriter, CFI_type_cptr)};
    }
    fir::emitFatalError(loc, "unhandled type in fir.box code generation");
  }

  /// Basic pattern to write a field in the descriptor
  mlir::Value insertField(mlir::ConversionPatternRewriter &rewriter,
                          mlir::Location loc, mlir::Value dest,
                          ArrayRef<unsigned> fldIndexes, mlir::Value value,
                          bool bitcast = false) const {
    auto boxTy = dest.getType();
    auto fldTy = this->getBoxEleTy(boxTy, fldIndexes);
    if (bitcast)
      value = rewriter.create<mlir::LLVM::BitcastOp>(loc, fldTy, value);
    else
      value = this->integerCast(loc, rewriter, fldTy, value);
    SmallVector<mlir::Attribute, 2> attrs;
    for (auto i : fldIndexes)
      attrs.push_back(rewriter.getI32IntegerAttr(i));
    auto indexesAttr = mlir::ArrayAttr::get(rewriter.getContext(), attrs);
    return rewriter.create<mlir::LLVM::InsertValueOp>(loc, boxTy, dest, value,
                                                      indexesAttr);
  }
  mlir::Value insertLowerBound(mlir::ConversionPatternRewriter &rewriter,
                               mlir::Location loc, mlir::Value dest,
                               unsigned dim, mlir::Value lb) const {
    return insertField(rewriter, loc, dest, {7, dim, 0}, lb);
  }
  mlir::Value insertExtent(mlir::ConversionPatternRewriter &rewriter,
                           mlir::Location loc, mlir::Value dest, unsigned dim,
                           mlir::Value extent) const {
    return insertField(rewriter, loc, dest, {7, dim, 1}, extent);
  }
  mlir::Value insertStride(mlir::ConversionPatternRewriter &rewriter,
                           mlir::Location loc, mlir::Value dest, unsigned dim,
                           mlir::Value stride) const {
    return insertField(rewriter, loc, dest, {7, dim, 2}, stride);
  }
  mlir::Value insertBaseAddress(mlir::ConversionPatternRewriter &rewriter,
                                mlir::Location loc, mlir::Value dest,
                                mlir::Value base) const {
    return insertField(rewriter, loc, dest, {0}, base, /*bitCast=*/true);
  }

  /// Get the address of the type descriptor global variable that was created by
  /// lowering for derived type \p recType.
  template <typename BOX>
  mlir::Value
  getTypeDescriptor(BOX box, mlir::ConversionPatternRewriter &rewriter,
                    mlir::Location loc, fir::RecordType recType) const {
    auto split = recType.getName().split('T');
    std::string name = (split.first + "E.dt." + split.second).str();
    auto module = box->template getParentOfType<mlir::ModuleOp>();
    if (auto global = module.template lookupSymbol<fir::GlobalOp>(name)) {
      auto ty = mlir::LLVM::LLVMPointerType::get(
          this->lowerTy().convertType(global.getType()));
      return rewriter.create<mlir::LLVM::AddressOfOp>(loc, ty,
                                                      global.sym_name());
    }
    if (auto global =
            module.template lookupSymbol<mlir::LLVM::GlobalOp>(name)) {
      // The global may have already been translated to LLVM.
      auto ty = mlir::LLVM::LLVMPointerType::get(global.getType());
      return rewriter.create<mlir::LLVM::AddressOfOp>(loc, ty,
                                                      global.sym_name());
    }
    // The global does not exist in the current translation unit, but may be
    // defined elsewhere (e.g., type defined in a module).
    // For now, create a extern_weak symbols (will become nullptr if unresolved)
    // to support generating code without the front-end generated symbols.
    // These could be made available_externally to require the symbols to be
    // defined elsewhere and to cause link-time failure otherwise.
    auto i8Ty = rewriter.getIntegerType(8);
    mlir::OpBuilder modBuilder(module.getBodyRegion());
    // TODO: The symbol should be lowered to constant in lowering, they are read
    // only.
    modBuilder.create<mlir::LLVM::GlobalOp>(loc, i8Ty, /*isConstant=*/false,
                                            mlir::LLVM::Linkage::ExternWeak,
                                            name, mlir::Attribute{});
    auto ty = mlir::LLVM::LLVMPointerType::get(i8Ty);
    return rewriter.create<mlir::LLVM::AddressOfOp>(loc, ty, name);
  }

  template <typename BOX>
  std::tuple<fir::BoxType, mlir::Value, mlir::Value>
  consDescriptorPrefix(BOX box, mlir::ConversionPatternRewriter &rewriter,
                       unsigned rank, mlir::ValueRange lenParams) const {
    auto loc = box.getLoc();
    auto boxTy = box.getType().template dyn_cast<fir::BoxType>();
    auto convTy = this->lowerTy().convertBoxType(boxTy, rank);
    auto llvmBoxPtrTy = convTy.template cast<mlir::LLVM::LLVMPointerType>();
    auto llvmBoxTy = llvmBoxPtrTy.getElementType();
    mlir::Value dest = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmBoxTy);

    llvm::SmallVector<mlir::Value> typeparams = lenParams;
    if constexpr (!std::is_same_v<BOX, fir::EmboxOp>) {
      if (!box.substr().empty() && fir::hasDynamicSize(boxTy.getEleTy()))
        typeparams.push_back(box.substr()[1]);
    }

    // Write each of the fields with the appropriate values
    auto [eleSize, cfiTy] =
        getSizeAndTypeCode(loc, rewriter, boxTy.getEleTy(), typeparams);
    dest = insertField(rewriter, loc, dest, {1}, eleSize);
    dest = insertField(rewriter, loc, dest, {2},
                       this->genConstantOffset(loc, rewriter, CFI_VERSION));
    dest = insertField(rewriter, loc, dest, {3},
                       this->genConstantOffset(loc, rewriter, rank));
    dest = insertField(rewriter, loc, dest, {4}, cfiTy);
    dest =
        insertField(rewriter, loc, dest, {5},
                    this->genConstantOffset(loc, rewriter, getCFIAttr(boxTy)));
    const bool hasAddendum = isDerivedType(boxTy);
    dest = insertField(rewriter, loc, dest, {6},
                       this->genConstantOffset(loc, rewriter, hasAddendum));

    if (hasAddendum) {
      auto isArray =
          fir::dyn_cast_ptrOrBoxEleTy(boxTy).template isa<fir::SequenceType>();
      unsigned typeDescFieldId = isArray ? 8 : 7;
      auto typeDesc =
          getTypeDescriptor(box, rewriter, loc, unwrapIfDerived(boxTy));
      dest = insertField(rewriter, loc, dest, {typeDescFieldId}, typeDesc,
                         /*bitCast=*/true);
    }

    return {boxTy, dest, eleSize};
  }

  /// Compute the base address of a substring given the base address of a scalar
  /// string and the zero based string lower bound.
  mlir::Value shiftSubstringBase(mlir::ConversionPatternRewriter &rewriter,
                                 mlir::Location loc, mlir::Value base,
                                 mlir::Value lowerBound) const {
    llvm::SmallVector<mlir::Value> gepOperands;
    auto baseType =
        base.getType().cast<mlir::LLVM::LLVMPointerType>().getElementType();
    if (baseType.isa<mlir::LLVM::LLVMArrayType>()) {
      auto idxTy = this->lowerTy().indexType();
      mlir::Value zero = genConstantIndex(loc, idxTy, rewriter, 0);
      gepOperands.push_back(zero);
      gepOperands.push_back(lowerBound);
    } else {
      gepOperands.push_back(lowerBound);
    }
    return this->genGEP(loc, base.getType(), rewriter, base, gepOperands);
  }

  /// If the embox is not in a globalOp body, allocate storage for the box and
  /// store the value inside. Return the input value otherwise.
  mlir::Value
  placeInMemoryIfNotGlobalInit(mlir::ConversionPatternRewriter &rewriter,
                               mlir::Location loc, mlir::Value boxValue) const {
    auto *thisBlock = rewriter.getInsertionBlock();
    if (thisBlock && mlir::isa<mlir::LLVM::GlobalOp>(thisBlock->getParentOp()))
      return boxValue;
    auto boxPtrTy = mlir::LLVM::LLVMPointerType::get(boxValue.getType());
    auto alloca = genAllocaWithType(loc, boxPtrTy, defaultAlign, rewriter);
    rewriter.create<mlir::LLVM::StoreOp>(loc, boxValue, alloca);
    return alloca;
  }
};

/// Compute the extent of a triplet slice (lb:ub:step).
static mlir::Value
computeTripletExtent(mlir::ConversionPatternRewriter &rewriter,
                     mlir::Location loc, mlir::Value lb, mlir::Value ub,
                     mlir::Value step, mlir::Value zero, mlir::Type type) {
  // auto type = ub.getType();
  mlir::Value extent = rewriter.create<mlir::LLVM::SubOp>(loc, type, ub, lb);
  extent = rewriter.create<mlir::LLVM::AddOp>(loc, type, extent, step);
  extent = rewriter.create<mlir::LLVM::SDivOp>(loc, type, extent, step);
  // If the resulting extent is negative (`ub-lb` and `step` have different
  // signs), zero must be returned instead.
  auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
      loc, mlir::LLVM::ICmpPredicate::sgt, extent, zero);
  return rewriter.create<mlir::LLVM::SelectOp>(loc, cmp, extent, zero);
}

/// Create a generic box on a memory reference. This conversions lowers the
/// abstract box to the appropriate, initialized descriptor.
struct EmboxOpConversion : public EmboxCommonConversion<fir::EmboxOp> {
  using EmboxCommonConversion::EmboxCommonConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::EmboxOp embox, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // There should be no dims on this embox op
    assert(!embox.getShape());
    auto [boxTy, dest, eleSize] = consDescriptorPrefix(
        embox, rewriter, /*rank=*/0, /*lenParams=*/operands.drop_front(1));
    dest = insertBaseAddress(rewriter, embox.getLoc(), dest, operands[0]);
    if (isDerivedTypeWithLenParams(boxTy))
      TODO(embox.getLoc(),
           "fir.embox codegen of derived with length parameters");
    auto result = placeInMemoryIfNotGlobalInit(rewriter, embox.getLoc(), dest);
    rewriter.replaceOp(embox, result);
    return success();
  }
};

/// create a generic box on a memory reference
struct XEmboxOpConversion : public EmboxCommonConversion<fir::cg::XEmboxOp> {
  using EmboxCommonConversion::EmboxCommonConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::cg::XEmboxOp xbox, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto [boxTy, dest, eleSize] =
        consDescriptorPrefix(xbox, rewriter, xbox.getOutRank(),
                             operands.drop_front(xbox.lenParamOffset()));
    // Generate the triples in the dims field of the descriptor
    auto i64Ty = mlir::IntegerType::get(xbox.getContext(), 64);
    auto base = operands[0];
    assert(!xbox.shape().empty() && "must have a shape");
    unsigned shapeOff = xbox.shapeOffset();
    bool hasShift = !xbox.shift().empty();
    unsigned shiftOff = xbox.shiftOffset();
    bool hasSlice = !xbox.slice().empty();
    unsigned sliceOff = xbox.sliceOffset();
    auto loc = xbox.getLoc();
    mlir::Value zero = genConstantIndex(loc, i64Ty, rewriter, 0);
    mlir::Value one = genConstantIndex(loc, i64Ty, rewriter, 1);
    mlir::Value prevDim = integerCast(loc, rewriter, i64Ty, eleSize);
    mlir::Value prevPtrOff = one;
    auto eleTy = boxTy.getEleTy();
    const auto rank = xbox.getRank();
    llvm::SmallVector<mlir::Value> gepArgs;
    unsigned constRows = 0;
    mlir::Value ptrOffset = zero;
    if (auto memEleTy = fir::dyn_cast_ptrEleTy(xbox.memref().getType()))
      if (auto seqTy = memEleTy.dyn_cast<fir::SequenceType>()) {
        auto seqEleTy = seqTy.getEleTy();
        // Adjust the element scaling factor if the element is a dependent type.
        if (fir::hasDynamicSize(seqEleTy)) {
          if (fir::isa_char(seqEleTy)) {
            assert(xbox.lenParams().size() == 1);
            prevPtrOff = integerCast(loc, rewriter, i64Ty,
                                     operands[xbox.lenParamOffset()]);
          } else if (seqEleTy.isa<fir::RecordType>()) {
            // prevPtrOff = ;
            TODO(loc, "generate call to calculate size of PDT");
          } else {
            fir::emitFatalError(loc, "unexpected dynamic type");
          }
        } else {
          constRows = seqTy.getConstantRows();
        }
      }

    auto hasSubcomp = !xbox.subcomponent().empty();
    mlir::Value stepExpr;
    if (hasSubcomp) {
      // We have a subcomponent. The step value needs to be the number of
      // bytes per element (which is a derived type).
      auto ty0 = base.getType();
      [[maybe_unused]] auto ptrTy = ty0.dyn_cast<mlir::LLVM::LLVMPointerType>();
      assert(ptrTy && "expected pointer type");
      auto memEleTy = fir::dyn_cast_ptrEleTy(xbox.memref().getType());
      assert(memEleTy && "expected fir pointer type");
      auto seqTy = memEleTy.dyn_cast<fir::SequenceType>();
      assert(seqTy && "expected sequence type");
      auto seqEleTy = seqTy.getEleTy();
      auto eleTy = mlir::LLVM::LLVMPointerType::get(convertType(seqEleTy));
      stepExpr = computeDerivedTypeSize(loc, eleTy, i64Ty, rewriter);
    }

    // Process the array subspace arguments (shape, shift, etc.), if any,
    // translating everything to values in the descriptor wherever the entity
    // has a dynamic array dimension.
    for (unsigned di = 0, descIdx = 0; di < rank; ++di) {
      mlir::Value extent = operands[shapeOff];
      mlir::Value outerExtent = extent;
      bool skipNext = false;
      if (hasSlice) {
        auto off = operands[sliceOff];
        auto adj = one;
        if (hasShift)
          adj = operands[shiftOff];
        auto ao = rewriter.create<mlir::LLVM::SubOp>(loc, i64Ty, off, adj);
        if (constRows > 0) {
          gepArgs.push_back(ao);
          --constRows;
        } else {
          auto dimOff =
              rewriter.create<mlir::LLVM::MulOp>(loc, i64Ty, ao, prevPtrOff);
          ptrOffset =
              rewriter.create<mlir::LLVM::AddOp>(loc, i64Ty, dimOff, ptrOffset);
        }
        if (mlir::isa_and_nonnull<fir::UndefOp>(
                xbox.slice()[3 * di + 1].getDefiningOp())) {
          // This dimension contains a scalar expression in the array slice op.
          // The dimension is loop invariant, will be dropped, and will not
          // appear in the descriptor.
          skipNext = true;
        }
      }
      if (!skipNext) {
        // store lower bound (normally 0)
        auto lb = zero;
        if (eleTy.isa<fir::PointerType>() || eleTy.isa<fir::HeapType>()) {
          lb = one;
          if (hasShift)
            lb = operands[shiftOff];
        }
        dest = insertLowerBound(rewriter, loc, dest, descIdx, lb);

        // store extent
        if (hasSlice)
          extent = computeTripletExtent(rewriter, loc, operands[sliceOff],
                                        operands[sliceOff + 1],
                                        operands[sliceOff + 2], zero, i64Ty);
        dest = insertExtent(rewriter, loc, dest, descIdx, extent);

        // store step (scaled by shaped extent)

        mlir::Value step = hasSubcomp ? stepExpr : prevDim;
        if (hasSlice)
          step = rewriter.create<mlir::LLVM::MulOp>(loc, i64Ty, step,
                                                    operands[sliceOff + 2]);
        dest = insertStride(rewriter, loc, dest, descIdx, step);
        ++descIdx;
      }

      // compute the stride and offset for the next natural dimension
      prevDim =
          rewriter.create<mlir::LLVM::MulOp>(loc, i64Ty, prevDim, outerExtent);
      if (constRows == 0)
        prevPtrOff = rewriter.create<mlir::LLVM::MulOp>(loc, i64Ty, prevPtrOff,
                                                        outerExtent);

      // increment iterators
      ++shapeOff;
      if (hasShift)
        ++shiftOff;
      if (hasSlice)
        sliceOff += 3;
    }
    if (hasSlice || hasSubcomp || !xbox.substr().empty()) {
      llvm::SmallVector<mlir::Value> args = {base, ptrOffset};
      args.append(gepArgs.rbegin(), gepArgs.rend());
      if (hasSubcomp) {
        // For each field in the path add the offset to base via the args list.
        // In the most general case, some offsets must be computed since
        // they are not be known until runtime.
        if (fir::hasDynamicSize(fir::unwrapSequenceType(
                fir::unwrapPassByRefType(xbox.memref().getType()))))
          TODO(loc, "fir.embox codegen dynamic size component in derived type");
        const auto *beginOffset = xbox.subcomponentOffset() + operands.begin();
        const auto *lastOffset = beginOffset + xbox.subcomponent().size();
        args.append(beginOffset, lastOffset);
      }
      base = rewriter.create<mlir::LLVM::GEPOp>(loc, base.getType(), args);
      if (!xbox.substr().empty())
        base = shiftSubstringBase(rewriter, loc, base, xbox.substr()[0]);
    }
    dest = insertBaseAddress(rewriter, loc, dest, base);
    if (isDerivedTypeWithLenParams(boxTy))
      TODO(loc, "fir.embox codegen of derived with length parameters");

    auto result = placeInMemoryIfNotGlobalInit(rewriter, loc, dest);
    rewriter.replaceOp(xbox, result);
    return success();
  }
};

/// Create a new box given a box reference.
struct XReboxOpConversion : public EmboxCommonConversion<fir::cg::XReboxOp> {
  using EmboxCommonConversion::EmboxCommonConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::cg::XReboxOp rebox, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = rebox.getLoc();
    auto idxTy = lowerTy().indexType();
    auto loweredBox = operands[0];

    // Create new descriptor and fill its non-shape related data.
    llvm::SmallVector<mlir::Value, 2> lenParams;
    auto inputEleTy = getInputEleTy(rebox);
    if (auto charTy = inputEleTy.dyn_cast<fir::CharacterType>()) {
      auto len = loadElementSizeFromBox(loc, idxTy, loweredBox, rewriter);
      if (charTy.getFKind() != 1) {
        auto width = genConstantIndex(loc, idxTy, rewriter, charTy.getFKind());
        len = rewriter.create<mlir::LLVM::SDivOp>(loc, idxTy, len, width);
      }
      lenParams.emplace_back(len);
    } else if (auto recTy = inputEleTy.dyn_cast<fir::RecordType>()) {
      if (recTy.getNumLenParams() != 0)
        TODO(loc, "reboxing descriptor of derived type with length parameters");
    }
    auto [boxTy, dest, eleSize] =
        consDescriptorPrefix(rebox, rewriter, rebox.getOutRank(), lenParams);

    // Read input extents, strides, and base address
    llvm::SmallVector<mlir::Value> inputExtents;
    llvm::SmallVector<mlir::Value> inputStrides;
    const auto inputRank = rebox.getRank();
    for (unsigned i = 0; i < inputRank; ++i) {
      auto dim = genConstantIndex(loc, idxTy, rewriter, i);
      auto dimInfo =
          getDimsFromBox(loc, {idxTy, idxTy, idxTy}, loweredBox, dim, rewriter);
      inputExtents.emplace_back(dimInfo[1]);
      inputStrides.emplace_back(dimInfo[2]);
    }

    auto baseTy = getBaseAddrTypeFromBox(loweredBox.getType());
    auto baseAddr = loadBaseAddrFromBox(loc, baseTy, loweredBox, rewriter);

    if (!rebox.slice().empty() || !rebox.subcomponent().empty())
      return sliceBox(rebox, dest, baseAddr, inputExtents, inputStrides,
                      rewriter);
    return reshapeBox(rebox, dest, baseAddr, inputExtents, inputStrides,
                      rewriter);
  }

private:
  /// Write resulting shape and base address in descriptor, and replace rebox
  /// op.
  mlir::LogicalResult
  finalizeRebox(fir::cg::XReboxOp rebox, mlir::Value dest, mlir::Value base,
                mlir::ValueRange lbounds, mlir::ValueRange extents,
                mlir::ValueRange strides,
                mlir::ConversionPatternRewriter &rewriter) const {
    auto loc = rebox.getLoc();
    auto one = genConstantIndex(loc, lowerTy().indexType(), rewriter, 1);
    for (auto iter : llvm::enumerate(llvm::zip(extents, strides))) {
      auto dim = iter.index();
      auto lb = lbounds.empty() ? one : lbounds[dim];
      dest = insertLowerBound(rewriter, loc, dest, dim, lb);
      dest = insertExtent(rewriter, loc, dest, dim, std::get<0>(iter.value()));
      dest = insertStride(rewriter, loc, dest, dim, std::get<1>(iter.value()));
    }
    dest = insertBaseAddress(rewriter, loc, dest, base);
    auto result = placeInMemoryIfNotGlobalInit(rewriter, rebox.getLoc(), dest);
    rewriter.replaceOp(rebox, result);
    return success();
  }

  // Apply slice given the base address, extents and strides of the input box.
  mlir::LogicalResult
  sliceBox(fir::cg::XReboxOp rebox, mlir::Value dest, mlir::Value base,
           mlir::ValueRange inputExtents, mlir::ValueRange inputStrides,
           mlir::ConversionPatternRewriter &rewriter) const {
    auto loc = rebox.getLoc();
    auto voidPtrTy = getVoidPtrType(rebox.getContext());
    auto idxTy = lowerTy().indexType();
    mlir::Value zero = genConstantIndex(loc, idxTy, rewriter, 0);
    // Apply subcomponent and substring shift on base address.
    if (!rebox.subcomponent().empty() || !rebox.substr().empty()) {
      // Cast to inputEleTy* so that a GEP can be used.
      auto inputEleTy = getInputEleTy(rebox);
      auto llvmElePtrTy =
          mlir::LLVM::LLVMPointerType::get(convertType(inputEleTy));
      base = rewriter.create<mlir::LLVM::BitcastOp>(loc, llvmElePtrTy, base);

      if (!rebox.subcomponent().empty()) {
        llvm::SmallVector<mlir::Value> gepOperands = {zero};
        gepOperands.append(rebox.subcomponent().begin(),
                           rebox.subcomponent().end());
        base = genGEP(loc, llvmElePtrTy, rewriter, base, gepOperands);
      }
      if (!rebox.substr().empty())
        base = shiftSubstringBase(rewriter, loc, base, rebox.substr()[0]);
    }

    if (rebox.slice().empty())
      // The array section is of the form array[%component][substring], keep
      // the input array extents and strides.
      return finalizeRebox(rebox, dest, base, /*lbounds*/ llvm::None,
                           inputExtents, inputStrides, rewriter);

    // Strides from the fir.box are in bytes.
    base = rewriter.create<mlir::LLVM::BitcastOp>(loc, voidPtrTy, base);

    // The slice is of the form array(i:j:k)[%component]. Compute new extents
    // and strides.
    llvm::SmallVector<mlir::Value> slicedExtents;
    llvm::SmallVector<mlir::Value> slicedStrides;
    auto one = genConstantIndex(loc, idxTy, rewriter, 1);
    const bool sliceHasOrigins = !rebox.shift().empty();
    auto sliceOps = rebox.slice().begin();
    auto shiftOps = rebox.shift().begin();
    auto strideOps = inputStrides.begin();
    const auto inputRank = inputStrides.size();
    for (unsigned i = 0; i < inputRank;
         ++i, ++strideOps, ++shiftOps, sliceOps += 3) {
      auto sliceLb = integerCast(loc, rewriter, idxTy, *sliceOps);
      auto inputStride = *strideOps; // already idxTy
      // Apply origin shift: base += (lb-shift)*input_stride
      auto sliceOrigin =
          sliceHasOrigins ? integerCast(loc, rewriter, idxTy, *shiftOps) : one;
      auto diff =
          rewriter.create<mlir::LLVM::SubOp>(loc, idxTy, sliceLb, sliceOrigin);
      auto offset =
          rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, diff, inputStride);
      base = genGEP(loc, voidPtrTy, rewriter, base, offset);
      // Apply upper bound and step if this is a triplet. Otherwise, the
      // dimension is dropped and no extents/strides are computed.
      mlir::Value upper = *(sliceOps + 1);
      const bool isTripletSlice =
          !mlir::isa_and_nonnull<fir::UndefOp>(upper.getDefiningOp());
      if (isTripletSlice) {
        auto step = integerCast(loc, rewriter, idxTy, *(sliceOps + 2));
        // extent = ub-lb+step/step
        auto sliceUb = integerCast(loc, rewriter, idxTy, upper);
        auto extent = computeTripletExtent(rewriter, loc, sliceLb, sliceUb,
                                           step, zero, idxTy);
        slicedExtents.emplace_back(extent);
        // stride = step*input_stride
        auto stride =
            rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, step, inputStride);
        slicedStrides.emplace_back(stride);
      }
    }
    return finalizeRebox(rebox, dest, base, /*lbounds*/ llvm::None,
                         slicedExtents, slicedStrides, rewriter);
  }

  /// Apply a new shape to the data described by a box given the base address,
  /// extents and strides of the box.
  mlir::LogicalResult
  reshapeBox(fir::cg::XReboxOp rebox, mlir::Value dest, mlir::Value base,
             mlir::ValueRange inputExtents, mlir::ValueRange inputStrides,
             mlir::ConversionPatternRewriter &rewriter) const {
    if (rebox.shape().empty())
      // Only setting new lower bounds.
      return finalizeRebox(rebox, dest, base, rebox.shift(), inputExtents,
                           inputStrides, rewriter);

    auto loc = rebox.getLoc();
    // Strides from the fir.box are in bytes.
    auto voidPtrTy = getVoidPtrType(rebox.getContext());
    base = rewriter.create<mlir::LLVM::BitcastOp>(loc, voidPtrTy, base);

    llvm::SmallVector<mlir::Value> newStrides;
    llvm::SmallVector<mlir::Value> newExtents;
    auto idxTy = lowerTy().indexType();
    // First stride from input box is kept. The rest is assumed contiguous
    // (it is not possible to reshape otherwise). If the input is scalar,
    // which may be OK if all new extents are ones, the stride does not
    // matter, use one.
    auto stride = inputStrides.empty()
                      ? genConstantIndex(loc, idxTy, rewriter, 1)
                      : inputStrides[0];
    for (auto rawExtent : rebox.shape()) {
      auto extent = integerCast(loc, rewriter, idxTy, rawExtent);
      newExtents.emplace_back(extent);
      newStrides.emplace_back(stride);
      // nextStride = extent * stride;
      stride = rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, extent, stride);
    }
    return finalizeRebox(rebox, dest, base, rebox.shift(), newExtents,
                         newStrides, rewriter);
  }

  /// Return scalar element type of the input box.
  static mlir::Type getInputEleTy(fir::cg::XReboxOp rebox) {
    auto ty = fir::dyn_cast_ptrOrBoxEleTy(rebox.box().getType());
    if (auto seqTy = ty.dyn_cast<fir::SequenceType>())
      return seqTy.getEleTy();
    return ty;
  }
};

/// create a procedure pointer box
struct EmboxProcOpConversion : public FIROpConversion<fir::EmboxProcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::EmboxProcOp emboxproc, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = emboxproc.getLoc();
    auto *ctx = emboxproc.getContext();
    auto ty = convertType(emboxproc.getType());
    auto c0 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(0));
    auto c1 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(1));
    auto un = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    auto r = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, un,
                                                        operands[0], c0);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(emboxproc, ty, r,
                                                           operands[1], c1);
    return success();
  }
};

// Code shared between insert_value and extract_value Ops.
struct ValueOpCommon {
  static mlir::Attribute getValue(mlir::Value value) {
    auto *defOp = value.getDefiningOp();
    if (auto v = dyn_cast<mlir::LLVM::ConstantOp>(defOp))
      return v.value();
    if (auto v = dyn_cast<mlir::arith::ConstantOp>(defOp))
      return v.value();
    llvm_unreachable("must be a constant op");
    return {};
  }

  // Translate the arguments pertaining to any multidimensional array to
  // row-major order for LLVM-IR.
  static void toRowMajor(SmallVectorImpl<mlir::Attribute> &attrs,
                         mlir::Type ty) {
    assert(ty && "type is null");
    const auto end = attrs.size();
    for (std::remove_const_t<decltype(end)> i = 0; i < end; ++i) {
      if (auto seq = ty.dyn_cast<mlir::LLVM::LLVMArrayType>()) {
        const auto dim = getDimension(seq);
        if (dim > 1) {
          auto ub = std::min(i + dim, end);
          std::reverse(attrs.begin() + i, attrs.begin() + ub);
          i += dim - 1;
        }
        ty = getArrayElementType(seq);
      } else if (auto st = ty.dyn_cast<mlir::LLVM::LLVMStructType>()) {
        ty = st.getBody()[attrs[i].cast<mlir::IntegerAttr>().getInt()];
      } else {
        llvm_unreachable("index into invalid type");
      }
    }
  }

  static llvm::SmallVector<mlir::Attribute>
  collectIndices(mlir::ConversionPatternRewriter &rewriter,
                 mlir::ArrayAttr arrAttr) {
    llvm::SmallVector<mlir::Attribute> attrs;
    for (auto i = arrAttr.begin(), e = arrAttr.end(); i != e; ++i) {
      if (i->isa<mlir::IntegerAttr>()) {
        attrs.push_back(*i);
      } else {
        auto fieldName = i->cast<mlir::StringAttr>().getValue();
        ++i;
        auto ty = i->cast<mlir::TypeAttr>().getValue();
        auto index = ty.cast<fir::RecordType>().getFieldIndex(fieldName);
        attrs.push_back(mlir::IntegerAttr::get(rewriter.getI32Type(), index));
      }
    }
    return attrs;
  }

private:
  static unsigned getDimension(mlir::LLVM::LLVMArrayType ty) {
    unsigned result = 1;
    for (auto eleTy = ty.getElementType().dyn_cast<mlir::LLVM::LLVMArrayType>();
         eleTy;
         eleTy = eleTy.getElementType().dyn_cast<mlir::LLVM::LLVMArrayType>())
      ++result;
    return result;
  }

  static mlir::Type getArrayElementType(mlir::LLVM::LLVMArrayType ty) {
    auto eleTy = ty.getElementType();
    while (auto arrTy = eleTy.dyn_cast<mlir::LLVM::LLVMArrayType>())
      eleTy = arrTy.getElementType();
    return eleTy;
  }
};

/// Extract a subobject value from an ssa-value of aggregate type
struct ExtractValueOpConversion
    : public FIROpAndTypeConversion<fir::ExtractValueOp>,
      public ValueOpCommon {
  using FIROpAndTypeConversion::FIROpAndTypeConversion;

  mlir::LogicalResult
  doRewrite(fir::ExtractValueOp extractVal, mlir::Type ty, OperandTy operands,
            mlir::ConversionPatternRewriter &rewriter) const override {
    if (!fir::allConstants(operands.drop_front(1)))
      llvm_unreachable("fir.extract_value incorrectly formed");
    auto attrs = collectIndices(rewriter, extractVal.coor());
    toRowMajor(attrs, operands[0].getType());
    auto position = mlir::ArrayAttr::get(extractVal.getContext(), attrs);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
        extractVal, ty, operands[0], position);
    return success();
  }
};

/// InsertValue is the generalized instruction for the composition of new
/// aggregate type values.
struct InsertValueOpConversion
    : public FIROpAndTypeConversion<fir::InsertValueOp>,
      public ValueOpCommon {
  using FIROpAndTypeConversion::FIROpAndTypeConversion;

  mlir::LogicalResult
  doRewrite(fir::InsertValueOp insertVal, mlir::Type ty, OperandTy operands,
            mlir::ConversionPatternRewriter &rewriter) const override {
    assert(fir::allConstants(operands.drop_front(2)));
    auto attrs = collectIndices(rewriter, insertVal.coor());
    toRowMajor(attrs, operands[0].getType());
    auto position = mlir::ArrayAttr::get(insertVal.getContext(), attrs);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        insertVal, ty, operands[0], operands[1], position);
    return success();
  }
};

/// InsertOnRange inserts a value into a sequence over a range of offsets.
struct InsertOnRangeOpConversion
    : public FIROpAndTypeConversion<fir::InsertOnRangeOp>,
      public ValueOpCommon {
  using FIROpAndTypeConversion::FIROpAndTypeConversion;

  // Increments an array of subscripts in a row major fasion.
  void incrementSubscripts(const SmallVector<uint64_t> &dims,
                           SmallVector<uint64_t> &subscripts) const {
    for (size_t i = dims.size(); i > 0; --i) {
      if (++subscripts[i - 1] < dims[i - 1]) {
        return;
      }
      subscripts[i - 1] = 0;
    }
  }

  mlir::LogicalResult
  doRewrite(fir::InsertOnRangeOp range, mlir::Type ty, OperandTy operands,
            mlir::ConversionPatternRewriter &rewriter) const override {
    assert(fir::allConstants(operands.drop_front(2)));

    llvm::SmallVector<mlir::Attribute> lowerBound;
    llvm::SmallVector<mlir::Attribute> upperBound;
    llvm::SmallVector<uint64_t> dims;
    auto type = operands[0].getType();

    // Iteratively extract the array dimensions from the type.
    while (auto t = type.dyn_cast<mlir::LLVM::LLVMArrayType>()) {
      dims.push_back(t.getNumElements());
      type = t.getElementType();
    }

    // Unzip the upper and lower bound subscripts.
    for (auto i = range.coor().begin(), e = range.coor().end(); i != e; ++i) {
      lowerBound.push_back(*i++);
      upperBound.push_back(*i);
    }

    SmallVector<std::uint64_t> lBounds;
    SmallVector<std::uint64_t> uBounds;

    // Extract the integer value from the attribute bounds and convert to row
    // major format.
    for (std::size_t i = lowerBound.size(); i > 0; --i) {
      lBounds.push_back(lowerBound[i - 1].cast<IntegerAttr>().getInt());
      uBounds.push_back(upperBound[i - 1].cast<IntegerAttr>().getInt());
    }

    auto subscripts(lBounds);
    auto loc = range.getLoc();
    mlir::Value lastOp = operands[0];
    mlir::Value insertVal = operands[1];

    while (subscripts != uBounds) {
      // Convert uint64_t's to Attribute's.
      SmallVector<mlir::Attribute> subscriptAttrs;
      for (const auto &subscript : subscripts)
        subscriptAttrs.push_back(
            IntegerAttr::get(rewriter.getI64Type(), subscript));
      mlir::ArrayRef<mlir::Attribute> arrayRef(subscriptAttrs);
      lastOp = rewriter.create<mlir::LLVM::InsertValueOp>(
          loc, ty, lastOp, insertVal,
          ArrayAttr::get(range.getContext(), arrayRef));

      incrementSubscripts(dims, subscripts);
    }

    // Convert uint64_t's to Attribute's.
    SmallVector<mlir::Attribute> subscriptAttrs;
    for (const auto &subscript : subscripts)
      subscriptAttrs.push_back(
          IntegerAttr::get(rewriter.getI64Type(), subscript));
    mlir::ArrayRef<mlir::Attribute> arrayRef(subscriptAttrs);

    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        range, ty, lastOp, insertVal,
        ArrayAttr::get(range.getContext(), arrayRef));

    return success();
  }
};

/// XArrayCoor is the address arithmetic on a dynamically shaped, etc. array.
/// (See the static restriction on coordinate_of.) array_coor determines the
/// coordinate (location) of a specific element.
struct XArrayCoorOpConversion
    : public FIROpAndTypeConversion<fir::cg::XArrayCoorOp> {
  using FIROpAndTypeConversion::FIROpAndTypeConversion;

  mlir::LogicalResult
  doRewrite(fir::cg::XArrayCoorOp coor, mlir::Type ty, OperandTy operands,
            mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = coor.getLoc();
    auto rank = coor.getRank();
    assert(coor.indices().size() == rank);
    assert(coor.shape().empty() || coor.shape().size() == rank);
    assert(coor.shift().empty() || coor.shift().size() == rank);
    assert(coor.slice().empty() || coor.slice().size() == 3 * rank);
    auto indexOps = coor.indices().begin();
    auto shapeOps = coor.shape().begin();
    auto shiftOps = coor.shift().begin();
    auto sliceOps = coor.slice().begin();
    auto idxTy = lowerTy().indexType();
    mlir::Value one = genConstantIndex(loc, idxTy, rewriter, 1);
    auto prevExt = one;
    mlir::Value zero = genConstantIndex(loc, idxTy, rewriter, 0);
    auto off = zero;
    const bool isShifted = !coor.shift().empty();
    const bool isSliced = !coor.slice().empty();
    const bool baseIsBoxed = coor.memref().getType().isa<fir::BoxType>();

    // For each dimension of the array, generate the offset calculation, off.
    for (unsigned i = 0; i < rank;
         ++i, ++indexOps, ++shapeOps, ++shiftOps, sliceOps += 3) {
      auto index = integerCast(loc, rewriter, idxTy, *indexOps);
      auto lb = isShifted ? integerCast(loc, rewriter, idxTy, *shiftOps) : one;
      auto step = one;
      auto normalSlice = isSliced;
      // Compute zero based index in dimension i of the element, applying
      // potential triplets and lower bounds.
      if (isSliced) {
        mlir::Value ub = *(sliceOps + 1);
        normalSlice = !mlir::isa_and_nonnull<fir::UndefOp>(ub.getDefiningOp());
        if (normalSlice)
          step = integerCast(loc, rewriter, idxTy, *(sliceOps + 2));
      }
      auto idx = rewriter.create<mlir::LLVM::SubOp>(loc, idxTy, index, lb);
      mlir::Value diff =
          rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, idx, step);
      if (normalSlice) {
        auto sliceLb = integerCast(loc, rewriter, idxTy, *sliceOps);
        auto adj = rewriter.create<mlir::LLVM::SubOp>(loc, idxTy, sliceLb, lb);
        diff = rewriter.create<mlir::LLVM::AddOp>(loc, idxTy, diff, adj);
      }
      // Update the offset given the stride and the zero based index `diff`
      // that was just computed.
      if (baseIsBoxed) {
        // Use stride in bytes from the descriptor.
        auto stride = loadStrideFromBox(loc, operands[0], i, rewriter);
        auto sc = rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, diff, stride);
        off = rewriter.create<mlir::LLVM::AddOp>(loc, idxTy, sc, off);
      } else {
        // Use stride computed at last iteration.
        auto sc = rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, diff, prevExt);
        off = rewriter.create<mlir::LLVM::AddOp>(loc, idxTy, sc, off);
        // Compute next stride assuming contiguity of the base array
        // (in element number).
        auto nextExt = integerCast(loc, rewriter, idxTy, *shapeOps);
        prevExt =
            rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, prevExt, nextExt);
      }
    }

    // Add computed offset to the base address.
    if (baseIsBoxed) {
      // Working with byte offsets. The base address is read from the fir.box.
      // and need to be casted to i8* to do the pointer arithmetic.
      auto baseTy = getBaseAddrTypeFromBox(operands[0].getType());
      auto base = loadBaseAddrFromBox(loc, baseTy, operands[0], rewriter);
      auto voidPtrTy = getVoidPtrType(coor.getContext());
      base = rewriter.create<mlir::LLVM::BitcastOp>(loc, voidPtrTy, base);
      llvm::SmallVector<mlir::Value> args{base, off};
      auto addr = rewriter.create<mlir::LLVM::GEPOp>(loc, voidPtrTy, args);
      if (coor.subcomponent().empty()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(coor, baseTy, addr);
      } else {
        auto casted = rewriter.create<mlir::LLVM::BitcastOp>(loc, baseTy, addr);
        llvm::SmallVector<mlir::Value> args = {casted, zero};
        if (!coor.lenParams().empty()) {
          // If type parameters are present, then we don't want to use a GEPOp
          // as below, as the LLVM struct type cannot be statically defined.
          TODO(loc, "derived type with type parameters");
        }
        // FIXME: array offset subcomponents must be converted to LLVM's
        // row-major layout here.
        for (auto i = coor.subcomponentOffset(); i != coor.indicesOffset(); ++i)
          args.push_back(operands[i]);
        rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(coor, baseTy, args);
      }
      return success();
    }

    // The array was not boxed, so it must be contiguous. off is therefore an
    // element offset and the base type is kept in the GEP unless the element
    // type size is itself dynamic.
    mlir::Value base;
    if (coor.subcomponent().empty()) {
      // No subcomponent.
      if (!coor.lenParams().empty()) {
        // Type parameters. Adjust element size explicitly.
        auto eleTy = fir::dyn_cast_ptrEleTy(coor.getType());
        assert(eleTy && "result must be a refence-like type");
        if (fir::characterWithDynamicLen(eleTy)) {
          assert(coor.lenParams().size() == 1);
          auto bitsInChar = lowerTy().getKindMap().getCharacterBitsize(
              eleTy.cast<fir::CharacterType>().getFKind());
          auto scaling = genConstantIndex(loc, idxTy, rewriter, bitsInChar / 8);
          auto scaledBySize =
              rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, off, scaling);
          auto length = integerCast(loc, rewriter, idxTy,
                                    operands[coor.lenParamsOffset()]);
          off = rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, scaledBySize,
                                                   length);
        } else {
          TODO(loc, "compute size of derived type with type parameters");
        }
      }
      // Cast the base address to a pointer to T.
      base = rewriter.create<mlir::LLVM::BitcastOp>(loc, ty, operands[0]);
    } else {
      // operands[0] must have a pointer type. For subcomponent slicing, we
      // want to cast away the array type and have a plain struct type.
      auto ty0 = operands[0].getType();
      auto ptrTy = ty0.dyn_cast<mlir::LLVM::LLVMPointerType>();
      assert(ptrTy && "expected pointer type");
      auto eleTy = ptrTy.getElementType();
      while (auto arrTy = eleTy.dyn_cast<mlir::LLVM::LLVMArrayType>())
        eleTy = arrTy.getElementType();
      auto newTy = mlir::LLVM::LLVMPointerType::get(eleTy);
      base = rewriter.create<mlir::LLVM::BitcastOp>(loc, newTy, operands[0]);
    }
    SmallVector<mlir::Value> args = {base, off};
    args.append(coor.subcomponent().begin(), coor.subcomponent().end());
    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(coor, ty, args);
    return success();
  }
};

/// Convert to (memory) reference to a reference to a subobject.
/// The coordinate_of op is a Swiss army knife operation that can be used on
/// (memory) references to records, arrays, complex, etc. as well as boxes.
/// With unboxed arrays, there is the restriction that the array have a static
/// shape in all but the last column.
struct CoordinateOpConversion
    : public FIROpAndTypeConversion<fir::CoordinateOp> {
  using FIROpAndTypeConversion::FIROpAndTypeConversion;

  mlir::LogicalResult
  doRewrite(fir::CoordinateOp coor, mlir::Type ty, OperandTy operands,
            mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = coor.getLoc();
    auto c0 = genConstantIndex(loc, lowerTy().indexType(), rewriter, 0);
    mlir::Value base = operands[0];
    auto firTy = coor.getBaseType();
    mlir::Type cpnTy = fir::dyn_cast_ptrOrBoxEleTy(firTy);
    assert(cpnTy && "not a reference type");
    bool columnIsDeferred = false;
    bool hasSubdimension = hasSubDimensions(cpnTy);

    // if argument 0 is complex, get the real or imaginary part
    if (fir::isa_complex(cpnTy)) {
      SmallVector<mlir::Value> offs = {c0};
      offs.append(std::next(operands.begin()), operands.end());
      mlir::Value gep = genGEP(loc, ty, rewriter, base, offs);
      rewriter.replaceOp(coor, gep);
      return success();
    }

    // if argument 0 is boxed, get the base pointer from the box
    if (auto boxTy = firTy.dyn_cast<fir::BoxType>()) {
      // Special case:
      //   %lenp = len_param_index foo, type<T(foo:i32)...>
      //   %addr = coordinate_of %box, %lenp
      if (coor.getNumOperands() == 2) {
        auto coorPtr = *coor.coor().begin();
        auto *s = coorPtr.getDefiningOp();
        if (isa_and_nonnull<fir::LenParamIndexOp>(s)) {
          mlir::Value lenParam = operands[1]; // byte offset
          auto bc =
              rewriter.create<mlir::LLVM::BitcastOp>(loc, voidPtrTy(), base);
          auto gep = genGEP(loc, ty, rewriter, bc, lenParam);
          rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(coor, ty, gep);
          return success();
        }
      }

      // General case:
      //   %box = ... : !fix.box<!fir.array<?xU>>
      //   %idx = ... : index
      //   %addr = coodinate_of %box, %idx : !fir.ref<U>
      auto baseTy = getBaseAddrTypeFromBox(base.getType());
      base = loadBaseAddrFromBox(loc, baseTy, base, rewriter);
      mlir::Value addr = base;
      auto cty = cpnTy;
      auto voidPtrTy = getVoidPtrType(coor.getContext());
      for (unsigned i = 1, last = operands.size(); i < last; ++i) {
        if (auto arrTy = cty.dyn_cast<fir::SequenceType>()) {
          // Applies byte strides from the box. Ignore lower bound from box
          // since fir.coordinate_of indexes are zero based. Lowering takes care
          // of lower bound aspects. This both accounts for dynamically sized
          // types and non contiguous arrays.
          auto idxTy = lowerTy().indexType();
          mlir::Value off = genConstantIndex(loc, idxTy, rewriter, 0);
          for (unsigned index = i, lastIndex = i + arrTy.getDimension();
               index < lastIndex; ++index) {
            auto stride =
                loadStrideFromBox(loc, operands[0], index - i, rewriter);
            auto sc = rewriter.create<mlir::LLVM::MulOp>(
                loc, idxTy, operands[index], stride);
            off = rewriter.create<mlir::LLVM::AddOp>(loc, idxTy, sc, off);
          }
          auto voidPtrBase =
              rewriter.create<mlir::LLVM::BitcastOp>(loc, voidPtrTy, addr);
          SmallVector<mlir::Value> args{voidPtrBase, off};
          addr = rewriter.create<mlir::LLVM::GEPOp>(loc, voidPtrTy, args);
          i += arrTy.getDimension() - 1;
          cty = arrTy.getEleTy();
        } else if (auto seqTy = cty.dyn_cast<fir::RecordType>()) {
          auto seqRefTy =
              mlir::LLVM::LLVMPointerType::get(lowerTy().convertType(seqTy));
          auto nxtOpnd = operands[i];
          auto memObj =
              rewriter.create<mlir::LLVM::BitcastOp>(loc, seqRefTy, addr);
          llvm::SmallVector<mlir::Value> args = {memObj, c0, nxtOpnd};
          cty = seqTy.getType(getFieldNumber(seqTy, nxtOpnd));
          auto llvmCty = lowerTy().convertType(cty);
          auto gep = rewriter.create<mlir::LLVM::GEPOp>(
              loc, mlir::LLVM::LLVMPointerType::get(llvmCty), args);
          addr = rewriter.create<mlir::LLVM::BitcastOp>(loc, voidPtrTy, gep);
        } else {
          fir::emitFatalError(loc, "unexpected type in coordinate_of");
        }
      }
      rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(coor, ty, addr);
      return success();
    }
    if (!hasSubdimension)
      columnIsDeferred = true;

    if (!validCoordinate(cpnTy, operands.drop_front(1)))
      return mlir::emitError(loc, "coordinate has incorrect dimension");

    // if arrays has known shape
    const bool hasKnownShape =
        arraysHaveKnownShape(cpnTy, operands.drop_front(1));

    // If only the column is `?`, then we can simply place the column value in
    // the 0-th GEP position.
    if (auto arrTy = cpnTy.dyn_cast<fir::SequenceType>()) {
      if (!hasKnownShape) {
        const auto sz = arrTy.getDimension();
        if (arraysHaveKnownShape(arrTy.getEleTy(),
                                 operands.drop_front(1 + sz))) {
          auto shape = arrTy.getShape();
          bool allConst = true;
          for (std::remove_const_t<decltype(sz)> i = 0; i < sz - 1; ++i)
            if (shape[i] < 0) {
              allConst = false;
              break;
            }
          if (allConst)
            columnIsDeferred = true;
        }
      }
    }

    if (fir::hasDynamicSize(fir::unwrapSequenceType(cpnTy)))
      return mlir::emitError(
          loc, "fir.coordinate_of with a dynamic element size is unsupported");

    if (hasKnownShape || columnIsDeferred) {
      SmallVector<mlir::Value> offs;
      if (hasKnownShape && hasSubdimension)
        offs.push_back(c0);
      const auto sz = operands.size();
      Optional<int> dims;
      SmallVector<mlir::Value> arrIdx;
      for (std::remove_const_t<decltype(sz)> i = 1; i < sz; ++i) {
        auto nxtOpnd = operands[i];

        if (!cpnTy)
          return mlir::emitError(loc, "invalid coordinate/check failed");

        // check if the i-th coordinate relates to an array
        if (dims.hasValue()) {
          arrIdx.push_back(nxtOpnd);
          int dimsLeft = *dims;
          if (dimsLeft > 1) {
            dims = dimsLeft - 1;
            continue;
          }
          cpnTy = cpnTy.cast<fir::SequenceType>().getEleTy();
          // append array range in reverse (FIR arrays are column-major)
          offs.append(arrIdx.rbegin(), arrIdx.rend());
          arrIdx.clear();
          dims.reset();
          continue;
        }
        if (auto arrTy = cpnTy.dyn_cast<fir::SequenceType>()) {
          int d = arrTy.getDimension() - 1;
          if (d > 0) {
            dims = d;
            arrIdx.push_back(nxtOpnd);
            continue;
          }
          cpnTy = cpnTy.cast<fir::SequenceType>().getEleTy();
          offs.push_back(nxtOpnd);
          continue;
        }

        // check if the i-th coordinate relates to a field
        if (auto strTy = cpnTy.dyn_cast<fir::RecordType>()) {
          cpnTy = strTy.getType(getFieldNumber(strTy, nxtOpnd));
        } else if (auto strTy = cpnTy.dyn_cast<mlir::TupleType>()) {
          cpnTy = strTy.getType(getIntValue(nxtOpnd));
        } else {
          cpnTy = nullptr;
        }
        offs.push_back(nxtOpnd);
      }
      if (dims.hasValue())
        offs.append(arrIdx.rbegin(), arrIdx.rend());
      mlir::Value retval = genGEP(loc, ty, rewriter, base, offs);
      rewriter.replaceOp(coor, retval);
      return success();
    }
    return mlir::emitError(
        loc, "fir.coordinate_of base operand has unsupported type");
  }

  unsigned getFieldNumber(fir::RecordType ty, mlir::Value op) const {
    return fir::hasDynamicSize(ty)
               ? op.getDefiningOp()
                     ->getAttrOfType<mlir::IntegerAttr>("field")
                     .getInt()
               : getIntValue(op);
  }

  bool hasSubDimensions(mlir::Type type) const {
    return type.isa<fir::SequenceType>() || type.isa<fir::RecordType>() ||
           type.isa<mlir::TupleType>();
  }

  /// Walk the abstract memory layout and determine if the path traverses any
  /// array types with unknown shape. Return true iff all the array types have a
  /// constant shape along the path.
  bool arraysHaveKnownShape(mlir::Type type, OperandTy coors) const {
    const auto sz = coors.size();
    std::remove_const_t<decltype(sz)> i = 0;
    for (; i < sz; ++i) {
      auto nxtOpnd = coors[i];
      if (auto arrTy = type.dyn_cast<fir::SequenceType>()) {
        if (fir::sequenceWithNonConstantShape(arrTy))
          return false;
        i += arrTy.getDimension() - 1;
        type = arrTy.getEleTy();
      } else if (auto strTy = type.dyn_cast<fir::RecordType>()) {
        type = strTy.getType(getFieldNumber(strTy, nxtOpnd));
      } else if (auto strTy = type.dyn_cast<mlir::TupleType>()) {
        type = strTy.getType(getIntValue(nxtOpnd));
      } else {
        return true;
      }
    }
    return true;
  }

  bool validCoordinate(mlir::Type type, OperandTy coors) const {
    const auto sz = coors.size();
    std::remove_const_t<decltype(sz)> i = 0;
    bool subEle = false;
    bool ptrEle = false;
    for (; i < sz; ++i) {
      auto nxtOpnd = coors[i];
      if (auto arrTy = type.dyn_cast<fir::SequenceType>()) {
        subEle = true;
        i += arrTy.getDimension() - 1;
        type = arrTy.getEleTy();
      } else if (auto strTy = type.dyn_cast<fir::RecordType>()) {
        subEle = true;
        type = strTy.getType(getFieldNumber(strTy, nxtOpnd));
      } else if (auto strTy = type.dyn_cast<mlir::TupleType>()) {
        subEle = true;
        type = strTy.getType(getIntValue(nxtOpnd));
      } else {
        ptrEle = true;
      }
    }
    if (ptrEle)
      return (!subEle) && (sz == 1);
    return subEle && (i >= sz);
  }

  /// return true if all `Value`s in `operands` are not `FieldIndexOp`s
  static bool noFieldIndexOps(mlir::Operation::operand_range operands) {
    for (auto opnd : operands) {
      if (auto *defop = opnd.getDefiningOp())
        if (dyn_cast<fir::FieldIndexOp>(defop))
          return false;
    }
    return true;
  }

  SmallVector<mlir::Value> arguments(OperandTy vec, unsigned s,
                                     unsigned e) const {
    return {vec.begin() + s, vec.begin() + e};
  }

  int64_t getIntValue(mlir::Value val) const {
    if (val) {
      if (auto *defop = val.getDefiningOp()) {
        if (auto constOp = dyn_cast<mlir::arith::ConstantIntOp>(defop))
          return constOp.value();
        if (auto llConstOp = dyn_cast<mlir::LLVM::ConstantOp>(defop))
          if (auto attr = llConstOp.value().dyn_cast<mlir::IntegerAttr>())
            return attr.getValue().getSExtValue();
      }
      fir::emitFatalError(val.getLoc(), "must be a constant");
    }
    llvm_unreachable("must not be null value");
  }
}; // namespace

/// convert a field index to a runtime function that computes the byte offset
/// of the dynamic field
struct FieldIndexOpConversion : public FIROpConversion<fir::FieldIndexOp> {
  using FIROpConversion::FIROpConversion;

  // NB: most field references should be resolved by this point
  mlir::LogicalResult
  matchAndRewrite(fir::FieldIndexOp field, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto recTy = field.on_type().cast<fir::RecordType>();
    auto index = recTy.getFieldIndex(field.field_id());
    if (!fir::hasDynamicSize(recTy)) {
      // Derived type has compile-time constant layout. Returns index of the
      // component type in the parent type (to be used in GEP).
      rewriter.replaceOp(field, mlir::ValueRange{genConstantOffset(
                                    field.getLoc(), rewriter, index)});
    } else {
      // Call the compiler generated function to determine the byte offset of
      // the field at runtime. This returns a non-constant.
      auto symAttr = mlir::SymbolRefAttr::get(
          field.getContext(), methodName(recTy, field.field_id()));
      auto callAttr = rewriter.getNamedAttr("callee", symAttr);
      auto fieldAttr = rewriter.getNamedAttr(
          "field", mlir::IntegerAttr::get(lowerTy().indexType(), index));
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
          field, lowerTy().offsetType(), operands,
          llvm::ArrayRef<mlir::NamedAttribute>{callAttr, fieldAttr});
    }
    return success();
  }

  // constructing the name of the method
  inline static std::string methodName(fir::RecordType recTy,
                                       llvm::StringRef field) {
    return recTy.getName().str() + "P." + field.str() + ".offset";
  }
};

struct LenParamIndexOpConversion
    : public FIROpConversion<fir::LenParamIndexOp> {
  using FIROpConversion::FIROpConversion;

  // FIXME: this should be specialized by the runtime target
  mlir::LogicalResult
  matchAndRewrite(fir::LenParamIndexOp lenp, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ity = lowerTy().indexType();
    auto onty = lenp.getOnType();
    // size of portable descriptor
    const unsigned boxsize = 24; // FIXME
    unsigned offset = boxsize;
    // add the size of the rows of triples
    if (auto arr = onty.dyn_cast<fir::SequenceType>())
      offset += 3 * arr.getDimension();

    // advance over some addendum fields
    const unsigned addendumOffset{sizeof(void *) + sizeof(uint64_t)};
    offset += addendumOffset;
    // add the offset into the LENs
    offset += 0; // FIXME
    auto attr = rewriter.getI64IntegerAttr(offset);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(lenp, ity, attr);
    return success();
  }
};

/// lower the fir.end operation to a null (erasing it)
struct FirEndOpConversion : public FIROpConversion<fir::FirEndOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::FirEndOp op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, {});
    return success();
  }
};

/// lower a type descriptor to a global constant
struct GenTypeDescOpConversion : public FIROpConversion<fir::GenTypeDescOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::GenTypeDescOp gentypedesc, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = gentypedesc.getLoc();
    auto inTy = gentypedesc.getInType();
    auto name = consName(rewriter, inTy);
    auto gty = convertType(inTy);
    auto pty = mlir::LLVM::LLVMPointerType::get(gty);
    auto module = gentypedesc->getParentOfType<mlir::ModuleOp>();
    createGlobal(loc, module, name, gty, rewriter);
    rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(gentypedesc, pty,
                                                         name);
    return success();
  }

  std::string consName(mlir::ConversionPatternRewriter &rewriter,
                       mlir::Type type) const {
    if (auto d = type.dyn_cast<fir::RecordType>()) {
      auto name = d.getName();
      auto pair = fir::NameUniquer::deconstruct(name);
      return fir::NameUniquer::doTypeDescriptor(
          pair.second.modules, pair.second.host, pair.second.name,
          pair.second.kinds);
    }
    llvm_unreachable("no name found");
  }
};

struct GlobalLenOpConversion : public FIROpConversion<fir::GlobalLenOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::GlobalLenOp globalLen, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    TODO(globalLen.getLoc(), "GlobalLenOp codegen");
    return success();
  }
};

/// Lower `fir.has_value` operation to `llvm.return` operation.
struct HasValueOpConversion : public FIROpConversion<fir::HasValueOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::HasValueOp op, OperandTy operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, operands);
    return success();
  }
};

/// Lower `fir.global` operation to `llvm.global` operation.
/// `fir.insert_on_range` operations are replaced with constant dense attribute
/// if they are applied on the full range.
struct GlobalOpConversion : public FIROpConversion<fir::GlobalOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::GlobalOp global, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto tyAttr = convertType(global.getType());
    if (global.getType().isa<fir::BoxType>())
      tyAttr = tyAttr.cast<mlir::LLVM::LLVMPointerType>().getElementType();
    auto loc = global.getLoc();
    mlir::Attribute initAttr{};
    if (global.initVal())
      initAttr = global.initVal().getValue();
    auto linkage = convertLinkage(global.linkName());
    auto isConst = global.constant().hasValue();
    auto g = rewriter.create<mlir::LLVM::GlobalOp>(
        loc, tyAttr, isConst, linkage, global.sym_name(), initAttr);
    auto &gr = g.getInitializerRegion();
    rewriter.inlineRegionBefore(global.region(), gr, gr.end());
    if (!gr.empty()) {
      // Replace insert_on_range with a constant dense attribute if the
      // initialization is on the full range.
      auto insertOnRangeOps = gr.front().getOps<fir::InsertOnRangeOp>();
      for (auto insertOp : insertOnRangeOps) {
        if (isFullRange(insertOp.coor(), insertOp.getType())) {
          auto seqTyAttr = convertType(insertOp.getType());
          auto *op = insertOp.val().getDefiningOp();
          auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(op);
          if (!constant) {
            auto convertOp = mlir::dyn_cast<fir::ConvertOp>(op);
            if (!convertOp)
              continue;
            constant = cast<mlir::arith::ConstantOp>(
                convertOp.value().getDefiningOp());
          }
          mlir::Type vecType = mlir::VectorType::get(
              insertOp.getType().getShape(), constant.getType());
          auto denseAttr = mlir::DenseElementsAttr::get(
              vecType.cast<ShapedType>(), constant.value());
          rewriter.setInsertionPointAfter(insertOp);
          rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
              insertOp, seqTyAttr, denseAttr);
        }
      }
    }
    rewriter.eraseOp(global);
    return success();
  }

  bool isFullRange(mlir::ArrayAttr indexes, fir::SequenceType seqTy) const {
    auto extents = seqTy.getShape();
    if (indexes.size() / 2 != extents.size())
      return false;
    for (unsigned i = 0; i < indexes.size(); i += 2) {
      if (indexes[i].cast<IntegerAttr>().getInt() != 0)
        return false;
      if (indexes[i + 1].cast<IntegerAttr>().getInt() != extents[i / 2] - 1)
        return false;
    }
    return true;
  }

  // TODO: String comparison should be avoided. Replace linkName with an
  // enumeration.
  mlir::LLVM::Linkage convertLinkage(Optional<StringRef> optLinkage) const {
    if (optLinkage.hasValue()) {
      auto name = optLinkage.getValue();
      if (name == "internal")
        return mlir::LLVM::Linkage::Internal;
      if (name == "linkonce")
        return mlir::LLVM::Linkage::Linkonce;
      if (name == "common")
        return mlir::LLVM::Linkage::Common;
      if (name == "weak")
        return mlir::LLVM::Linkage::Weak;
    }
    return mlir::LLVM::Linkage::External;
  }
};

// convert to LLVM IR dialect `load`
struct LoadOpConversion : public FIROpConversion<fir::LoadOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::LoadOp load, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // fir.box is a special case because it is considered as an ssa values in
    // fir, but it is lowered as a pointer to a descriptor. So fir.ref<fir.box>
    // and fir.box end up being the same llvm types and loading a fir.ref<box>
    // is actually a no op in LLVM.
    if (load.getType().isa<fir::BoxType>()) {
      rewriter.replaceOp(load, operands[0]);
    } else {
      auto ty = convertType(load.getType());
      auto at = load->getAttrs();
      rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(load, ty, operands, at);
    }
    return success();
  }
};

// FIXME: how do we want to enforce this in LLVM-IR? Can we manipulate the fast
// math flags?
struct NoReassocOpConversion : public FIROpConversion<fir::NoReassocOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::NoReassocOp noreassoc, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(noreassoc, operands[0]);
    return success();
  }
};

void genCondBrOp(mlir::Location loc, mlir::Value cmp, mlir::Block *dest,
                 Optional<OperandTy> destOps,
                 mlir::ConversionPatternRewriter &rewriter,
                 mlir::Block *newBlock) {
  if (destOps.hasValue())
    rewriter.create<mlir::LLVM::CondBrOp>(loc, cmp, dest, destOps.getValue(),
                                          newBlock, mlir::ValueRange());
  else
    rewriter.create<mlir::LLVM::CondBrOp>(loc, cmp, dest, newBlock);
}

template <typename A, typename B>
void genBrOp(A caseOp, mlir::Block *dest, Optional<B> destOps,
             mlir::ConversionPatternRewriter &rewriter) {
  if (destOps.hasValue())
    rewriter.replaceOpWithNewOp<mlir::LLVM::BrOp>(caseOp, destOps.getValue(),
                                                  dest);
  else
    rewriter.replaceOpWithNewOp<mlir::LLVM::BrOp>(caseOp, llvm::None, dest);
}

void genCaseLadderStep(mlir::Location loc, mlir::Value cmp, mlir::Block *dest,
                       Optional<OperandTy> destOps,
                       mlir::ConversionPatternRewriter &rewriter) {
  auto *thisBlock = rewriter.getInsertionBlock();
  auto *newBlock = createBlock(rewriter, dest);
  rewriter.setInsertionPointToEnd(thisBlock);
  genCondBrOp(loc, cmp, dest, destOps, rewriter, newBlock);
  rewriter.setInsertionPointToEnd(newBlock);
}

/// Conversion of `fir.select_case`
///
/// TODO: lowering of CHARACTER type cases
struct SelectCaseOpConversion : public FIROpConversion<fir::SelectCaseOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::SelectCaseOp caseOp, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    const auto conds = caseOp.getNumConditions();
    auto attrName = fir::SelectCaseOp::getCasesAttr();
    auto cases = caseOp->getAttrOfType<mlir::ArrayAttr>(attrName).getValue();
    // Type can be CHARACTER, INTEGER, or LOGICAL (C1145)
    LLVM_ATTRIBUTE_UNUSED auto ty = caseOp.getSelector().getType();
    auto selector = caseOp.getSelector(operands);
    auto loc = caseOp.getLoc();
    assert(conds > 0 && "fir.selectcase must have cases");
    for (std::remove_const_t<decltype(conds)> t = 0; t != conds; ++t) {
      mlir::Block *dest = caseOp.getSuccessor(t);
      auto destOps = caseOp.getSuccessorOperands(operands, t);
      auto cmpOps = *caseOp.getCompareOperands(operands, t);
      auto caseArg = *cmpOps.begin();
      auto &attr = cases[t];
      if (attr.isa<fir::PointIntervalAttr>()) {
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::eq, selector, caseArg);
        genCaseLadderStep(loc, cmp, dest, destOps, rewriter);
        continue;
      }
      if (attr.isa<fir::LowerBoundAttr>()) {
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::sle, caseArg, selector);
        genCaseLadderStep(loc, cmp, dest, destOps, rewriter);
        continue;
      }
      if (attr.isa<fir::UpperBoundAttr>()) {
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::sle, selector, caseArg);
        genCaseLadderStep(loc, cmp, dest, destOps, rewriter);
        continue;
      }
      if (attr.isa<fir::ClosedIntervalAttr>()) {
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::sle, caseArg, selector);
        auto *thisBlock = rewriter.getInsertionBlock();
        auto *newBlock1 = createBlock(rewriter, dest);
        auto *newBlock2 = createBlock(rewriter, dest);
        rewriter.setInsertionPointToEnd(thisBlock);
        rewriter.create<mlir::LLVM::CondBrOp>(loc, cmp, newBlock1, newBlock2);
        rewriter.setInsertionPointToEnd(newBlock1);
        auto caseArg0 = *(cmpOps.begin() + 1);
        auto cmp0 = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::sle, selector, caseArg0);
        genCondBrOp(loc, cmp0, dest, destOps, rewriter, newBlock2);
        rewriter.setInsertionPointToEnd(newBlock2);
        continue;
      }
      assert(attr.isa<mlir::UnitAttr>());
      assert((t + 1 == conds) && "unit must be last");
      genBrOp(caseOp, dest, destOps, rewriter);
    }
    return success();
  }
};

template <typename OP>
void selectMatchAndRewrite(fir::LLVMTypeConverter &lowering, OP select,
                           OperandTy operands,
                           mlir::ConversionPatternRewriter &rewriter) {
  unsigned conds = select.getNumConditions();
  auto cases = select.getCases().getValue();
  mlir::Value selector = select.selector();
  auto loc = select.getLoc();
  assert(conds > 0 && "select must have cases");

  llvm::SmallVector<mlir::Block *> destinations;
  llvm::SmallVector<mlir::ValueRange> destinationsOperands;
  mlir::Block *defaultDestination;
  mlir::ValueRange defaultOperands;
  llvm::SmallVector<int32_t> caseValues;

  for (unsigned t = 0; t != conds; ++t) {
    mlir::Block *dest = select.getSuccessor(t);
    auto destOps = select.getSuccessorOperands(operands, t);
    const mlir::Attribute &attr = cases[t];
    if (auto intAttr = attr.template dyn_cast<mlir::IntegerAttr>()) {
      destinations.push_back(dest);
      destinationsOperands.push_back(destOps.hasValue() ? *destOps
                                                        : ValueRange());
      caseValues.push_back(intAttr.getInt());
      continue;
    }
    assert(attr.template dyn_cast_or_null<mlir::UnitAttr>());
    assert((t + 1 == conds) && "unit must be last");
    defaultDestination = dest;
    defaultOperands = destOps.hasValue() ? *destOps : ValueRange();
  }

  // LLVM::SwitchOp takes a i32 type for the selector.
  if (select.getSelector().getType() != rewriter.getI32Type())
    selector =
        rewriter.create<LLVM::TruncOp>(loc, rewriter.getI32Type(), selector);

  rewriter.replaceOpWithNewOp<mlir::LLVM::SwitchOp>(
      select, selector,
      /*defaultDestination=*/defaultDestination,
      /*defaultOperands=*/defaultOperands,
      /*caseValues=*/caseValues,
      /*caseDestinations=*/destinations,
      /*caseOperands=*/destinationsOperands,
      /*branchWeights=*/ArrayRef<int32_t>());
}

/// conversion of fir::SelectOp to an if-then-else ladder
struct SelectOpConversion : public FIROpConversion<fir::SelectOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::SelectOp op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    selectMatchAndRewrite<fir::SelectOp>(lowerTy(), op, operands, rewriter);
    return success();
  }
};

/// conversion of fir::SelectRankOp to an if-then-else ladder
struct SelectRankOpConversion : public FIROpConversion<fir::SelectRankOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::SelectRankOp op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    selectMatchAndRewrite<fir::SelectRankOp>(lowerTy(), op, operands, rewriter);
    return success();
  }
};

struct SelectTypeOpConversion : public FIROpConversion<fir::SelectTypeOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::SelectTypeOp select, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::emitError(select.getLoc(),
                    "fir.select_type should have already been converted");
    return failure();
  }
};

// convert to LLVM IR dialect `store`
struct StoreOpConversion : public FIROpConversion<fir::StoreOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::StoreOp store, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (store.value().getType().isa<fir::BoxType>()) {
      // fir.box value is actually in memory, load it first before storing it.
      auto loc = store.getLoc();
      auto boxPtrTy = operands[0].getType();
      auto val = rewriter.create<mlir::LLVM::LoadOp>(
          loc, boxPtrTy.cast<mlir::LLVM::LLVMPointerType>().getElementType(),
          operands[0]);
      rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(store, val, operands[1]);
    } else {
      rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(store, operands[0],
                                                       operands[1]);
    }
    return success();
  }
};

// cons an extractvalue on a tuple value, returning value at element `x`
mlir::LLVM::ExtractValueOp
genExtractValueWithIndex(mlir::Location loc, mlir::Value tuple, mlir::Type ty,
                         mlir::ConversionPatternRewriter &rewriter,
                         mlir::MLIRContext *ctx, int x) {
  auto cx = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(x));
  auto xty = ty.cast<mlir::LLVM::LLVMStructType>().getBody()[x];
  return rewriter.create<mlir::LLVM::ExtractValueOp>(loc, xty, tuple, cx);
}

// unbox a CHARACTER box value, yielding its components
struct UnboxCharOpConversion : public FIROpConversion<fir::UnboxCharOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::UnboxCharOp unboxchar, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *ctx = unboxchar.getContext();
    auto lenTy = convertType(unboxchar.getType(1));
    auto loc = unboxchar.getLoc();
    auto tuple = operands[0];
    auto ty = tuple.getType();
    mlir::Value ptr =
        genExtractValueWithIndex(loc, tuple, ty, rewriter, ctx, 0);
    auto len1 = genExtractValueWithIndex(loc, tuple, ty, rewriter, ctx, 1);
    auto len = integerCast(loc, rewriter, lenTy, len1);
    rewriter.replaceOp(unboxchar, ArrayRef<mlir::Value>{ptr, len});
    return success();
  }
};

// unbox a procedure box value, yielding its components
struct UnboxProcOpConversion : public FIROpConversion<fir::UnboxProcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::UnboxProcOp unboxproc, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *ctx = unboxproc.getContext();
    auto loc = unboxproc.getLoc();
    auto tuple = operands[0];
    auto ty = tuple.getType();
    mlir::Value ptr =
        genExtractValueWithIndex(loc, tuple, ty, rewriter, ctx, 0);
    mlir::Value host =
        genExtractValueWithIndex(loc, tuple, ty, rewriter, ctx, 1);
    llvm::SmallVector<mlir::Value> repls{ptr, host};
    rewriter.replaceOp(unboxproc, repls);
    return success();
  }
};

// convert to LLVM IR dialect `undef`
struct UndefOpConversion : public FIROpConversion<fir::UndefOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::UndefOp undef, OperandTy,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::UndefOp>(
        undef, convertType(undef.getType()));
    return success();
  }
};

struct ZeroOpConversion : public FIROpConversion<fir::ZeroOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::ZeroOp zero, OperandTy,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ty = convertType(zero.getType());
    if (ty.isa<mlir::LLVM::LLVMPointerType>()) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::NullOp>(zero, ty);
    } else if (ty.isa<mlir::IntegerType>()) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(
          zero, ty, mlir::IntegerAttr::get(zero.getType(), 0));
    } else if (mlir::LLVM::isCompatibleFloatingPointType(ty)) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(
          zero, ty, mlir::IntegerAttr::get(zero.getType(), 0.0));
    } else {
      // FIXME/TODO: how do we create a ConstantAggregateZero?
      rewriter.replaceOpWithNewOp<mlir::LLVM::UndefOp>(zero, ty);
    }
    return success();
  }
};

// convert to LLVM IR dialect `unreachable`
struct UnreachableOpConversion : public FIROpConversion<fir::UnreachableOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::UnreachableOp unreach, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::UnreachableOp>(unreach);
    return success();
  }
};

// Check if an argument is present.
struct IsPresentOpConversion : public FIROpConversion<fir::IsPresentOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::IsPresentOp isPresent, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto idxTy = lowerTy().indexType();
    auto loc = isPresent.getLoc();
    auto ptr = operands[0];
    if (isPresent.val().getType().isa<fir::BoxCharType>()) {
      auto structTy = ptr.getType().cast<mlir::LLVM::LLVMStructType>();
      assert(!structTy.isOpaque() && !structTy.getBody().empty());
      auto ty = structTy.getBody()[0];
      auto *ctx = isPresent.getContext();
      auto c0 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(0));
      ptr = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, ptr, c0);
    }
    auto c0 = genConstantIndex(isPresent.getLoc(), idxTy, rewriter, 0);
    auto addr = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, idxTy, ptr);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        isPresent, mlir::LLVM::ICmpPredicate::ne, addr, c0);
    return success();
  }
};

// Create value signaling an absent optional argument in a call.
struct AbsentOpConversion : public FIROpConversion<fir::AbsentOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::AbsentOp absent, OperandTy,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ty = convertType(absent.getType());
    auto loc = absent.getLoc();
    if (absent.getType().isa<fir::BoxCharType>()) {
      auto structTy = ty.cast<mlir::LLVM::LLVMStructType>();
      assert(!structTy.isOpaque() && !structTy.getBody().empty());
      auto undefStruct = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
      auto nullField =
          rewriter.create<mlir::LLVM::NullOp>(loc, structTy.getBody()[0]);
      auto *ctx = absent.getContext();
      auto c0 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(0));
      rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
          absent, ty, undefStruct, nullField, c0);
    } else {
      rewriter.replaceOpWithNewOp<mlir::LLVM::NullOp>(absent, ty);
    }
    return success();
  }
};

//
// Primitive operations on Complex types
//

/// Generate inline code for complex addition/subtraction
template <typename LLVMOP, typename OPTY>
mlir::LLVM::InsertValueOp complexSum(OPTY sumop, OperandTy opnds,
                                     mlir::ConversionPatternRewriter &rewriter,
                                     fir::LLVMTypeConverter &lowering) {
  auto a = opnds[0];
  auto b = opnds[1];
  auto loc = sumop.getLoc();
  auto ctx = sumop.getContext();
  auto c0 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(0));
  auto c1 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(1));
  auto eleTy = lowering.convertType(getComplexEleTy(sumop.getType()));
  auto ty = lowering.convertType(sumop.getType());
  auto x0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, a, c0);
  auto y0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, a, c1);
  auto x1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, b, c0);
  auto y1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, b, c1);
  auto rx = rewriter.create<LLVMOP>(loc, eleTy, x0, x1);
  auto ry = rewriter.create<LLVMOP>(loc, eleTy, y0, y1);
  auto r0 = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
  auto r1 = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, r0, rx, c0);
  return rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, r1, ry, c1);
}

struct AddcOpConversion : public FIROpConversion<fir::AddcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::AddcOp addc, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // given: (x + iy) * (x' + iy')
    // result: (x + x') + i(y + y')
    auto r =
        complexSum<mlir::LLVM::FAddOp>(addc, operands, rewriter, lowerTy());
    rewriter.replaceOp(addc, r.getResult());
    return success();
  }
};

struct SubcOpConversion : public FIROpConversion<fir::SubcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::SubcOp subc, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // given: (x + iy) * (x' + iy')
    // result: (x - x') + i(y - y')
    auto r =
        complexSum<mlir::LLVM::FSubOp>(subc, operands, rewriter, lowerTy());
    rewriter.replaceOp(subc, r.getResult());
    return success();
  }
};

/// Inlined complex multiply
struct MulcOpConversion : public FIROpConversion<fir::MulcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::MulcOp mulc, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // TODO: should this just call __muldc3 ?
    // given: (x + iy) * (x' + iy')
    // result: (xx'-yy')+i(xy'+yx')
    auto a = operands[0];
    auto b = operands[1];
    auto loc = mulc.getLoc();
    auto *ctx = mulc.getContext();
    auto c0 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(0));
    auto c1 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(1));
    auto eleTy = convertType(getComplexEleTy(mulc.getType()));
    auto ty = convertType(mulc.getType());
    auto x0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, a, c0);
    auto y0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, a, c1);
    auto x1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, b, c0);
    auto y1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, b, c1);
    auto xx = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, x0, x1);
    auto yx = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, y0, x1);
    auto xy = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, x0, y1);
    auto ri = rewriter.create<mlir::LLVM::FAddOp>(loc, eleTy, xy, yx);
    auto yy = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, y0, y1);
    auto rr = rewriter.create<mlir::LLVM::FSubOp>(loc, eleTy, xx, yy);
    auto ra = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    auto r1 = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, ra, rr, c0);
    auto r0 = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, r1, ri, c1);
    rewriter.replaceOp(mulc, r0.getResult());
    return success();
  }
};

/// Inlined complex division
struct DivcOpConversion : public FIROpConversion<fir::DivcOp> {
  using FIROpConversion::FIROpConversion;

  // Should this just call __divdc3? Just generate inline code for now.
  mlir::LogicalResult
  matchAndRewrite(fir::DivcOp divc, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // given: (x + iy) / (x' + iy')
    // result: ((xx'+yy')/d) + i((yx'-xy')/d) where d = x'x' + y'y'
    auto a = operands[0];
    auto b = operands[1];
    auto loc = divc.getLoc();
    auto *ctx = divc.getContext();
    auto c0 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(0));
    auto c1 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(1));
    auto eleTy = convertType(getComplexEleTy(divc.getType()));
    auto ty = convertType(divc.getType());
    auto x0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, a, c0);
    auto y0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, a, c1);
    auto x1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, b, c0);
    auto y1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, b, c1);
    auto xx = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, x0, x1);
    auto x1x1 = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, x1, x1);
    auto yx = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, y0, x1);
    auto xy = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, x0, y1);
    auto yy = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, y0, y1);
    auto y1y1 = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, y1, y1);
    auto d = rewriter.create<mlir::LLVM::FAddOp>(loc, eleTy, x1x1, y1y1);
    auto rrn = rewriter.create<mlir::LLVM::FAddOp>(loc, eleTy, xx, yy);
    auto rin = rewriter.create<mlir::LLVM::FSubOp>(loc, eleTy, yx, xy);
    auto rr = rewriter.create<mlir::LLVM::FDivOp>(loc, eleTy, rrn, d);
    auto ri = rewriter.create<mlir::LLVM::FDivOp>(loc, eleTy, rin, d);
    auto ra = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    auto r1 = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, ra, rr, c0);
    auto r0 = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, r1, ri, c1);
    rewriter.replaceOp(divc, r0.getResult());
    return success();
  }
};

/// Inlined complex negation
struct NegcOpConversion : public FIROpConversion<fir::NegcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::NegcOp neg, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // given: -(x + iy)
    // result: -x - iy
    auto *ctxt = neg.getContext();
    auto eleTy = convertType(getComplexEleTy(neg.getType()));
    auto ty = convertType(neg.getType());
    auto loc = neg.getLoc();
    auto &o0 = operands[0];
    auto c0 = mlir::ArrayAttr::get(ctxt, rewriter.getI32IntegerAttr(0));
    auto c1 = mlir::ArrayAttr::get(ctxt, rewriter.getI32IntegerAttr(1));
    auto rp = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, o0, c0);
    auto ip = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, o0, c1);
    auto nrp = rewriter.create<mlir::LLVM::FNegOp>(loc, eleTy, rp);
    auto nip = rewriter.create<mlir::LLVM::FNegOp>(loc, eleTy, ip);
    auto r = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, o0, nrp, c0);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(neg, ty, r, nip, c1);
    return success();
  }
};

template <typename OP>
struct MustBeDeadConversion : public FIROpConversion<OP> {
  explicit MustBeDeadConversion(mlir::MLIRContext *ctx,
                                fir::LLVMTypeConverter &lowering)
      : FIROpConversion<OP>(ctx, lowering) {}

  mlir::LogicalResult
  matchAndRewrite(OP op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    if (!op->getUses().empty())
      return mlir::emitError(op.getLoc(), "op must be dead");
    rewriter.eraseOp(op);
    return success();
  }
};

struct ShapeOpConversion : public MustBeDeadConversion<fir::ShapeOp> {
  using MustBeDeadConversion::MustBeDeadConversion;
};

struct ShapeShiftOpConversion : public MustBeDeadConversion<fir::ShapeShiftOp> {
  using MustBeDeadConversion::MustBeDeadConversion;
};

struct ShiftOpConversion : public MustBeDeadConversion<fir::ShiftOp> {
  using MustBeDeadConversion::MustBeDeadConversion;
};

struct SliceOpConversion : public MustBeDeadConversion<fir::SliceOp> {
  using MustBeDeadConversion::MustBeDeadConversion;
};

/// Convert FIR dialect to LLVM dialect
///
/// This pass lowers all FIR dialect operations to LLVM IR dialect.  An
/// MLIR pass is used to lower residual Std dialect to LLVM IR dialect.
class FIRToLLVMLowering : public fir::FIRToLLVMLoweringBase<FIRToLLVMLowering> {
public:
  mlir::ModuleOp getModule() { return getOperation(); }

  void runOnOperation() override final {
    auto *context = getModule().getContext();
    fir::LLVMTypeConverter typeConverter{getModule()};
    auto loc = mlir::UnknownLoc::get(context);
    mlir::OwningRewritePatternList pattern(context);
    pattern.insert<
        AbsentOpConversion, AddcOpConversion, AddrOfOpConversion,
        AllocaOpConversion, AllocMemOpConversion, BoxAddrOpConversion,
        BoxCharLenOpConversion, BoxDimsOpConversion, BoxEleSizeOpConversion,
        BoxIsAllocOpConversion, BoxIsArrayOpConversion, BoxIsPtrOpConversion,
        BoxProcHostOpConversion, BoxRankOpConversion, BoxTypeDescOpConversion,
        CallOpConversion, CmpcOpConversion, ConstcOpConversion,
        ConvertOpConversion, CoordinateOpConversion, DispatchOpConversion,
        DispatchTableOpConversion, DivcOpConversion, DTEntryOpConversion,
        EmboxOpConversion, EmboxCharOpConversion, EmboxProcOpConversion,
        FieldIndexOpConversion, FirEndOpConversion, ExtractValueOpConversion,
        IsPresentOpConversion, FreeMemOpConversion, GenTypeDescOpConversion,
        GlobalLenOpConversion, GlobalOpConversion, HasValueOpConversion,
        InsertOnRangeOpConversion, InsertValueOpConversion,
        LenParamIndexOpConversion, LoadOpConversion, MulcOpConversion,
        NegcOpConversion, NoReassocOpConversion, SelectCaseOpConversion,
        SelectOpConversion, SelectRankOpConversion, SelectTypeOpConversion,
        ShapeOpConversion, ShapeShiftOpConversion, ShiftOpConversion,
        SliceOpConversion, StoreOpConversion, StringLitOpConversion,
        SubcOpConversion, UnboxCharOpConversion, UnboxProcOpConversion,
        UndefOpConversion, UnreachableOpConversion, XArrayCoorOpConversion,
        XEmboxOpConversion, XReboxOpConversion,
        ZeroOpConversion>(context, typeConverter);
    mlir::populateStdToLLVMConversionPatterns(typeConverter, pattern);
    mlir::populateOpenMPToLLVMConversionPatterns(typeConverter, pattern);
    mlir::arith::populateArithmeticToLLVMConversionPatterns(typeConverter,
                                                            pattern);
    mlir::ConversionTarget target{*context};
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    // The OpenMP dialect is legal for Operations without regions, for those
    // which contains regions it is legal if the region contains only the
    // LLVM dialect.
    target.addDynamicallyLegalOp<mlir::omp::ParallelOp, mlir::omp::WsLoopOp,
                                 mlir::omp::MasterOp>([&](Operation *op) {
      return typeConverter.isLegal(&op->getRegion(0));
    });
    target.addLegalDialect<mlir::omp::OpenMPDialect>();

    // required NOPs for applying a full conversion
    target.addLegalOp<mlir::ModuleOp>();

    // apply the patterns
    if (mlir::failed(mlir::applyFullConversion(getModule(), target,
                                               std::move(pattern)))) {
      mlir::emitError(loc, "error in converting to LLVM-IR dialect\n");
      signalPassFailure();
    }
  }
};

/// Lower from LLVM IR dialect to proper LLVM-IR and dump the module
struct LLVMIRLoweringPass
    : public mlir::PassWrapper<LLVMIRLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  using Printer = fir::LLVMIRLoweringPrinter;
  LLVMIRLoweringPass(raw_ostream &output, Printer p)
      : output{output}, printer{p} {}

  mlir::ModuleOp getModule() { return getOperation(); }

  void runOnOperation() override final {
    auto *ctx = getModule().getContext();
    auto optName = getModule().getName();
    LLVMContext llvmCtx;
    if (auto llvmModule = mlir::translateModuleToLLVMIR(
            getModule(), llvmCtx, optName ? *optName : "FIRModule")) {
      printer(*llvmModule, output);
      return;
    }

    mlir::emitError(mlir::UnknownLoc::get(ctx), "could not emit LLVM-IR\n");
    signalPassFailure();
  }

private:
  raw_ostream &output;
  Printer printer;
};

} // namespace

std::unique_ptr<mlir::Pass> fir::createFIRToLLVMPass() {
  return std::make_unique<FIRToLLVMLowering>();
}

std::unique_ptr<mlir::Pass>
fir::createLLVMDialectToLLVMPass(raw_ostream &output,
                                 fir::LLVMIRLoweringPrinter printer) {
  return std::make_unique<LLVMIRLoweringPass>(output, printer);
}
