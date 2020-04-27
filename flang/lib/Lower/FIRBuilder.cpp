//===-- FIRBuilder.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/FIRBuilder.h"
#include "SymbolMap.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Semantics/symbol.h"
#include "llvm/Support/ErrorHandling.h"

mlir::FuncOp Fortran::lower::FirOpBuilder::createFunction(
    mlir::Location loc, mlir::ModuleOp module, llvm::StringRef name,
    mlir::FunctionType ty) {
  return fir::createFuncOp(loc, module, name, ty);
}

mlir::FuncOp
Fortran::lower::FirOpBuilder::getNamedFunction(mlir::ModuleOp modOp,
                                               llvm::StringRef name) {
  return modOp.lookupSymbol<mlir::FuncOp>(name);
}

fir::GlobalOp
Fortran::lower::FirOpBuilder::getNamedGlobal(mlir::ModuleOp modOp,
                                             llvm::StringRef name) {
  return modOp.lookupSymbol<fir::GlobalOp>(name);
}

mlir::Type Fortran::lower::FirOpBuilder::getRefType(mlir::Type eleTy) {
  return fir::ReferenceType::get(eleTy);
}

mlir::Value
Fortran::lower::FirOpBuilder::createIntegerConstant(mlir::Type intType,
                                                    std::int64_t cst) {
  return createHere<mlir::ConstantOp>(intType, getIntegerAttr(intType, cst));
}

mlir::Value Fortran::lower::FirOpBuilder::createRealConstant(
    mlir::Location loc, mlir::Type realType, const llvm::APFloat &val) {
  return create<mlir::ConstantOp>(loc, realType, getFloatAttr(realType, val));
}

// This seems a bit hacky. The scaling ssa-values must be type `index`. Should
// these just be constructed with the correct type, rather than massaged lazily
// here?
mlir::Value
Fortran::lower::FirOpBuilder::allocateLocal(mlir::Location loc, mlir::Type ty,
                                            llvm::StringRef nm,
                                            llvm::ArrayRef<mlir::Value> shape) {
  llvm::SmallVector<mlir::Value, 8> indices;
  auto idxTy = getIndexType();
  llvm::for_each(shape, [&](mlir::Value sh) {
    if (sh.getType().isa<mlir::IndexType>()) {
      indices.push_back(sh);
    } else {
      auto cast = create<fir::ConvertOp>(loc, idxTy, sh);
      indices.push_back(cast);
    }
  });
  // FIXME: have the builder drop the attribute when the name is empty
  if (nm.size())
    return create<fir::AllocaOp>(loc, ty, nm, llvm::None, indices);
  return create<fir::AllocaOp>(loc, ty, llvm::None, indices);
}

/// Create a temporary variable on the stack. Anonymous temporaries have no
/// `name` value.
mlir::Value Fortran::lower::FirOpBuilder::createTemporary(
    mlir::Location loc, mlir::Type type, llvm::StringRef name,
    llvm::ArrayRef<mlir::Value> shape) {
  auto insPt = saveInsertionPoint();
  if (shape.empty())
    setInsertionPointToStart(getEntryBlock());
  else
    setInsertionPointAfter(shape.back().getDefiningOp());
  assert(!type.isa<fir::ReferenceType>() && "cannot be a reference");
  auto ae = create<fir::AllocaOp>(loc, type, name, llvm::None, shape);
  restoreInsertionPoint(insPt);
  return ae;
}

/// Create a global variable in the (read-only) data section. A global variable
/// must have a unique name to identify and reference it.
fir::GlobalOp Fortran::lower::FirOpBuilder::createGlobal(
    mlir::Location loc, mlir::Type type, llvm::StringRef name,
    mlir::StringAttr linkage, mlir::Attribute value, bool isConst) {
  auto module = getModule();
  auto insertPt = saveInsertionPoint();
  if (auto glob = module.lookupSymbol<fir::GlobalOp>(name))
    return glob;
  setInsertionPoint(module.getBody()->getTerminator());
  auto glob = create<fir::GlobalOp>(loc, name, isConst, type, value, linkage);
  restoreInsertionPoint(insertPt);
  return glob;
}

fir::GlobalOp Fortran::lower::FirOpBuilder::createGlobal(
    mlir::Location loc, mlir::Type type, llvm::StringRef name, bool isConst,
    std::function<void(FirOpBuilder &)> bodyBuilder, mlir::StringAttr linkage) {
  auto module = getModule();
  auto insertPt = saveInsertionPoint();
  if (auto glob = module.lookupSymbol<fir::GlobalOp>(name))
    return glob;
  setInsertionPoint(module.getBody()->getTerminator());
  auto glob = create<fir::GlobalOp>(loc, name, isConst, type, mlir::Attribute{},
                                    linkage);
  auto &region = glob.getRegion();
  region.push_back(new mlir::Block);
  auto &block = glob.getRegion().back();
  setInsertionPointToStart(&block);
  bodyBuilder(*this);
  restoreInsertionPoint(insertPt);
  return glob;
}

//===----------------------------------------------------------------------===//
// LoopOp builder
//===----------------------------------------------------------------------===//

void Fortran::lower::FirOpBuilder::createLoop(
    mlir::Value lb, mlir::Value ub, mlir::Value step,
    const BodyGenerator &bodyGenerator) {
  auto lbi = convertToIndexType(lb);
  auto ubi = convertToIndexType(ub);
  assert(step && "step must be an actual Value");
  auto inc = convertToIndexType(step);
  auto loop = createHere<fir::LoopOp>(lbi, ubi, inc);
  auto insertPt = saveInsertionPoint();
  setInsertionPointToStart(loop.getBody());
  auto index = loop.getInductionVar();
  bodyGenerator(*this, index);
  restoreInsertionPoint(insertPt);
}

void Fortran::lower::FirOpBuilder::createLoop(
    mlir::Value lb, mlir::Value ub, const BodyGenerator &bodyGenerator) {
  createLoop(lb, ub, createIntegerConstant(getIndexType(), 1), bodyGenerator);
}

void Fortran::lower::FirOpBuilder::createLoop(
    mlir::Value count, const BodyGenerator &bodyGenerator) {
  auto indexType = getIndexType();
  auto zero = createIntegerConstant(indexType, 0);
  auto one = createIntegerConstant(count.getType(), 1);
  auto up = createHere<mlir::SubIOp>(count, one);
  createLoop(zero, up, one, bodyGenerator);
}

mlir::Value
Fortran::lower::FirOpBuilder::convertToIndexType(mlir::Value integer) {
  // abort now if not an integral type
  fir::verifyIntegralType(integer.getType());
  return createHere<fir::ConvertOp>(getIndexType(), integer);
}

//===----------------------------------------------------------------------===//
// CharacterOpsBuilder implementation
//===----------------------------------------------------------------------===//

namespace {
/// CharacterOpsBuilder implementation
struct CharacterOpsBuilderImpl {
  CharacterOpsBuilderImpl(Fortran::lower::FirOpBuilder &b) : builder{b} {}

  /// Opened char box
  /// Interchange format to avoid inserting unbox/embox everywhere while
  /// evaluating character expressions.
  struct Char {
    /// Get fir.char<kind> type with the same kind as inside str.
    static inline fir::CharacterType getCharacterType(mlir::Type type) {
      if (auto boxType = type.dyn_cast<fir::BoxCharType>())
        return boxType.getEleTy();
      if (auto refType = type.dyn_cast<fir::ReferenceType>())
        type = refType.getEleTy();
      if (auto seqType = type.dyn_cast<fir::SequenceType>())
        type = seqType.getEleTy();
      if (auto charType = type.dyn_cast<fir::CharacterType>())
        return charType;
      llvm_unreachable("Invalid character value type");
    }

    fir::CharacterType getCharacterType() const {
      return getCharacterType(data.getType());
    }
    /// Get fir.ref<fir.char<kind>> type.
    fir::ReferenceType getReferenceType() const {
      return fir::ReferenceType::get(getCharacterType());
    }

    bool needToMaterialize() const {
      return data.getType().isa<fir::SequenceType>() ||
             data.getType().isa<fir::CharacterType>();
    }

    std::optional<fir::SequenceType::Extent> getCompileTimeLength() const {
      auto type = data.getType();
      if (type.isa<fir::CharacterType>())
        return 1;
      if (auto refType = type.dyn_cast<fir::ReferenceType>())
        type = refType.getEleTy();
      if (auto seqType = type.dyn_cast<fir::SequenceType>()) {
        auto shape = seqType.getShape();
        assert(shape.size() == 1 && "only scalar character supported");
        return shape[0];
      }
      return {};
    }

    /// Data must be of type:
    /// - fir.ref<fir.char<kind>> (dynamic length)
    /// - fir.ref<fir.array<len x fir.char<kind>>> (len compile time constant).
    /// - fir.array<len x fir.char<kind>> (character constant)
    /// - fir.char<kind> (length one character)
    mlir::Value data;
    mlir::Value len;
  };

  Char materializeValue(Char str) {
    if (!str.needToMaterialize())
      return str;
    auto variable = builder.createHere<fir::AllocaOp>(str.data.getType());
    builder.createHere<fir::StoreOp>(str.data, variable);
    return {variable, str.len};
  }

  Char toDataLengthPair(mlir::Value character) {
    auto lenType = builder.getLengthType();
    auto type = character.getType();
    if (auto boxCharType = type.dyn_cast<fir::BoxCharType>()) {
      auto refType = fir::ReferenceType::get(boxCharType.getEleTy());
      auto unboxed =
          builder.createHere<fir::UnboxCharOp>(refType, lenType, character);
      return {unboxed.getResult(0), unboxed.getResult(1)};
    }
    if (auto seqType = type.dyn_cast<fir::CharacterType>()) {
      // Materialize length for usage into character manipulations.
      auto len = builder.createIntegerConstant(lenType, 1);
      return {character, len};
    }
    if (auto refType = type.dyn_cast<fir::ReferenceType>())
      type = refType.getEleTy();
    if (auto seqType = type.dyn_cast<fir::SequenceType>()) {
      assert(seqType.hasConstantShape() &&
             "ssa array value must have constant length");
      auto shape = seqType.getShape();
      assert(shape.size() == 1 && "only scalar character supported");
      // Materialize length for usage into character manipulations.
      auto len = builder.createIntegerConstant(lenType, shape[0]);
      // FIXME: this seems to work for tests, but don't think it is correct
      if (auto load = dyn_cast<fir::LoadOp>(character.getDefiningOp()))
        return {load.memref(), len};
      return {character, len};
    }
    llvm_unreachable("unexpected character type");
  }

  mlir::Value createEmbox(Char str) {
    // BoxChar require a reference.
    if (str.needToMaterialize())
      str = materializeValue(str);
    auto kind = str.getCharacterType().getFKind();
    auto boxCharType = fir::BoxCharType::get(builder.getContext(), kind);
    auto refType = str.getReferenceType();
    // So far, fir.emboxChar fails lowering to llvm when it is given
    // fir.data<fir.array<len x fir.char<kind>>> types, so convert to
    // fir.data<fir.char<kind>> if needed.
    if (refType != str.data.getType())
      str.data = builder.createHere<fir::ConvertOp>(refType, str.data);
    // Convert in case the provided length is not of the integer type that must
    // be used in boxchar.
    auto lenType = builder.getLengthType();
    if (str.len.getType() != lenType)
      str.len = builder.createHere<fir::ConvertOp>(lenType, str.len);
    return builder.createHere<fir::EmboxCharOp>(boxCharType, str.data, str.len);
  }

  mlir::Value createLoadCharAt(Char str, mlir::Value index) {
    auto addr = builder.createHere<fir::CoordinateOp>(str.getReferenceType(),
                                                      str.data, index);
    return builder.createHere<fir::LoadOp>(addr);
  }
  void createStoreCharAt(Char str, mlir::Value index, mlir::Value c) {
    assert(!str.needToMaterialize() && "not in memory");
    auto addr = builder.createHere<fir::CoordinateOp>(str.getReferenceType(),
                                                      str.data, index);
    builder.createHere<fir::StoreOp>(c, addr);
  }

  void createCopy(Char dest, Char src, mlir::Value count) {
    builder.createLoop(
        count, [&](Fortran::lower::FirOpBuilder &handler, mlir::Value index) {
          CharacterOpsBuilderImpl charHandler{handler};
          auto charVal = charHandler.createLoadCharAt(src, index);
          charHandler.createStoreCharAt(dest, index, charVal);
        });
  }

  void createPadding(Char str, mlir::Value lower, mlir::Value upper) {
    auto blank = createBlankConstant(str.getCharacterType());
    // Always create the loop, if upper < lower, no iteration will be
    // executed.
    builder.createLoop(
        lower, upper,
        [&](Fortran::lower::FirOpBuilder &handler, mlir::Value index) {
          CharacterOpsBuilderImpl charHandler{handler};
          charHandler.createStoreCharAt(str, index, blank);
        });
  }

  Char createTemp(mlir::Type type, mlir::Value len) {
    assert(type.isa<fir::CharacterType>() && "expected fir character type");
    llvm::SmallVector<mlir::Value, 3> sizes{len};
    auto ref =
        builder.allocateLocal(builder.getLoc(), type, llvm::StringRef{}, sizes);
    return {ref, len};
  }

  mlir::Value createTemp(mlir::Type type, int len) {
    assert(type.isa<fir::CharacterType>() && "expected fir character type");
    assert(len >= 0 && "expected positive length");
    fir::SequenceType::Shape shape{len};
    auto seqType = fir::SequenceType::get(shape, type);
    return builder.createHere<fir::AllocaOp>(seqType);
  }

  // Simple length one character assignment without loops.
  void createLengthOneAssign(Char lhs, Char rhs) {
    auto addr = lhs.data;
    auto refType = lhs.getReferenceType();
    addr = builder.createHere<fir::ConvertOp>(refType, addr);

    auto val = rhs.data;
    if (!rhs.needToMaterialize()) {
      mlir::Value rhsAddr = rhs.data;
      rhsAddr = builder.createHere<fir::ConvertOp>(refType, rhsAddr);
      val = builder.createHere<fir::LoadOp>(rhsAddr);
    }

    builder.createHere<fir::StoreOp>(val, addr);
  }

  void createAssign(Char lhs, Char rhs) {
    auto rhsCstLen = rhs.getCompileTimeLength();
    auto lhsCstLen = lhs.getCompileTimeLength();
    bool compileTimeSameLength =
        lhsCstLen && rhsCstLen && *lhsCstLen == *rhsCstLen;

    if (compileTimeSameLength && *lhsCstLen == 1) {
      createLengthOneAssign(lhs, rhs);
      return;
    }

    // Copy the minimum of the lhs and rhs lengths and pad the lhs remainder
    // if needed.
    mlir::Value copyCount = lhs.len;
    if (!compileTimeSameLength)
      copyCount = builder.genMin({lhs.len, rhs.len});

    Char safeRhs{rhs};
    if (rhs.needToMaterialize()) {
      // TODO: revisit now that character constant handling changed.
      // Need to materialize the constant to get its elements.
      // (No equivalent of fir.coordinate_of for array value).
      safeRhs = materializeValue(rhs);
    } else {
      // If rhs is in memory, always assumes rhs might overlap with lhs
      // in a way that require a temp for the copy. That can be optimize later.
      // Only create a temp of copyCount size because we do not need more from
      // rhs.
      auto temp = createTemp(rhs.getCharacterType(), copyCount);
      createCopy(temp, rhs, copyCount);
      safeRhs = temp;
    }

    // Actual copy
    createCopy(lhs, safeRhs, copyCount);

    // Pad if needed.
    if (!compileTimeSameLength) {
      auto one = builder.createIntegerConstant(lhs.len.getType(), 1);
      auto maxPadding = builder.createHere<mlir::SubIOp>(lhs.len, one);
      createPadding(lhs, copyCount, maxPadding);
    }
  }

  mlir::Value createBlankConstant(fir::CharacterType type) {
    auto byteTy = mlir::IntegerType::get(8, builder.getContext());
    auto asciiSpace = builder.createIntegerConstant(byteTy, 0x20);
    return builder.createHere<fir::ConvertOp>(type, asciiSpace);
  }

  Char createSubstring(Char str, llvm::ArrayRef<mlir::Value> bounds) {
    // Constant need to be materialize in memory to use fir.coordinate_of.
    if (str.needToMaterialize())
      str = materializeValue(str);

    auto nbounds{bounds.size()};
    if (nbounds < 1 || nbounds > 2) {
      mlir::emitError(builder.getLoc(),
                      "Incorrect number of bounds in substring");
      return {mlir::Value{}, mlir::Value{}};
    }
    auto indexType = mlir::IndexType::get(builder.getContext());
    auto lowerBound = builder.createHere<fir::ConvertOp>(indexType, bounds[0]);
    // FIR CoordinateOp is zero based but Fortran substring are one based.
    auto oneIndex = builder.createIntegerConstant(indexType, 1);
    auto offsetIndex =
        builder.createHere<mlir::SubIOp>(lowerBound, oneIndex).getResult();
    auto substringRef = builder.createHere<fir::CoordinateOp>(
        str.getReferenceType(), str.data, offsetIndex);

    // Compute the length.
    mlir::Value substringLen{};
    if (nbounds < 2) {
      substringLen = builder.createHere<mlir::SubIOp>(str.len, bounds[0]);
    } else {
      substringLen = builder.createHere<mlir::SubIOp>(bounds[1], bounds[0]);
    }
    auto one = builder.createIntegerConstant(substringLen.getType(), 1);
    substringLen = builder.createHere<mlir::AddIOp>(substringLen, one);

    // Set length to zero if bounds were reversed (Fortran 2018 9.4.1)
    auto zero = builder.createIntegerConstant(substringLen.getType(), 0);
    auto cdt = builder.createHere<mlir::CmpIOp>(mlir::CmpIPredicate::slt,
                                                substringLen, zero);
    substringLen = builder.createHere<mlir::SelectOp>(cdt, zero, substringLen);

    return {substringRef, substringLen};
  }
  Fortran::lower::FirOpBuilder &builder;
};
} // namespace

fir::StringLitOp Fortran::lower::FirOpBuilder::createStringLit(
    mlir::Location loc, mlir::Type eleTy, llvm::StringRef data) {
  auto strAttr = mlir::StringAttr::get(data, getContext());
  auto valTag = mlir::Identifier::get(fir::StringLitOp::value(), getContext());
  mlir::NamedAttribute dataAttr(valTag, strAttr);
  auto sizeTag = mlir::Identifier::get(fir::StringLitOp::size(), getContext());
  mlir::NamedAttribute sizeAttr(sizeTag, getI64IntegerAttr(data.size()));
  llvm::SmallVector<mlir::NamedAttribute, 2> attrs{dataAttr, sizeAttr};
  auto arrTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, data.size()), eleTy);
  return create<fir::StringLitOp>(loc, llvm::ArrayRef<mlir::Type>{arrTy},
                                  llvm::None, attrs);
}

template <typename T>
void Fortran::lower::CharacterOpsBuilder<T>::createCopy(mlir::Value dest,
                                                        mlir::Value src,
                                                        mlir::Value count) {
  CharacterOpsBuilderImpl bimpl{impl()};
  bimpl.createCopy(bimpl.toDataLengthPair(dest), bimpl.toDataLengthPair(src),
                   count);
}

template void Fortran::lower::CharacterOpsBuilder<
    Fortran::lower::FirOpBuilder>::createCopy(mlir::Value, mlir::Value,
                                              mlir::Value);

template <typename T>
void Fortran::lower::CharacterOpsBuilder<T>::createPadding(mlir::Value str,
                                                           mlir::Value lower,
                                                           mlir::Value upper) {
  CharacterOpsBuilderImpl bimpl{impl()};
  bimpl.createPadding(bimpl.toDataLengthPair(str), lower, upper);
}
template void Fortran::lower::CharacterOpsBuilder<
    Fortran::lower::FirOpBuilder>::createPadding(mlir::Value, mlir::Value,
                                                 mlir::Value);

template <typename T>
mlir::Value Fortran::lower::CharacterOpsBuilder<T>::createSubstring(
    mlir::Value str, llvm::ArrayRef<mlir::Value> bounds) {
  CharacterOpsBuilderImpl bimpl{impl()};
  return bimpl.createEmbox(
      bimpl.createSubstring(bimpl.toDataLengthPair(str), bounds));
}
template mlir::Value Fortran::lower::CharacterOpsBuilder<
    Fortran::lower::FirOpBuilder>::createSubstring(mlir::Value,
                                                   llvm::ArrayRef<mlir::Value>);

template <typename T>
void Fortran::lower::CharacterOpsBuilder<T>::createAssign(mlir::Value lhs,
                                                          mlir::Value rhs) {
  CharacterOpsBuilderImpl bimpl{impl()};
  bimpl.createAssign(bimpl.toDataLengthPair(lhs), bimpl.toDataLengthPair(rhs));
}
template void Fortran::lower::CharacterOpsBuilder<
    Fortran::lower::FirOpBuilder>::createAssign(mlir::Value, mlir::Value);

template <typename T>
void Fortran::lower::CharacterOpsBuilder<T>::createAssign(mlir::Value lptr,
                                                          mlir::Value llen,
                                                          mlir::Value rptr,
                                                          mlir::Value rlen) {
  CharacterOpsBuilderImpl bimpl = impl();
  bimpl.createAssign(CharacterOpsBuilderImpl::Char{lptr, llen},
                     CharacterOpsBuilderImpl::Char{rptr, rlen});
}
template void
Fortran::lower::CharacterOpsBuilder<Fortran::lower::FirOpBuilder>::createAssign(
    mlir::Value lptr, mlir::Value llen, mlir::Value rptr, mlir::Value rlen);

template <typename T>
mlir::Value
Fortran::lower::CharacterOpsBuilder<T>::createEmboxChar(mlir::Value addr,
                                                        mlir::Value len) {
  return CharacterOpsBuilderImpl{impl()}.createEmbox(
      CharacterOpsBuilderImpl::Char{addr, len});
}
template mlir::Value Fortran::lower::CharacterOpsBuilder<
    Fortran::lower::FirOpBuilder>::createEmboxChar(mlir::Value, mlir::Value);

template <typename T>
std::pair<mlir::Value, mlir::Value>
Fortran::lower::CharacterOpsBuilder<T>::createUnboxChar(mlir::Value boxChar) {
  auto c{CharacterOpsBuilderImpl{impl()}.toDataLengthPair(boxChar)};
  return {c.data, c.len};
}

template std::pair<mlir::Value, mlir::Value>
    Fortran::lower::CharacterOpsBuilder<
        Fortran::lower::FirOpBuilder>::createUnboxChar(mlir::Value);

template <typename T>
mlir::Value
Fortran::lower::CharacterOpsBuilder<T>::createCharacterTemp(mlir::Type type,
                                                            mlir::Value len) {
  CharacterOpsBuilderImpl bimpl{impl()};
  return bimpl.createEmbox(bimpl.createTemp(type, len));
}
template mlir::Value Fortran::lower::CharacterOpsBuilder<
    Fortran::lower::FirOpBuilder>::createCharacterTemp(mlir::Type, mlir::Value);
template <typename T>
mlir::Value
Fortran::lower::CharacterOpsBuilder<T>::createCharacterTemp(mlir::Type type,
                                                            int len) {
  CharacterOpsBuilderImpl bimpl{impl()};
  return bimpl.createTemp(type, len);
}
template mlir::Value Fortran::lower::CharacterOpsBuilder<
    Fortran::lower::FirOpBuilder>::createCharacterTemp(mlir::Type, int);

template <typename T>
std::pair<mlir::Value, mlir::Value>
Fortran::lower::CharacterOpsBuilder<T>::materializeCharacter(mlir::Value str) {
  CharacterOpsBuilderImpl bimpl{impl()};
  auto c = bimpl.toDataLengthPair(str);
  if (c.needToMaterialize())
    c = bimpl.materializeValue(c);
  return {c.data, c.len};
}
template std::pair<mlir::Value, mlir::Value>
    Fortran::lower::CharacterOpsBuilder<
        Fortran::lower::FirOpBuilder>::materializeCharacter(mlir::Value);

template <typename T>
bool Fortran::lower::CharacterOpsBuilder<T>::isCharacterLiteral(
    mlir::Type type) {
  if (auto seqType = type.dyn_cast<fir::SequenceType>())
    return seqType.getEleTy().isa<fir::CharacterType>();
  return false;
}
template bool Fortran::lower::CharacterOpsBuilder<
    Fortran::lower::FirOpBuilder>::isCharacterLiteral(mlir::Type);

template <typename T>
bool Fortran::lower::CharacterOpsBuilder<T>::isCharacter(mlir::Type type) {
  if (type.isa<fir::BoxCharType>())
    return true;
  if (auto refType = type.dyn_cast<fir::ReferenceType>())
    type = refType.getEleTy();
  if (auto seqType = type.dyn_cast<fir::SequenceType>()) {
    type = seqType.getEleTy();
  }
  return type.isa<fir::CharacterType>();
}
template bool Fortran::lower::CharacterOpsBuilder<
    Fortran::lower::FirOpBuilder>::isCharacter(mlir::Type);

template <typename T>
int Fortran::lower::CharacterOpsBuilder<T>::getCharacterKind(mlir::Type type) {
  return CharacterOpsBuilderImpl::Char::getCharacterType(type).getFKind();
}
template int Fortran::lower::CharacterOpsBuilder<
    Fortran::lower::FirOpBuilder>::getCharacterKind(mlir::Type);

template <typename T>
mlir::Type Fortran::lower::CharacterOpsBuilder<T>::getLengthType() {
  // FIXME: make this IndexType? Breaks some tests though.
  return impl().getIntegerType(64);
}
template mlir::Type Fortran::lower::CharacterOpsBuilder<
    Fortran::lower::FirOpBuilder>::getLengthType();

//===----------------------------------------------------------------------===//
// ComplexOpsBuilder implementation
//===----------------------------------------------------------------------===//

template <typename T>
mlir::Type Fortran::lower::ComplexOpsBuilder<T>::getComplexPartType(
    mlir::Type complexType) {
  return Fortran::lower::convertReal(
      complexType.getContext(), complexType.cast<fir::CplxType>().getFKind());
}
template mlir::Type Fortran::lower::ComplexOpsBuilder<
    Fortran::lower::FirOpBuilder>::getComplexPartType(mlir::Type);

template <typename T>
mlir::Type
Fortran::lower::ComplexOpsBuilder<T>::getComplexPartType(mlir::Value cplx) {
  return getComplexPartType(cplx.getType());
}
template mlir::Type Fortran::lower::ComplexOpsBuilder<
    Fortran::lower::FirOpBuilder>::getComplexPartType(mlir::Value);

template <typename T>
mlir::Value Fortran::lower::ComplexOpsBuilder<T>::createComplex(
    fir::KindTy kind, mlir::Value real, mlir::Value imag) {
  auto complexTy = fir::CplxType::get(impl().getContext(), kind);
  mlir::Value und = impl().template createHere<fir::UndefOp>(complexTy);
  return insert<Part::Imag>(insert<Part::Real>(und, real), imag);
}
template mlir::Value Fortran::lower::ComplexOpsBuilder<
    Fortran::lower::FirOpBuilder>::createComplex(fir::KindTy, mlir::Value,
                                                 mlir::Value);

template <typename T>
mlir::Value Fortran::lower::ComplexOpsBuilder<T>::createComplex(
    mlir::Location loc, mlir::Type cplxTy, mlir::Value real, mlir::Value imag) {
  mlir::Value und = impl().template create<fir::UndefOp>(loc, cplxTy);
  return insert<Part::Imag>(insert<Part::Real>(und, real), imag);
}
template mlir::Value Fortran::lower::ComplexOpsBuilder<
    Fortran::lower::FirOpBuilder>::createComplex(mlir::Location, mlir::Type,
                                                 mlir::Value, mlir::Value);

template <typename T>
mlir::Value Fortran::lower::ComplexOpsBuilder<T>::createComplexCompare(
    mlir::Value cplx1, mlir::Value cplx2, bool eq) {
  auto real1 = extract<Part::Real>(cplx1);
  auto real2 = extract<Part::Real>(cplx2);
  auto imag1 = extract<Part::Imag>(cplx1);
  auto imag2 = extract<Part::Imag>(cplx2);

  mlir::CmpFPredicate predicate =
      eq ? mlir::CmpFPredicate::UEQ : mlir::CmpFPredicate::UNE;
  auto &b = impl();
  mlir::Value realCmp =
      b.template createHere<mlir::CmpFOp>(predicate, real1, real2);
  mlir::Value imagCmp =
      b.template createHere<mlir::CmpFOp>(predicate, imag1, imag2);

  return eq ? b.template createHere<mlir::AndOp>(realCmp, imagCmp).getResult()
            : b.template createHere<mlir::OrOp>(realCmp, imagCmp).getResult();
}
template mlir::Value Fortran::lower::ComplexOpsBuilder<
    Fortran::lower::FirOpBuilder>::createComplexCompare(mlir::Value cplx1,
                                                        mlir::Value cplx2,
                                                        bool eq);
