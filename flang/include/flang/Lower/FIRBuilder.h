//===-- Lower/FirBuilder.h -- FIR operation builder -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builder routines for constructing the FIR dialect of MLIR. As FIR is a
// dialect of MLIR, it makes extensive use of MLIR interfaces and MLIR's coding
// style (https://mlir.llvm.org/getting_started/DeveloperGuide/) is used in this
// module.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_FIRBUILDER_H
#define FORTRAN_LOWER_FIRBUILDER_H

#include "flang/Common/reference.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"

namespace Fortran::lower {

class AbstractConverter;

//===----------------------------------------------------------------------===//
// FirOpBuilder interface extensions
//===----------------------------------------------------------------------===//

// TODO: Used CRTP to extend the FirOpBuilder interface, but this leads to some
// complex and downright ugly template code.

/// Extension class to facilitate lowering of CHARACTER operation
template <typename T>
class CharacterOpsBuilder {
public:
  // access the implementation
  T &impl() { return *static_cast<T *>(this); }

  /// Unless otherwise stated, all mlir::Value inputs of these pseudo-fir ops
  /// must be of type:
  /// - fir.boxchar<kind> (dynamic length character),
  /// - fir.ref<fir.array<len x fir.char<kind>>> (character with compile time
  ///      constant length),
  /// - fir.array<len x fir.char<kind>> (compile time constant character)

  /// Copy the \p count first characters of \p src into \p dest.
  /// \p count can have any integer type.
  void createCopy(mlir::Value dest, mlir::Value src, mlir::Value count);

  /// Set characters of \p str at position [\p lower, \p upper) to blanks.
  /// \p lower and \upper bounds are zero based.
  /// If \p upper <= \p lower, no padding is done.
  /// \p upper and \p lower can have any integer type.
  void createPadding(mlir::Value str, mlir::Value lower, mlir::Value upper);

  /// Create str(lb:ub), lower bounds must always be specified, upper
  /// bound is optional.
  mlir::Value createSubstring(mlir::Value str,
                              llvm::ArrayRef<mlir::Value> bounds);

  /// Return blank character of given \p type !fir.char<kind>
  mlir::Value createBlankConstant(fir::CharacterType type);

  /// Lower \p lhs = \p rhs where \p lhs and \p rhs are scalar characters.
  /// It handles cases where \p lhs and \p rhs may overlap.
  void createAssign(mlir::Value lhs, mlir::Value rhs);

  /// Lower an assignment where the buffer and LEN parameter are known and do
  /// not need to be unboxed.
  void createAssign(mlir::Value lptr, mlir::Value llen, mlir::Value rptr,
                    mlir::Value rlen);

  /// Embox \p addr and \p len and return fir.boxchar.
  /// Take care of type conversions before emboxing.
  /// \p len is converted to the integer type for character lengths if needed.
  mlir::Value createEmboxChar(mlir::Value addr, mlir::Value len);
  /// Unbox \p boxchar into (fir.ref<fir.char<kind>>, getLengthType()).
  std::pair<mlir::Value, mlir::Value> createUnboxChar(mlir::Value boxChar);

  /// Allocate a temp of fir::CharacterType type and length len.
  /// Returns related fir.ref<fir.char<kind>>.
  mlir::Value createCharacterTemp(mlir::Type type, mlir::Value len);
  /// Allocate a temp of compile time constant length.
  /// Returns related fir.ref<fir.array<len x fir.char<kind>>>.
  mlir::Value createCharacterTemp(mlir::Type type, int len);

  /// Return buffer/length pair of character str, if str is a constant,
  /// it is allocated into a temp, otherwise, its memory reference is
  /// returned as the buffer.
  /// The buffer type of str is of type:
  ///   - fir.ref<fir.array<len x fir.char<kind>>> if str has compile time
  ///      constant length.
  ///   - fir.ref<fir.char<kind>> if str has dynamic length.
  std::pair<mlir::Value, mlir::Value> materializeCharacter(mlir::Value str);

  /// Return true if \p type is a character literal type (is
  /// fir.array<len x fir.char<kind>>).;
  static bool isCharacterLiteral(mlir::Type type);

  /// Return true if \p type is one of the following type
  /// - fir.boxchar<kind>
  /// - fir.ref<fir.array<len x fir.char<kind>>>
  /// - fir.array<len x fir.char<kind>>
  static bool isCharacter(mlir::Type type);

  /// Extract the kind of a character type
  static int getCharacterKind(mlir::Type type);

  /// Return the integer type that must be used to manipulate
  /// Character lengths.
  mlir::Type getLengthType();
};

/// Extension class to facilitate lowering of COMPLEX manipulations in FIR.
template <typename T>
class ComplexOpsBuilder {
public:
  // The values of part enum members are meaningful for
  // InsertValueOp and ExtractValueOp so they are explicit.
  enum class Part { Real = 0, Imag = 1 };

  // access the implementation
  T &impl() { return *static_cast<T *>(this); }

  /// Type helper. They do not create MLIR operations.
  mlir::Type getComplexPartType(mlir::Value cplx);
  mlir::Type getComplexPartType(mlir::Type complexType);

  /// Complex operation creation helper. They create MLIR operations.
  mlir::Value createComplex(fir::KindTy kind, mlir::Value real,
                            mlir::Value imag);

  /// Create a complex value.
  mlir::Value createComplex(mlir::Location loc, mlir::Type complexType,
                            mlir::Value real, mlir::Value imag);

  mlir::Value extractComplexPart(mlir::Value cplx, bool isImagPart) {
    return isImagPart ? extract<Part::Imag>(cplx) : extract<Part::Real>(cplx);
  }

  /// Returns (Real, Imag) pair of \p cplx
  std::pair<mlir::Value, mlir::Value> extractParts(mlir::Value cplx) {
    return {extract<Part::Real>(cplx), extract<Part::Imag>(cplx)};
  }
  mlir::Value insertComplexPart(mlir::Value cplx, mlir::Value part,
                                bool isImagPart) {
    return isImagPart ? insert<Part::Imag>(cplx, part)
                      : insert<Part::Real>(cplx, part);
  }

  mlir::Value createComplexCompare(mlir::Value cplx1, mlir::Value cplx2,
                                   bool eq);

protected:
  template <Part partId>
  mlir::Value extract(mlir::Value cplx) {
    return impl().template createHere<fir::ExtractValueOp>(
        getComplexPartType(cplx), cplx, createPartId<partId>());
  }

  template <Part partId>
  mlir::Value insert(mlir::Value cplx, mlir::Value part) {
    return impl().template createHere<fir::InsertValueOp>(
        cplx.getType(), cplx, part, createPartId<partId>());
  }

  template <Part partId>
  mlir::Value createPartId() {
    return impl().createIntegerConstant(impl().getIntegerType(32),
                                        static_cast<int>(partId));
  }
};

/// Extension class to facilitate lowering of COMPLEX manipulations in FIR.
template <typename T>
class IntrinsicCallOpsBuilder {
public:
  // access the implementation
  T &impl() { return *static_cast<T *>(this); }

  // TODO: Expose interface to get specific intrinsic function address.
  // TODO: Handle intrinsic subroutine.
  // TODO: Intrinsics that do not require their arguments to be defined
  //   (e.g shape inquiries) might not fit in the current interface that
  //   requires mlir::Value to be provided.
  // TODO: Error handling interface ?
  // TODO: Implementation is incomplete. Many intrinsics to tbd.

  /// Generate the FIR+MLIR operations for the generic intrinsic \p name
  /// with arguments \p args and expected result type \p resultType.
  /// Returned mlir::Value is the returned Fortran intrinsic value.
  mlir::Value genIntrinsicCall(llvm::StringRef name, mlir::Type resultType,
                               llvm::ArrayRef<mlir::Value> args);
  /// Direct access to intrinsics that may be used by lowering outside
  /// of intrinsic call lowering.

  /// Generate maximum. There must be at least one argument and all arguments
  /// must have the same type.
  mlir::Value genMax(llvm::ArrayRef<mlir::Value> args);
  /// Generate minimum. Same constraints as genMax.
  mlir::Value genMin(llvm::ArrayRef<mlir::Value> args);
  /// Generate power function x**y with given the expected
  /// result type.
  mlir::Value genPow(mlir::Type resultType, mlir::Value x, mlir::Value y);
};

//===----------------------------------------------------------------------===//
// FirOpBuilder
//===----------------------------------------------------------------------===//

/// Extends the MLIR OpBuilder to provide methods for building common FIR
/// patterns.
class FirOpBuilder : public mlir::OpBuilder,
                     public CharacterOpsBuilder<FirOpBuilder>,
                     public ComplexOpsBuilder<FirOpBuilder>,
                     public IntrinsicCallOpsBuilder<FirOpBuilder> {
public:
  using OpBuilder::OpBuilder;

  /// TODO: remove this as caching the location may have the location
  /// unexpectedly overridden along the way.
  /// Set the current location. Used by createHere template method, etc.
  void setLocation(mlir::Location loc) { currentLoc = loc; }

  /// Get the current location (if any) or return unknown location.
  mlir::Location getLoc() {
    return currentLoc.hasValue() ? currentLoc.getValue() : getUnknownLoc();
  }

  template <typename OP, typename... AS>
  auto createHere(AS... args) {
    return create<OP>(getLoc(), std::forward<AS>(args)...);
  }

  mlir::Region &getRegion() { return *getBlock()->getParent(); }

  /// Get the current Module
  mlir::ModuleOp getModule() {
    return getRegion().getParentOfType<mlir::ModuleOp>();
  }

  /// Get the current Function
  mlir::FuncOp getFunction() {
    return getRegion().getParentOfType<mlir::FuncOp>();
  }

  /// Get the entry block of the current Function
  mlir::Block *getEntryBlock() { return &getFunction().front(); }

  mlir::Type getRefType(mlir::Type eleTy);

  /// Create an integer constant of type \p type and value \p i.
  mlir::Value createIntegerConstant(mlir::Type integerType, std::int64_t i);

  mlir::Value createRealConstant(mlir::Location loc, mlir::Type realType,
                                 const llvm::APFloat &val);

  mlir::Value allocateLocal(mlir::Location loc, mlir::Type ty,
                            llvm::StringRef nm,
                            llvm::ArrayRef<mlir::Value> shape);

  /// Create a temporary. A temp is allocated using `fir.alloca` and can be read
  /// and written using `fir.load` and `fir.store`, resp.  The temporary can be
  /// given a name via a front-end `Symbol` or a `StringRef`.
  mlir::Value createTemporary(mlir::Location loc, mlir::Type type,
                              llvm::StringRef name = {},
                              llvm::ArrayRef<mlir::Value> shape = {});

  mlir::Value createTemporary(mlir::Type type, llvm::StringRef name = {},
                              llvm::ArrayRef<mlir::Value> shape = {}) {
    return createTemporary(getLoc(), type, name, shape);
  }

  /// Create an unnamed and untracked temporary on the stack.
  mlir::Value createTemporary(mlir::Type type,
                              llvm::ArrayRef<mlir::Value> shape) {
    return createTemporary(getLoc(), type, llvm::StringRef{}, shape);
  }

  /// Create a global value.
  fir::GlobalOp createGlobal(mlir::Location loc, mlir::Type type,
                             llvm::StringRef name,
                             mlir::StringAttr linkage = {},
                             mlir::Attribute value = {}, bool isConst = false);

  fir::GlobalOp createGlobal(mlir::Location loc, mlir::Type type,
                             llvm::StringRef name, bool isConst,
                             std::function<void(FirOpBuilder &)> bodyBuilder,
                             mlir::StringAttr linkage = {});

  /// Create a global constant (read-only) value.
  fir::GlobalOp createGlobalConstant(mlir::Location loc, mlir::Type type,
                                     llvm::StringRef name,
                                     mlir::StringAttr linkage = {},
                                     mlir::Attribute value = {}) {
    return createGlobal(loc, type, name, linkage, value, /*isConst=*/true);
  }

  fir::GlobalOp
  createGlobalConstant(mlir::Location loc, mlir::Type type,
                       llvm::StringRef name,
                       std::function<void(FirOpBuilder &)> bodyBuilder,
                       mlir::StringAttr linkage = {}) {
    return createGlobal(loc, type, name, /*isConst=*/true, bodyBuilder,
                        linkage);
  }

  /// Convert a StringRef string into a fir::StringLitOp.
  fir::StringLitOp createStringLit(mlir::Location loc, mlir::Type eleTy,
                                   llvm::StringRef string);

  /// Get a function by name. If the function exists in the current module, it
  /// is returned. Otherwise, a null FuncOp is returned.
  mlir::FuncOp getNamedFunction(llvm::StringRef name) {
    return getNamedFunction(getModule(), name);
  }

  static mlir::FuncOp getNamedFunction(mlir::ModuleOp module,
                                       llvm::StringRef name);

  fir::GlobalOp getNamedGlobal(llvm::StringRef name) {
    return getNamedGlobal(getModule(), name);
  }

  static fir::GlobalOp getNamedGlobal(mlir::ModuleOp module,
                                      llvm::StringRef name);

  /// Create a new FuncOp. If the function may have already been created, use
  /// `addNamedFunction` instead.
  mlir::FuncOp createFunction(mlir::Location loc, llvm::StringRef name,
                              mlir::FunctionType ty) {
    return createFunction(loc, getModule(), name, ty);
  }

  mlir::FuncOp createFunction(llvm::StringRef name, mlir::FunctionType ty) {
    return createFunction(getLoc(), name, ty);
  }

  static mlir::FuncOp createFunction(mlir::Location loc, mlir::ModuleOp module,
                                     llvm::StringRef name,
                                     mlir::FunctionType ty);

  /// Determine if the named function is already in the module. Return the
  /// instance if found, otherwise add a new named function to the module.
  mlir::FuncOp addNamedFunction(mlir::Location loc, llvm::StringRef name,
                                mlir::FunctionType ty) {
    if (auto func = getNamedFunction(name))
      return func;
    return createFunction(loc, name, ty);
  }

  mlir::FuncOp addNamedFunction(llvm::StringRef name, mlir::FunctionType ty) {
    if (auto func = getNamedFunction(name))
      return func;
    return createFunction(name, ty);
  }

  static mlir::FuncOp addNamedFunction(mlir::Location loc,
                                       mlir::ModuleOp module,
                                       llvm::StringRef name,
                                       mlir::FunctionType ty) {
    if (auto func = getNamedFunction(module, name))
      return func;
    return createFunction(loc, module, name, ty);
  }

  //===--------------------------------------------------------------------===//
  // LoopOp helpers
  //===--------------------------------------------------------------------===//

  using BodyGenerator = std::function<void(FirOpBuilder &, mlir::Value)>;

  /// Build loop [\p lb, \p ub] with step \p step.
  /// If \p step is an empty value, 1 is used for the step.
  void createLoop(mlir::Value lb, mlir::Value ub, mlir::Value step,
                  const BodyGenerator &bodyGenerator);

  /// Build loop [\p lb,  \p ub] with step 1.
  void createLoop(mlir::Value lb, mlir::Value ub,
                  const BodyGenerator &bodyGenerator);

  /// Build loop [0, \p count) with step 1.
  void createLoop(mlir::Value count, const BodyGenerator &bodyGenerator);

  /// Cast the input value to IndexType.
  mlir::Value convertToIndexType(mlir::Value integer);

private:
  llvm::Optional<mlir::Location> currentLoc{};
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_FIRBUILDER_H
