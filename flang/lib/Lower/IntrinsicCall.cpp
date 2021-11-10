//===-- IntrinsicCall.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper routines for constructing the FIR dialect of MLIR. As FIR is a
// dialect of MLIR, it makes extensive use of MLIR interfaces and MLIR's coding
// style (https://mlir.llvm.org/getting_started/DeveloperGuide/) is used in this
// module.
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/IntrinsicCall.h"
#include "StatementContext.h"
#include "SymbolMap.h"
#include "flang/Common/static-multimap-view.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/Runtime.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Runtime/Character.h"
#include "flang/Optimizer/Builder/Runtime/Numeric.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Builder/Runtime/Reduction.h"
#include "flang/Optimizer/Builder/Runtime/Transformational.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <string_view>
#include <utility>

#define DEBUG_TYPE "flang-lower-intrinsic"

#define PGMATH_DECLARE
#include "flang/Evaluate/pgmath.h.inc"

/// This file implements lowering of Fortran intrinsic procedures.
/// Intrinsics are lowered to a mix of FIR and MLIR operations as
/// well as call to runtime functions or LLVM intrinsics.

/// Lowering of intrinsic procedure calls is based on a map that associates
/// Fortran intrinsic generic names to FIR generator functions.
/// All generator functions are member functions of the IntrinsicLibrary class
/// and have the same interface.
/// If no generator is given for an intrinsic name, a math runtime library
/// is searched for an implementation and, if a runtime function is found,
/// a call is generated for it. LLVM intrinsics are handled as a math
/// runtime library here.

/// Enums used to templatize and share lowering of MIN and MAX.
enum class Extremum { Min, Max };

// There are different ways to deal with NaNs in MIN and MAX.
// Known existing behaviors are listed below and can be selected for
// f18 MIN/MAX implementation.
enum class ExtremumBehavior {
  // Note: the Signaling/quiet aspect of NaNs in the behaviors below are
  // not described because there is no way to control/observe such aspect in
  // MLIR/LLVM yet. The IEEE behaviors come with requirements regarding this
  // aspect that are therefore currently not enforced. In the descriptions
  // below, NaNs can be signaling or quite. Returned NaNs may be signaling
  // if one of the input NaN was signaling but it cannot be guaranteed either.
  // Existing compilers using an IEEE behavior (gfortran) also do not fulfill
  // signaling/quiet requirements.
  IeeeMinMaximumNumber,
  // IEEE minimumNumber/maximumNumber behavior (754-2019, section 9.6):
  // If one of the argument is and number and the other is NaN, return the
  // number. If both arguements are NaN, return NaN.
  // Compilers: gfortran.
  IeeeMinMaximum,
  // IEEE minimum/maximum behavior (754-2019, section 9.6):
  // If one of the argument is NaN, return NaN.
  MinMaxss,
  // x86 minss/maxss behavior:
  // If the second argument is a number and the other is NaN, return the number.
  // In all other cases where at least one operand is NaN, return NaN.
  // Compilers: xlf (only for MAX), ifort, pgfortran -nollvm, and nagfor.
  PgfortranLlvm,
  // "Opposite of" x86 minss/maxss behavior:
  // If the first argument is a number and the other is NaN, return the
  // number.
  // In all other cases where at least one operand is NaN, return NaN.
  // Compilers: xlf (only for MIN), and pgfortran (with llvm).
  IeeeMinMaxNum
  // IEEE minNum/maxNum behavior (754-2008, section 5.3.1):
  // TODO: Not implemented.
  // It is the only behavior where the signaling/quiet aspect of a NaN argument
  // impacts if the result should be NaN or the argument that is a number.
  // LLVM/MLIR do not provide ways to observe this aspect, so it is not
  // possible to implement it without some target dependent runtime.
};

fir::ExtendedValue Fortran::lower::getAbsentIntrinsicArgument() {
  return fir::UnboxedValue{};
}

/// Test if an ExtendedValue is absent.
static bool isAbsent(const fir::ExtendedValue &exv) {
  return !fir::getBase(exv);
}
static bool isAbsent(llvm::ArrayRef<fir::ExtendedValue> args, size_t argIndex) {
  return args.size() <= argIndex || isAbsent(args[argIndex]);
}

/// Test if an ExtendedValue is present.
static bool isPresent(const fir::ExtendedValue &exv) { return !isAbsent(exv); }

/// Process calls to Maxval, Minval, Product, Sum intrinsic functions that
/// take a DIM argument.
template <typename FD>
static fir::ExtendedValue
genFuncDim(FD funcDim, mlir::Type resultType, fir::FirOpBuilder &builder,
           mlir::Location loc, Fortran::lower::StatementContext *stmtCtx,
           llvm::StringRef errMsg, mlir::Value array, fir::ExtendedValue dimArg,
           mlir::Value mask, int rank) {

  // Create mutable fir.box to be passed to the runtime for the result.
  auto resultArrayType = builder.getVarLenSeqTy(resultType, rank - 1);
  auto resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  auto resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  auto dim = isAbsent(dimArg)
                 ? builder.createIntegerConstant(loc, builder.getIndexType(), 0)
                 : fir::getBase(dimArg);
  funcDim(builder, loc, resultIrBox, array, dim, mask);

  auto res = fir::factory::genMutableBoxRead(builder, loc, resultMutableBox);
  return res.match(
      [&](const fir::ArrayBoxValue &box) -> fir::ExtendedValue {
        // Add cleanup code
        assert(stmtCtx);
        auto *bldr = &builder;
        auto temp = box.getAddr();
        stmtCtx->attachCleanup(
            [=]() { bldr->create<fir::FreeMemOp>(loc, temp); });
        return box;
      },
      [&](const fir::CharArrayBoxValue &box) -> fir::ExtendedValue {
        // Add cleanup code
        assert(stmtCtx);
        auto *bldr = &builder;
        auto temp = box.getAddr();
        stmtCtx->attachCleanup(
            [=]() { bldr->create<fir::FreeMemOp>(loc, temp); });
        return box;
      },
      [&](const auto &) -> fir::ExtendedValue {
        fir::emitFatalError(loc, errMsg);
      });
}

/// Process calls to Product, Sum intrinsic functions
template <typename FN, typename FD>
static fir::ExtendedValue
genProdOrSum(FN func, FD funcDim, mlir::Type resultType,
             fir::FirOpBuilder &builder, mlir::Location loc,
             Fortran::lower::StatementContext *stmtCtx, llvm::StringRef errMsg,
             llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 3);

  // Handle required array argument
  fir::BoxValue arryTmp = builder.createBox(loc, args[0]);
  mlir::Value array = fir::getBase(arryTmp);
  int rank = arryTmp.rank();
  assert(rank >= 1);

  // Handle optional mask argument
  auto mask = isAbsent(args[2])
                  ? builder.create<fir::AbsentOp>(
                        loc, fir::BoxType::get(builder.getI1Type()))
                  : builder.createBox(loc, args[2]);

  bool absentDim = isAbsent(args[1]);

  // We call the type specific versions because the result is scalar
  // in the case below.
  if (absentDim || rank == 1) {
    auto ty = array.getType();
    auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);
    auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
    if (fir::isa_complex(eleTy)) {
      auto result = builder.createTemporary(loc, eleTy);
      func(builder, loc, array, mask, result);
      return builder.create<fir::LoadOp>(loc, result);
    }
    auto resultBox = builder.create<fir::AbsentOp>(
        loc, fir::BoxType::get(builder.getI1Type()));
    return func(builder, loc, array, mask, resultBox);
  }
  // Handle Product/Sum cases that have an array result.
  return genFuncDim(funcDim, resultType, builder, loc, stmtCtx, errMsg, array,
                    args[1], mask, rank);
}

/// Process calls to DotProduct
template <typename FN>
static fir::ExtendedValue
genDotProd(FN func, mlir::Type resultType, fir::FirOpBuilder &builder,
           mlir::Location loc, Fortran::lower::StatementContext *stmtCtx,
           llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 2);

  // Handle required vector arguments
  mlir::Value vectorA = fir::getBase(args[0]);
  mlir::Value vectorB = fir::getBase(args[1]);

  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(vectorA.getType())
                   .cast<fir::SequenceType>()
                   .getEleTy();
  if (fir::isa_complex(eleTy)) {
    auto result = builder.createTemporary(loc, eleTy);
    func(builder, loc, vectorA, vectorB, result);
    return builder.create<fir::LoadOp>(loc, result);
  }

  auto resultBox = builder.create<fir::AbsentOp>(
      loc, fir::BoxType::get(builder.getI1Type()));
  return func(builder, loc, vectorA, vectorB, resultBox);
}

/// Process calls to Maxval, Minval, Product, Sum intrinsic functions
template <typename FN, typename FD, typename FC>
static fir::ExtendedValue
genExtremumVal(FN func, FD funcDim, FC funcChar, mlir::Type resultType,
               fir::FirOpBuilder &builder, mlir::Location loc,
               Fortran::lower::StatementContext *stmtCtx,
               llvm::StringRef errMsg,
               llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 3);

  // Handle required array argument
  fir::BoxValue arryTmp = builder.createBox(loc, args[0]);
  mlir::Value array = fir::getBase(arryTmp);
  int rank = arryTmp.rank();
  assert(rank >= 1);
  bool hasCharacterResult = arryTmp.isCharacter();

  // Handle optional mask argument
  auto mask = isAbsent(args[2])
                  ? builder.create<fir::AbsentOp>(
                        loc, fir::BoxType::get(builder.getI1Type()))
                  : builder.createBox(loc, args[2]);

  bool absentDim = isAbsent(args[1]);

  // For Maxval/MinVal, we call the type specific versions of
  // Maxval/Minval because the result is scalar in the case below.
  if (!hasCharacterResult && (absentDim || rank == 1))
    return func(builder, loc, array, mask);

  if (hasCharacterResult && (absentDim || rank == 1)) {
    // Create mutable fir.box to be passed to the runtime for the result.
    auto resultMutableBox =
        fir::factory::createTempMutableBox(builder, loc, resultType);
    auto resultIrBox =
        fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

    funcChar(builder, loc, resultIrBox, array, mask);

    // Handle cleanup of allocatable result descriptor and return
    auto res = fir::factory::genMutableBoxRead(builder, loc, resultMutableBox);
    return res.match(
        [&](const fir::CharBoxValue &box) -> fir::ExtendedValue {
          // Add cleanup code
          assert(stmtCtx);
          auto *bldr = &builder;
          auto temp = box.getAddr();
          stmtCtx->attachCleanup(
              [=]() { bldr->create<fir::FreeMemOp>(loc, temp); });
          return box;
        },
        [&](const auto &) -> fir::ExtendedValue {
          fir::emitFatalError(loc, errMsg);
        });
  }

  // Handle Min/Maxval cases that have an array result.
  return genFuncDim(funcDim, resultType, builder, loc, stmtCtx, errMsg, array,
                    args[1], mask, rank);
}

/// Process calls to Minloc, Maxloc intrinsic functions
template <typename FN, typename FD>
static fir::ExtendedValue genExtremumloc(
    FN func, FD funcDim, mlir::Type resultType, fir::FirOpBuilder &builder,
    mlir::Location loc, Fortran::lower::StatementContext *stmtCtx,
    llvm::StringRef errMsg, llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 5);

  // Handle required array argument
  auto array = builder.createBox(loc, args[0]);
  fir::BoxValue arryTmp = builder.createBox(loc, args[0]);
  auto rank = arryTmp.rank();
  assert(rank >= 1);

  // Handle optional mask argument
  auto mask = isAbsent(args[2])
                  ? builder.create<fir::AbsentOp>(
                        loc, fir::BoxType::get(builder.getI1Type()))
                  : builder.createBox(loc, args[2]);

  // Handle optional kind argument
  auto kind = isAbsent(args[3]) ? builder.createIntegerConstant(
                                      loc, builder.getIndexType(),
                                      builder.getKindMap().defaultIntegerKind())
                                : fir::getBase(args[3]);

  // Handle optional back argument
  auto back = isAbsent(args[4]) ? builder.createBool(loc, false)
                                : fir::getBase(args[4]);

  bool absentDim = isAbsent(args[1]);

  if (!absentDim && rank == 1) {
    // If dim argument is present and the array is rank 1, then the result is
    // a scalar (since the the result is rank-1 or 0).
    // Therefore, we use a scalar result descriptor with Min/MaxlocDim().
    auto dim = fir::getBase(args[1]);
    // Create mutable fir.box to be passed to the runtime for the result.
    auto resultMutableBox =
        fir::factory::createTempMutableBox(builder, loc, resultType);
    auto resultIrBox =
        fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

    funcDim(builder, loc, resultIrBox, array, dim, mask, kind, back);

    // Handle cleanup of allocatable result descriptor and return
    auto res = fir::factory::genMutableBoxRead(builder, loc, resultMutableBox);
    return res.match(
        [&](const mlir::Value &tempAddr) -> fir::ExtendedValue {
          // Add cleanup code
          assert(stmtCtx);
          auto *bldr = &builder;
          stmtCtx->attachCleanup(
              [=]() { bldr->create<fir::FreeMemOp>(loc, tempAddr); });
          return builder.create<fir::LoadOp>(loc, resultType, tempAddr);
        },
        [&](const auto &) -> fir::ExtendedValue {
          fir::emitFatalError(loc, errMsg);
        });
  }

  // Note: The Min/Maxloc/val cases below have an array result.

  // Create mutable fir.box to be passed to the runtime for the result.
  auto resultArrayType =
      builder.getVarLenSeqTy(resultType, absentDim ? 1 : rank - 1);
  auto resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  auto resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  if (absentDim) {
    // Handle min/maxloc/val case where there is no dim argument
    // (calls Min/Maxloc()/MinMaxval() runtime routine)
    func(builder, loc, resultIrBox, array, mask, kind, back);
  } else {
    // else handle min/maxloc case with dim argument (calls
    // Min/Max/loc/val/Dim() runtime routine).
    auto dim = fir::getBase(args[1]);
    funcDim(builder, loc, resultIrBox, array, dim, mask, kind, back);
  }

  return fir::factory::genMutableBoxRead(builder, loc, resultMutableBox)
      .match(
          [&](const fir::ArrayBoxValue &box) -> fir::ExtendedValue {
            // Add cleanup code
            assert(stmtCtx);
            auto *bldr = &builder;
            auto temp = box.getAddr();
            stmtCtx->attachCleanup(
                [=]() { bldr->create<fir::FreeMemOp>(loc, temp); });
            return box;
          },
          [&](const auto &) -> fir::ExtendedValue {
            fir::emitFatalError(loc, errMsg);
          });
}

// TODO error handling -> return a code or directly emit messages ?
struct IntrinsicLibrary {

  // Constructors.
  explicit IntrinsicLibrary(fir::FirOpBuilder &builder, mlir::Location loc,
                            Fortran::lower::StatementContext *stmtCtx = nullptr)
      : builder{builder}, loc{loc}, stmtCtx{stmtCtx} {}
  IntrinsicLibrary() = delete;
  IntrinsicLibrary(const IntrinsicLibrary &) = delete;

  /// Generate FIR for call to Fortran intrinsic \p name with arguments \p arg
  /// and expected result type \p resultType.
  fir::ExtendedValue genIntrinsicCall(llvm::StringRef name,
                                      llvm::Optional<mlir::Type> resultType,
                                      llvm::ArrayRef<fir::ExtendedValue> arg);

  /// Search a runtime function that is associated to the generic intrinsic name
  /// and whose signature matches the intrinsic arguments and result types.
  /// If no such runtime function is found but a runtime function associated
  /// with the Fortran generic exists and has the same number of arguments,
  /// conversions will be inserted before and/or after the call. This is to
  /// mainly to allow 16 bits float support even-though little or no math
  /// runtime is currently available for it.
  mlir::Value genRuntimeCall(llvm::StringRef name, mlir::Type,
                             llvm::ArrayRef<mlir::Value>);

  using RuntimeCallGenerator = std::function<mlir::Value(
      fir::FirOpBuilder &, mlir::Location, llvm::ArrayRef<mlir::Value>)>;
  RuntimeCallGenerator
  getRuntimeCallGenerator(llvm::StringRef name,
                          mlir::FunctionType soughtFuncType);

  mlir::Value genAbs(mlir::Type, llvm::ArrayRef<mlir::Value>);

  template <void (*CallRuntime)(fir::FirOpBuilder &, mlir::Location loc,
                                mlir::Value, mlir::Value)>
  fir::ExtendedValue genAdjustRtCall(mlir::Type,
                                     llvm::ArrayRef<fir::ExtendedValue>);

  mlir::Value genAimag(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genAint(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genAll(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genAllocated(mlir::Type,
                                  llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genAnint(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genAny(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genAssociated(mlir::Type,
                                   llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genBtest(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genCeiling(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genChar(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  template <mlir::arith::CmpIPredicate pred>
  fir::ExtendedValue genCharacterCompare(mlir::Type,
                                         llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genConjg(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genCount(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  void genCpuTime(llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genCshift(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  void genDateAndTime(llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genDim(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genDotProduct(mlir::Type,
                                   llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genDprod(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genEoshift(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genExponent(mlir::Type, llvm::ArrayRef<mlir::Value>);
  template <Extremum, ExtremumBehavior>
  mlir::Value genExtremum(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genFloor(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genFraction(mlir::Type resultType,
                          mlir::ArrayRef<mlir::Value> args);
  mlir::Value genIand(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genIbclr(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genIbits(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genIbset(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genIchar(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genIeor(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genIndex(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genIor(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genIshft(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genIshftc(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genLen(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genLenTrim(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genMatmul(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genMaxloc(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genMaxval(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genMerge(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genMinloc(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genMinval(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genMod(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genModulo(mlir::Type, llvm::ArrayRef<mlir::Value>);
  void genMvbits(llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genNearest(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genNint(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genNot(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genNull(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genPack(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genPresent(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genProduct(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  void genRandomInit(llvm::ArrayRef<fir::ExtendedValue>);
  void genRandomNumber(llvm::ArrayRef<fir::ExtendedValue>);
  void genRandomSeed(llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genRepeat(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genReshape(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genRRSpacing(mlir::Type resultType,
                           llvm::ArrayRef<mlir::Value> args);
  mlir::Value genScale(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genScan(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genSetExponent(mlir::Type resultType,
                             llvm::ArrayRef<mlir::Value> args);
  mlir::Value genSign(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genSize(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genSpacing(mlir::Type resultType,
                         llvm::ArrayRef<mlir::Value> args);
  fir::ExtendedValue genSpread(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genSum(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  void genSystemClock(llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genTransfer(mlir::Type,
                                 llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genTranspose(mlir::Type,
                                  llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genTrim(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genUnpack(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genVerify(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  /// Implement all conversion functions like DBLE, the first argument is
  /// the value to convert. There may be an additional KIND arguments that
  /// is ignored because this is already reflected in the result type.
  mlir::Value genConversion(mlir::Type, llvm::ArrayRef<mlir::Value>);

  /// Define the different FIR generators that can be mapped to intrinsic to
  /// generate the related code.
  using ElementalGenerator = decltype(&IntrinsicLibrary::genAbs);
  using ExtendedGenerator = decltype(&IntrinsicLibrary::genLenTrim);
  using SubroutineGenerator = decltype(&IntrinsicLibrary::genDateAndTime);
  using Generator =
      std::variant<ElementalGenerator, ExtendedGenerator, SubroutineGenerator>;

  /// All generators can be outlined. This will build a function named
  /// "fir."+ <generic name> + "." + <result type code> and generate the
  /// intrinsic implementation inside instead of at the intrinsic call sites.
  /// This can be used to keep the FIR more readable. Only one function will
  /// be generated for all the similar calls in a program.
  /// If the Generator is nullptr, the wrapper uses genRuntimeCall.
  template <typename GeneratorType>
  mlir::Value outlineInWrapper(GeneratorType, llvm::StringRef name,
                               mlir::Type resultType,
                               llvm::ArrayRef<mlir::Value> args);
  template <typename GeneratorType>
  fir::ExtendedValue
  outlineInExtendedWrapper(GeneratorType, llvm::StringRef name,
                           llvm::Optional<mlir::Type> resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args);

  template <typename GeneratorType>
  mlir::FuncOp getWrapper(GeneratorType, llvm::StringRef name,
                          mlir::FunctionType, bool loadRefArguments = false);

  /// Generate calls to ElementalGenerator, handling the elemental aspects
  template <typename GeneratorType>
  fir::ExtendedValue
  genElementalCall(GeneratorType, llvm::StringRef name, mlir::Type resultType,
                   llvm::ArrayRef<fir::ExtendedValue> args, bool outline);

  /// Helper to invoke code generator for the intrinsics given arguments.
  mlir::Value invokeGenerator(ElementalGenerator generator,
                              mlir::Type resultType,
                              llvm::ArrayRef<mlir::Value> args);
  mlir::Value invokeGenerator(RuntimeCallGenerator generator,
                              mlir::Type resultType,
                              llvm::ArrayRef<mlir::Value> args);
  mlir::Value invokeGenerator(ExtendedGenerator generator,
                              mlir::Type resultType,
                              llvm::ArrayRef<mlir::Value> args);
  mlir::Value invokeGenerator(SubroutineGenerator generator,
                              llvm::ArrayRef<mlir::Value> args);

  /// Get pointer to unrestricted intrinsic. Generate the related unrestricted
  /// intrinsic if it is not defined yet.
  mlir::SymbolRefAttr
  getUnrestrictedIntrinsicSymbolRefAttr(llvm::StringRef name,
                                        mlir::FunctionType signature);

  /// Add clean-up for \p temp to the current statement context;
  void addCleanUpForTemp(mlir::Location loc, mlir::Value temp);
  /// Helper function for generating code clean-up for result descriptors
  fir::ExtendedValue readAndAddCleanUp(fir::MutableBoxValue resultMutableBox,
                                       mlir::Type resultType,
                                       llvm::StringRef errMsg);

  fir::FirOpBuilder &builder;
  mlir::Location loc;
  Fortran::lower::StatementContext *stmtCtx;
};

struct IntrinsicDummyArgument {
  const char *name = nullptr;
  Fortran::lower::LowerIntrinsicArgAs lowerAs =
      Fortran::lower::LowerIntrinsicArgAs::Value;
};
struct Fortran::lower::IntrinsicArgumentLoweringRules {
  /// There is no more than 7 non repeated arguments in Fortran intrinsics.
  IntrinsicDummyArgument args[7];
  constexpr bool hasDefaultRules() const { return args[0].name == nullptr; }
};

/// Structure describing what needs to be done to lower intrinsic "name".
struct IntrinsicHandler {
  const char *name;
  IntrinsicLibrary::Generator generator;
  Fortran::lower::IntrinsicArgumentLoweringRules argLoweringRules = {};
  bool isElemental = true;
  /// Code heavy intrinsic can be outlined to make FIR
  /// more readable.
  bool outline = false;
};

constexpr auto asValue = Fortran::lower::LowerIntrinsicArgAs::Value;
constexpr auto asAddr = Fortran::lower::LowerIntrinsicArgAs::Addr;
constexpr auto asBox = Fortran::lower::LowerIntrinsicArgAs::Box;
constexpr auto asInquired = Fortran::lower::LowerIntrinsicArgAs::Inquired;
using I = IntrinsicLibrary;

/// Table that drives the fir generation depending on the intrinsic.
/// one to one mapping with Fortran arguments. If no mapping is
/// defined here for a generic intrinsic, genRuntimeCall will be called
/// to look for a match in the runtime a emit a call. Note that the argument
/// lowering rules for an intrinsic need to be provided only if at least one
/// argument must not be lowered by value. In which case, the lowering rules
/// should be provided for all the intrinsic arguments for completeness.
static constexpr IntrinsicHandler handlers[]{
    {"abs", &I::genAbs},
    {"achar", &I::genChar},
    {"adjustl",
     &I::genAdjustRtCall<fir::runtime::genAdjustL>,
     {{{"string", asAddr}}},
     /*isElemental=*/true},
    {"adjustr",
     &I::genAdjustRtCall<fir::runtime::genAdjustR>,
     {{{"string", asAddr}}},
     /*isElemental=*/true},
    {"aimag", &I::genAimag},
    {"aint", &I::genAint},
    {"all",
     &I::genAll,
     {{{"mask", asAddr}, {"dim", asValue}}},
     /*isElemental=*/false},
    {"allocated",
     &I::genAllocated,
     {{{"array", asInquired}, {"scalar", asInquired}}},
     /*isElemental=*/false},
    {"anint", &I::genAnint},
    {"any",
     &I::genAny,
     {{{"mask", asAddr}, {"dim", asValue}}},
     /*isElemental=*/false},
    {"associated",
     &I::genAssociated,
     {{{"pointer", asInquired}, {"target", asBox}}},
     /*isElemental=*/false},
    {"btest", &I::genBtest},
    {"ceiling", &I::genCeiling},
    {"char", &I::genChar},
    {"conjg", &I::genConjg},
    {"count",
     &I::genCount,
     {{{"mask", asAddr}, {"dim", asValue}, {"kind", asValue}}},
     /*isElemental=*/false},
    {"cpu_time",
     &I::genCpuTime,
     {{{"time", asAddr}}},
     /*isElemental=*/false},
    {"cshift",
     &I::genCshift,
     {{{"array", asAddr}, {"shift", asAddr}, {"dim", asValue}}},
     /*isElemental=*/false},
    {"date_and_time",
     &I::genDateAndTime,
     {{{"date", asAddr},
       {"time", asAddr},
       {"zone", asAddr},
       {"values", asBox}}},
     /*isElemental=*/false},
    {"dble", &I::genConversion},
    {"dim", &I::genDim},
    {"dot_product",
     &I::genDotProduct,
     {{{"vector_a", asBox}, {"vector_b", asBox}}},
     /*isElemental=*/false},
    {"dprod", &I::genDprod},
    {"eoshift",
     &I::genEoshift,
     {{{"array", asAddr},
       {"shift", asAddr},
       {"boundary", asAddr},
       {"dim", asValue}}},
     /*isElemental=*/false},
    {"exponent", &I::genExponent},
    {"floor", &I::genFloor},
    {"fraction", &I::genFraction},
    {"iachar", &I::genIchar},
    {"iand", &I::genIand},
    {"ibclr", &I::genIbclr},
    {"ibits", &I::genIbits},
    {"ibset", &I::genIbset},
    {"ichar", &I::genIchar},
    {"ieor", &I::genIeor},
    {"index",
     &I::genIndex,
     {{{"string", asAddr},
       {"substring", asAddr},
       {"back", asValue},
       {"kind", asValue}}}},
    {"ior", &I::genIor},
    {"ishft", &I::genIshft},
    {"ishftc", &I::genIshftc},
    {"len", &I::genLen},
    {"len_trim", &I::genLenTrim},
    {"lge", &I::genCharacterCompare<mlir::arith::CmpIPredicate::sge>},
    {"lgt", &I::genCharacterCompare<mlir::arith::CmpIPredicate::sgt>},
    {"lle", &I::genCharacterCompare<mlir::arith::CmpIPredicate::sle>},
    {"llt", &I::genCharacterCompare<mlir::arith::CmpIPredicate::slt>},
    {"matmul",
     &I::genMatmul,
     {{{"matrix_a", asAddr}, {"matrix_b", asAddr}}},
     /*isElemental=*/false},
    {"max", &I::genExtremum<Extremum::Max, ExtremumBehavior::MinMaxss>},
    {"maxloc",
     &I::genMaxloc,
     {{{"array", asAddr},
       {"dim", asValue},
       {"mask", asAddr},
       {"kind", asValue},
       {"back", asValue}}},
     /*isElemental=*/false},
    {"maxval",
     &I::genMaxval,
     {{{"array", asAddr}, {"dim", asValue}, {"mask", asAddr}}},
     /*isElemental=*/false},
    {"merge", &I::genMerge},
    {"min", &I::genExtremum<Extremum::Min, ExtremumBehavior::MinMaxss>},
    {"minloc",
     &I::genMinloc,
     {{{"array", asAddr},
       {"dim", asValue},
       {"mask", asAddr},
       {"kind", asValue},
       {"back", asValue}}},
     /*isElemental=*/false},
    {"minval",
     &I::genMinval,
     {{{"array", asAddr}, {"dim", asValue}, {"mask", asAddr}}},
     /*isElemental=*/false},
    {"mod", &I::genMod},
    {"modulo", &I::genModulo},
    {"mvbits",
     &I::genMvbits,
     {{{"from", asValue},
       {"frompos", asValue},
       {"len", asValue},
       {"to", asAddr},
       {"topos", asValue}}}},
    {"nearest", &I::genNearest},
    {"nint", &I::genNint},
    {"not", &I::genNot},
    {"null", &I::genNull, {{{"mold", asInquired}}}, /*isElemental=*/false},
    {"pack",
     &I::genPack,
     {{{"array", asAddr}, {"mask", asAddr}, {"vector", asAddr}}},
     /*isElemental=*/false},
    {"present",
     &I::genPresent,
     {{{"a", asInquired}}},
     /*isElemental=*/false},
    {"product",
     &I::genProduct,
     {{{"array", asBox}, {"dim", asValue}, {"mask", asBox}}},
     /*isElemental=*/false},
    {"random_init",
     &I::genRandomInit,
     {{{"repeatable", asValue}, {"image_distinct", asValue}}},
     /*isElemental=*/false},
    {"random_number",
     &I::genRandomNumber,
     {{{"harvest", asBox}}},
     /*isElemental=*/false},
    {"random_seed",
     &I::genRandomSeed,
     {{{"size", asBox}, {"put", asBox}, {"get", asBox}}},
     /*isElemental=*/false},
    {"repeat",
     &I::genRepeat,
     {{{"string", asAddr}, {"ncopies", asValue}}},
     /*isElemental=*/false},
    {"reshape",
     &I::genReshape,
     {{{"source", asAddr},
       {"shape", asAddr},
       {"pad", asAddr},
       {"order", asAddr}}},
     /*isElemental=*/false},
    {"rrspacing", &I::genRRSpacing},
    {"scale",
     &I::genScale,
     {{{"x", asValue}, {"i", asValue}}},
     /*isElemental=*/true},
    {"scan",
     &I::genScan,
     {{{"string", asAddr},
       {"set", asAddr},
       {"back", asValue},
       {"kind", asValue}}},
     /*isElemental=*/true},
    {"set_exponent", &I::genSetExponent},
    {"sign", &I::genSign},
    {"size",
     &I::genSize,
     {{{"array", asAddr}, {"dim", asValue}, {"kind", asValue}}},
     /*isElemental=*/false},
    {"spacing", &I::genSpacing},
    {"spread",
     &I::genSpread,
     {{{"source", asAddr}, {"dim", asValue}, {"ncopies", asValue}}},
     /*isElemental=*/false},
    {"sum",
     &I::genSum,
     {{{"array", asBox}, {"dim", asValue}, {"mask", asBox}}},
     /*isElemental=*/false},
    {"system_clock",
     &I::genSystemClock,
     {{{"count", asAddr}, {"count_rate", asAddr}, {"count_max", asAddr}}},
     /*isElemental=*/false},
    {"transfer",
     &I::genTransfer,
     {{{"source", asAddr}, {"mold", asAddr}, {"size", asValue}}},
     /*isElemental=*/false},
    {"transpose",
     &I::genTranspose,
     {{{"matrix", asAddr}}},
     /*isElemental=*/false},
    {"trim", &I::genTrim, {{{"string", asAddr}}}, /*isElemental=*/false},
    {"unpack",
     &I::genUnpack,
     {{{"vector", asAddr}, {"mask", asAddr}, {"field", asAddr}}},
     /*isElemental=*/false},
    {"verify",
     &I::genVerify,
     {{{"string", asAddr},
       {"set", asAddr},
       {"back", asValue},
       {"kind", asValue}}},
     /*isElemental=*/true},
};

static const IntrinsicHandler *findIntrinsicHandler(llvm::StringRef name) {
  auto compare = [](const IntrinsicHandler &handler, llvm::StringRef name) {
    return name.compare(handler.name) > 0;
  };
  auto result =
      std::lower_bound(std::begin(handlers), std::end(handlers), name, compare);
  return result != std::end(handlers) && result->name == name ? result
                                                              : nullptr;
}

/// To make fir output more readable for debug, one can outline all intrinsic
/// implementation in wrappers (overrides the IntrinsicHandler::outline flag).
static llvm::cl::opt<bool> outlineAllIntrinsics(
    "outline-intrinsics",
    llvm::cl::desc(
        "Lower all intrinsic procedure implementation in their own functions"),
    llvm::cl::init(false));

//===----------------------------------------------------------------------===//
// Math runtime description and matching utility
//===----------------------------------------------------------------------===//

/// Command line option to modify math runtime version used to implement
/// intrinsics.
enum MathRuntimeVersion {
  fastVersion,
  relaxedVersion,
  preciseVersion,
  llvmOnly
};
llvm::cl::opt<MathRuntimeVersion> mathRuntimeVersion(
    "math-runtime", llvm::cl::desc("Select math runtime version:"),
    llvm::cl::values(
        clEnumValN(fastVersion, "fast", "use pgmath fast runtime"),
        clEnumValN(relaxedVersion, "relaxed", "use pgmath relaxed runtime"),
        clEnumValN(preciseVersion, "precise", "use pgmath precise runtime"),
        clEnumValN(llvmOnly, "llvm",
                   "only use LLVM intrinsics (may be incomplete)")),
    llvm::cl::init(fastVersion));

struct RuntimeFunction {
  // llvm::StringRef comparison operator are not constexpr, so use string_view.
  using Key = std::string_view;
  // Needed for implicit compare with keys.
  constexpr operator Key() const { return key; }
  Key key; // intrinsic name
  llvm::StringRef symbol;
  fir::runtime::FuncTypeBuilderFunc typeGenerator;
};

#define RUNTIME_STATIC_DESCRIPTION(name, func)                                 \
  {#name, #func, fir::runtime::RuntimeTableKey<decltype(func)>::getTypeModel()},
static constexpr RuntimeFunction pgmathFast[] = {
#define PGMATH_FAST
#define PGMATH_USE_ALL_TYPES(name, func) RUNTIME_STATIC_DESCRIPTION(name, func)
#include "flang/Evaluate/pgmath.h.inc"
};
static constexpr RuntimeFunction pgmathRelaxed[] = {
#define PGMATH_RELAXED
#define PGMATH_USE_ALL_TYPES(name, func) RUNTIME_STATIC_DESCRIPTION(name, func)
#include "flang/Evaluate/pgmath.h.inc"
};
static constexpr RuntimeFunction pgmathPrecise[] = {
#define PGMATH_PRECISE
#define PGMATH_USE_ALL_TYPES(name, func) RUNTIME_STATIC_DESCRIPTION(name, func)
#include "flang/Evaluate/pgmath.h.inc"
};

static mlir::FunctionType genF32F32FuncType(mlir::MLIRContext *context) {
  auto t = mlir::FloatType::getF32(context);
  return mlir::FunctionType::get(context, {t}, {t});
}

static mlir::FunctionType genF64F64FuncType(mlir::MLIRContext *context) {
  auto t = mlir::FloatType::getF64(context);
  return mlir::FunctionType::get(context, {t}, {t});
}

static mlir::FunctionType genF32F32F32FuncType(mlir::MLIRContext *context) {
  auto t = mlir::FloatType::getF32(context);
  return mlir::FunctionType::get(context, {t, t}, {t});
}

static mlir::FunctionType genF64F64F64FuncType(mlir::MLIRContext *context) {
  auto t = mlir::FloatType::getF64(context);
  return mlir::FunctionType::get(context, {t, t}, {t});
}

template <int Bits>
static mlir::FunctionType genIntF64FuncType(mlir::MLIRContext *context) {
  auto t = mlir::FloatType::getF64(context);
  auto r = mlir::IntegerType::get(context, Bits);
  return mlir::FunctionType::get(context, {t}, {r});
}

template <int Bits>
static mlir::FunctionType genIntF32FuncType(mlir::MLIRContext *context) {
  auto t = mlir::FloatType::getF32(context);
  auto r = mlir::IntegerType::get(context, Bits);
  return mlir::FunctionType::get(context, {t}, {r});
}

// TODO : Fill-up this table with more intrinsic.
// Note: These are also defined as operations in LLVM dialect. See if this
// can be use and has advantages.
static constexpr RuntimeFunction llvmIntrinsics[] = {
    {"abs", "llvm.fabs.f32", genF32F32FuncType},
    {"abs", "llvm.fabs.f64", genF64F64FuncType},
    {"aint", "llvm.trunc.f32", genF32F32FuncType},
    {"aint", "llvm.trunc.f64", genF64F64FuncType},
    {"anint", "llvm.round.f32", genF32F32FuncType},
    {"anint", "llvm.round.f64", genF64F64FuncType},
    {"atan", "atanf", genF32F32FuncType},
    {"atan", "atan", genF64F64FuncType},
    // ceil is used for CEILING but is different, it returns a real.
    {"ceil", "llvm.ceil.f32", genF32F32FuncType},
    {"ceil", "llvm.ceil.f64", genF64F64FuncType},
    {"cos", "llvm.cos.f32", genF32F32FuncType},
    {"cos", "llvm.cos.f64", genF64F64FuncType},
    {"cosh", "coshf", genF32F32FuncType},
    {"cosh", "cosh", genF64F64FuncType},
    {"exp", "llvm.exp.f32", genF32F32FuncType},
    {"exp", "llvm.exp.f64", genF64F64FuncType},
    // llvm.floor is used for FLOOR, but returns real.
    {"floor", "llvm.floor.f32", genF32F32FuncType},
    {"floor", "llvm.floor.f64", genF64F64FuncType},
    {"log", "llvm.log.f32", genF32F32FuncType},
    {"log", "llvm.log.f64", genF64F64FuncType},
    {"log10", "llvm.log10.f32", genF32F32FuncType},
    {"log10", "llvm.log10.f64", genF64F64FuncType},
    {"nint", "llvm.lround.i64.f64", genIntF64FuncType<64>},
    {"nint", "llvm.lround.i64.f32", genIntF32FuncType<64>},
    {"nint", "llvm.lround.i32.f64", genIntF64FuncType<32>},
    {"nint", "llvm.lround.i32.f32", genIntF32FuncType<32>},
    {"pow", "llvm.pow.f32", genF32F32F32FuncType},
    {"pow", "llvm.pow.f64", genF64F64F64FuncType},
    {"sin", "llvm.sin.f32", genF32F32FuncType},
    {"sin", "llvm.sin.f64", genF64F64FuncType},
    {"sinh", "sinhf", genF32F32FuncType},
    {"sinh", "sinh", genF64F64FuncType},
    {"sqrt", "llvm.sqrt.f32", genF32F32FuncType},
    {"sqrt", "llvm.sqrt.f64", genF64F64FuncType},
};

// This helper class computes a "distance" between two function types.
// The distance measures how many narrowing conversions of actual arguments
// and result of "from" must be made in order to use "to" instead of "from".
// For instance, the distance between ACOS(REAL(10)) and ACOS(REAL(8)) is
// greater than the one between ACOS(REAL(10)) and ACOS(REAL(16)). This means
// if no implementation of ACOS(REAL(10)) is available, it is better to use
// ACOS(REAL(16)) with casts rather than ACOS(REAL(8)).
// Note that this is not a symmetric distance and the order of "from" and "to"
// arguments matters, d(foo, bar) may not be the same as d(bar, foo) because it
// may be safe to replace foo by bar, but not the opposite.
class FunctionDistance {
public:
  FunctionDistance() : infinite{true} {}

  FunctionDistance(mlir::FunctionType from, mlir::FunctionType to) {
    auto nInputs = from.getNumInputs();
    auto nResults = from.getNumResults();
    if (nResults != to.getNumResults() || nInputs != to.getNumInputs()) {
      infinite = true;
    } else {
      for (decltype(nInputs) i{0}; i < nInputs && !infinite; ++i)
        addArgumentDistance(from.getInput(i), to.getInput(i));
      for (decltype(nResults) i{0}; i < nResults && !infinite; ++i)
        addResultDistance(to.getResult(i), from.getResult(i));
    }
  }

  /// Beware both d1.isSmallerThan(d2) *and* d2.isSmallerThan(d1) may be
  /// false if both d1 and d2 are infinite. This implies that
  ///  d1.isSmallerThan(d2) is not equivalent to !d2.isSmallerThan(d1)
  bool isSmallerThan(const FunctionDistance &d) const {
    return !infinite &&
           (d.infinite || std::lexicographical_compare(
                              conversions.begin(), conversions.end(),
                              d.conversions.begin(), d.conversions.end()));
  }

  bool isLosingPrecision() const {
    return conversions[narrowingArg] != 0 || conversions[extendingResult] != 0;
  }

  bool isInfinite() const { return infinite; }

private:
  enum class Conversion { Forbidden, None, Narrow, Extend };

  void addArgumentDistance(mlir::Type from, mlir::Type to) {
    switch (conversionBetweenTypes(from, to)) {
    case Conversion::Forbidden:
      infinite = true;
      break;
    case Conversion::None:
      break;
    case Conversion::Narrow:
      conversions[narrowingArg]++;
      break;
    case Conversion::Extend:
      conversions[nonNarrowingArg]++;
      break;
    }
  }

  void addResultDistance(mlir::Type from, mlir::Type to) {
    switch (conversionBetweenTypes(from, to)) {
    case Conversion::Forbidden:
      infinite = true;
      break;
    case Conversion::None:
      break;
    case Conversion::Narrow:
      conversions[nonExtendingResult]++;
      break;
    case Conversion::Extend:
      conversions[extendingResult]++;
      break;
    }
  }

  // Floating point can be mlir::FloatType or fir::real
  static unsigned getFloatingPointWidth(mlir::Type t) {
    if (auto f{t.dyn_cast<mlir::FloatType>()})
      return f.getWidth();
    // FIXME: Get width another way for fir.real/complex
    // - use fir/KindMapping.h and llvm::Type
    // - or use evaluate/type.h
    if (auto r{t.dyn_cast<fir::RealType>()})
      return r.getFKind() * 4;
    if (auto cplx{t.dyn_cast<fir::ComplexType>()})
      return cplx.getFKind() * 4;
    llvm_unreachable("not a floating-point type");
  }

  static Conversion conversionBetweenTypes(mlir::Type from, mlir::Type to) {
    if (from == to) {
      return Conversion::None;
    }
    if (auto fromIntTy{from.dyn_cast<mlir::IntegerType>()}) {
      if (auto toIntTy{to.dyn_cast<mlir::IntegerType>()}) {
        return fromIntTy.getWidth() > toIntTy.getWidth() ? Conversion::Narrow
                                                         : Conversion::Extend;
      }
    }
    if (fir::isa_real(from) && fir::isa_real(to)) {
      return getFloatingPointWidth(from) > getFloatingPointWidth(to)
                 ? Conversion::Narrow
                 : Conversion::Extend;
    }
    if (auto fromCplxTy{from.dyn_cast<fir::ComplexType>()}) {
      if (auto toCplxTy{to.dyn_cast<fir::ComplexType>()}) {
        return getFloatingPointWidth(fromCplxTy) >
                       getFloatingPointWidth(toCplxTy)
                   ? Conversion::Narrow
                   : Conversion::Extend;
      }
    }
    // Notes:
    // - No conversion between character types, specialization of runtime
    // functions should be made instead.
    // - It is not clear there is a use case for automatic conversions
    // around Logical and it may damage hidden information in the physical
    // storage so do not do it.
    return Conversion::Forbidden;
  }

  // Below are indexes to access data in conversions.
  // The order in data does matter for lexicographical_compare
  enum {
    narrowingArg = 0,   // usually bad
    extendingResult,    // usually bad
    nonExtendingResult, // usually ok
    nonNarrowingArg,    // usually ok
    dataSize
  };

  std::array<int, dataSize> conversions{/* zero init*/};
  bool infinite{false}; // When forbidden conversion or wrong argument number
};

/// Build mlir::FuncOp from runtime symbol description and add
/// fir.runtime attribute.
static mlir::FuncOp getFuncOp(mlir::Location loc, fir::FirOpBuilder &builder,
                              const RuntimeFunction &runtime) {
  auto function = builder.addNamedFunction(
      loc, runtime.symbol, runtime.typeGenerator(builder.getContext()));
  function->setAttr("fir.runtime", builder.getUnitAttr());
  return function;
}

/// Select runtime function that has the smallest distance to the intrinsic
/// function type and that will not imply narrowing arguments or extending the
/// result.
/// If nothing is found, the mlir::FuncOp will contain a nullptr.
mlir::FuncOp searchFunctionInLibrary(
    mlir::Location loc, fir::FirOpBuilder &builder,
    const Fortran::common::StaticMultimapView<RuntimeFunction> &lib,
    llvm::StringRef name, mlir::FunctionType funcType,
    const RuntimeFunction **bestNearMatch,
    FunctionDistance &bestMatchDistance) {
  auto range = lib.equal_range(name);
  for (auto iter{range.first}; iter != range.second && iter; ++iter) {
    const auto &impl = *iter;
    auto implType = impl.typeGenerator(builder.getContext());
    if (funcType == implType) {
      return getFuncOp(loc, builder, impl); // exact match
    } else {
      FunctionDistance distance(funcType, implType);
      if (distance.isSmallerThan(bestMatchDistance)) {
        *bestNearMatch = &impl;
        bestMatchDistance = std::move(distance);
      }
    }
  }
  return {};
}

/// Search runtime for the best runtime function given an intrinsic name
/// and interface. The interface may not be a perfect match in which case
/// the caller is responsible to insert argument and return value conversions.
/// If nothing is found, the mlir::FuncOp will contain a nullptr.
static mlir::FuncOp getRuntimeFunction(mlir::Location loc,
                                       fir::FirOpBuilder &builder,
                                       llvm::StringRef name,
                                       mlir::FunctionType funcType) {
  const RuntimeFunction *bestNearMatch = nullptr;
  FunctionDistance bestMatchDistance{};
  mlir::FuncOp match;
  using RtMap = Fortran::common::StaticMultimapView<RuntimeFunction>;
  static constexpr RtMap pgmathF(pgmathFast);
  static_assert(pgmathF.Verify() && "map must be sorted");
  static constexpr RtMap pgmathR(pgmathRelaxed);
  static_assert(pgmathR.Verify() && "map must be sorted");
  static constexpr RtMap pgmathP(pgmathPrecise);
  static_assert(pgmathP.Verify() && "map must be sorted");
  if (mathRuntimeVersion == fastVersion) {
    match = searchFunctionInLibrary(loc, builder, pgmathF, name, funcType,
                                    &bestNearMatch, bestMatchDistance);
  } else if (mathRuntimeVersion == relaxedVersion) {
    match = searchFunctionInLibrary(loc, builder, pgmathR, name, funcType,
                                    &bestNearMatch, bestMatchDistance);
  } else if (mathRuntimeVersion == preciseVersion) {
    match = searchFunctionInLibrary(loc, builder, pgmathP, name, funcType,
                                    &bestNearMatch, bestMatchDistance);
  } else {
    assert(mathRuntimeVersion == llvmOnly && "unknown math runtime");
  }
  if (match)
    return match;

  // Go through llvm intrinsics if not exact match in libpgmath or if
  // mathRuntimeVersion == llvmOnly
  static constexpr RtMap llvmIntr(llvmIntrinsics);
  static_assert(llvmIntr.Verify() && "map must be sorted");
  if (auto exactMatch =
          searchFunctionInLibrary(loc, builder, llvmIntr, name, funcType,
                                  &bestNearMatch, bestMatchDistance))
    return exactMatch;

  if (bestNearMatch != nullptr) {
    assert(!bestMatchDistance.isLosingPrecision() &&
           "runtime selection loses precision");
    return getFuncOp(loc, builder, *bestNearMatch);
  }
  return {};
}

/// Helpers to get function type from arguments and result type.
static mlir::FunctionType getFunctionType(llvm::Optional<mlir::Type> resultType,
                                          llvm::ArrayRef<mlir::Value> arguments,
                                          fir::FirOpBuilder &builder) {
  llvm::SmallVector<mlir::Type> argTypes;
  for (auto &arg : arguments)
    argTypes.push_back(arg.getType());
  llvm::SmallVector<mlir::Type> resTypes;
  if (resultType)
    resTypes.push_back(*resultType);
  return mlir::FunctionType::get(builder.getModule().getContext(), argTypes,
                                 resTypes);
}

/// fir::ExtendedValue to mlir::Value translation layer

fir::ExtendedValue toExtendedValue(mlir::Value val, fir::FirOpBuilder &builder,
                                   mlir::Location loc) {
  assert(val && "optional unhandled here");
  auto type = val.getType();
  auto base = val;
  auto indexType = builder.getIndexType();
  llvm::SmallVector<mlir::Value> extents;

  fir::factory::CharacterExprHelper charHelper{builder, loc};
  // FIXME: we may want to allow non character scalar here.
  if (charHelper.isCharacterScalar(type))
    return charHelper.toExtendedValue(val);

  if (auto refType = type.dyn_cast<fir::ReferenceType>())
    type = refType.getEleTy();

  if (auto arrayType = type.dyn_cast<fir::SequenceType>()) {
    type = arrayType.getEleTy();
    for (auto extent : arrayType.getShape()) {
      if (extent == fir::SequenceType::getUnknownExtent())
        break;
      extents.emplace_back(
          builder.createIntegerConstant(loc, indexType, extent));
    }
    // Last extent might be missing in case of assumed-size. If more extents
    // could not be deduced from type, that's an error (a fir.box should
    // have been used in the interface).
    if (extents.size() + 1 < arrayType.getShape().size())
      mlir::emitError(loc, "cannot retrieve array extents from type");
  } else if (type.isa<fir::BoxType>() || type.isa<fir::RecordType>()) {
    mlir::emitError(loc, "descriptor or derived type not yet handled");
  }

  if (!extents.empty())
    return fir::ArrayBoxValue{base, extents};
  return base;
}

mlir::Value toValue(const fir::ExtendedValue &val, fir::FirOpBuilder &builder,
                    mlir::Location loc) {
  if (auto charBox = val.getCharBox()) {
    auto buffer = charBox->getBuffer();
    if (buffer.getType().isa<fir::BoxCharType>())
      return buffer;
    return fir::factory::CharacterExprHelper{builder, loc}.createEmboxChar(
        buffer, charBox->getLen());
  }

  // FIXME: need to access other ExtendedValue variants and handle them
  // properly.
  return fir::getBase(val);
}

//===----------------------------------------------------------------------===//
// IntrinsicLibrary
//===----------------------------------------------------------------------===//

/// Emit a TODO error message for as yet unimplemented intrinsics.
static void crashOnMissingIntrinsic(mlir::Location loc, llvm::StringRef name) {
  mlir::emitError(loc,
                  "TODO: missing intrinsic lowering: " + llvm::Twine(name));
  exit(1);
}

template <typename GeneratorType>
fir::ExtendedValue IntrinsicLibrary::genElementalCall(
    GeneratorType generator, llvm::StringRef name, mlir::Type resultType,
    llvm::ArrayRef<fir::ExtendedValue> args, bool outline) {
  llvm::SmallVector<mlir::Value> scalarArgs;
  for (const auto &arg : args)
    if (arg.getUnboxed() || arg.getCharBox())
      scalarArgs.emplace_back(fir::getBase(arg));
    else
      fir::emitFatalError(loc, "nonscalar intrinsic argument");
  if (outline)
    return outlineInWrapper(generator, name, resultType, scalarArgs);
  return invokeGenerator(generator, resultType, scalarArgs);
}

template <>
fir::ExtendedValue
IntrinsicLibrary::genElementalCall<IntrinsicLibrary::ExtendedGenerator>(
    ExtendedGenerator generator, llvm::StringRef name, mlir::Type resultType,
    llvm::ArrayRef<fir::ExtendedValue> args, bool outline) {
  for (const auto &arg : args)
    if (!arg.getUnboxed() && !arg.getCharBox())
      fir::emitFatalError(loc, "nonscalar intrinsic argument");
  if (outline)
    return outlineInExtendedWrapper(generator, name, resultType, args);
  return std::invoke(generator, *this, resultType, args);
}

template <>
fir::ExtendedValue
IntrinsicLibrary::genElementalCall<IntrinsicLibrary::SubroutineGenerator>(
    SubroutineGenerator generator, llvm::StringRef name, mlir::Type resultType,
    llvm::ArrayRef<fir::ExtendedValue> args, bool outline) {
  for (const auto &arg : args)
    if (!arg.getUnboxed() && !arg.getCharBox())
      // fir::emitFatalError(loc, "nonscalar intrinsic argument");
      crashOnMissingIntrinsic(loc, name);
  if (outline)
    return outlineInExtendedWrapper(generator, name, resultType, args);
  std::invoke(generator, *this, args);
  return mlir::Value();
}

static fir::ExtendedValue
invokeHandler(IntrinsicLibrary::ElementalGenerator generator,
              const IntrinsicHandler &handler,
              llvm::Optional<mlir::Type> resultType,
              llvm::ArrayRef<fir::ExtendedValue> args, bool outline,
              IntrinsicLibrary &lib) {
  assert(resultType && "expect elemental intrinsic to be functions");
  return lib.genElementalCall(generator, handler.name, *resultType, args,
                              outline);
}

static fir::ExtendedValue
invokeHandler(IntrinsicLibrary::ExtendedGenerator generator,
              const IntrinsicHandler &handler,
              llvm::Optional<mlir::Type> resultType,
              llvm::ArrayRef<fir::ExtendedValue> args, bool outline,
              IntrinsicLibrary &lib) {
  assert(resultType && "expect intrinsic function");
  if (handler.isElemental)
    return lib.genElementalCall(generator, handler.name, *resultType, args,
                                outline);
  if (outline)
    return lib.outlineInExtendedWrapper(generator, handler.name, *resultType,
                                        args);
  return std::invoke(generator, lib, *resultType, args);
}
static fir::ExtendedValue
invokeHandler(IntrinsicLibrary::SubroutineGenerator generator,
              const IntrinsicHandler &handler,
              llvm::Optional<mlir::Type> resultType,
              llvm::ArrayRef<fir::ExtendedValue> args, bool outline,
              IntrinsicLibrary &lib) {
  if (handler.isElemental)
    return lib.genElementalCall(generator, handler.name, mlir::Type{}, args,
                                outline);
  if (outline)
    return lib.outlineInExtendedWrapper(generator, handler.name, resultType,
                                        args);
  std::invoke(generator, lib, args);
  return mlir::Value{};
}

fir::ExtendedValue
IntrinsicLibrary::genIntrinsicCall(llvm::StringRef name,
                                   llvm::Optional<mlir::Type> resultType,
                                   llvm::ArrayRef<fir::ExtendedValue> args) {
  if (const auto &handler = findIntrinsicHandler(name)) {
    bool outline = handler->outline || outlineAllIntrinsics;
    return std::visit(
        [&](auto &generator) -> fir::ExtendedValue {
          return invokeHandler(generator, *handler, resultType, args, outline,
                               *this);
        },
        handler->generator);
  }

  if (!resultType)
    // Subroutine should have a handler, they are likely missing for now.
    crashOnMissingIntrinsic(loc, name);

  // Try the runtime if no special handler was defined for the
  // intrinsic being called. Maths runtime only has numerical elemental.
  // No optional arguments are expected at this point, the code will
  // crash if it gets absent optional.

  // FIXME: using toValue to get the type won't work with array arguments.
  llvm::SmallVector<mlir::Value> mlirArgs;
  for (const auto &extendedVal : args) {
    auto val = toValue(extendedVal, builder, loc);
    if (!val)
      // If an absent optional gets there, most likely its handler has just
      // not yet been defined.
      crashOnMissingIntrinsic(loc, name);
    mlirArgs.emplace_back(val);
  }
  mlir::FunctionType soughtFuncType =
      getFunctionType(*resultType, mlirArgs, builder);

  auto runtimeCallGenerator = getRuntimeCallGenerator(name, soughtFuncType);
  return genElementalCall(runtimeCallGenerator, name, *resultType, args,
                          /* outline */ true);
}

mlir::Value
IntrinsicLibrary::invokeGenerator(ElementalGenerator generator,
                                  mlir::Type resultType,
                                  llvm::ArrayRef<mlir::Value> args) {
  return std::invoke(generator, *this, resultType, args);
}

mlir::Value
IntrinsicLibrary::invokeGenerator(RuntimeCallGenerator generator,
                                  mlir::Type resultType,
                                  llvm::ArrayRef<mlir::Value> args) {
  return generator(builder, loc, args);
}

mlir::Value
IntrinsicLibrary::invokeGenerator(ExtendedGenerator generator,
                                  mlir::Type resultType,
                                  llvm::ArrayRef<mlir::Value> args) {
  llvm::SmallVector<fir::ExtendedValue> extendedArgs;
  for (auto arg : args)
    extendedArgs.emplace_back(toExtendedValue(arg, builder, loc));
  auto extendedResult = std::invoke(generator, *this, resultType, extendedArgs);
  return toValue(extendedResult, builder, loc);
}

mlir::Value
IntrinsicLibrary::invokeGenerator(SubroutineGenerator generator,
                                  llvm::ArrayRef<mlir::Value> args) {
  llvm::SmallVector<fir::ExtendedValue> extendedArgs;
  for (auto arg : args)
    extendedArgs.emplace_back(toExtendedValue(arg, builder, loc));
  std::invoke(generator, *this, extendedArgs);
  return mlir::Value{};
}

template <typename GeneratorType>
mlir::FuncOp IntrinsicLibrary::getWrapper(GeneratorType generator,
                                          llvm::StringRef name,
                                          mlir::FunctionType funcType,
                                          bool loadRefArguments) {
  std::string wrapperName = fir::mangleIntrinsicProcedure(name, funcType);
  auto function = builder.getNamedFunction(wrapperName);
  if (!function) {
    // First time this wrapper is needed, build it.
    function = builder.createFunction(loc, wrapperName, funcType);
    function->setAttr("fir.intrinsic", builder.getUnitAttr());
    auto internalLinkage = mlir::LLVM::linkage::Linkage::Internal;
    auto linkage =
        mlir::LLVM::LinkageAttr::get(builder.getContext(), internalLinkage);
    function->setAttr("llvm.linkage", linkage);
    function.addEntryBlock();

    // Create local context to emit code into the newly created function
    // This new function is not linked to a source file location, only
    // its calls will be.
    auto localBuilder =
        std::make_unique<fir::FirOpBuilder>(function, builder.getKindMap());
    localBuilder->setInsertionPointToStart(&function.front());
    // Location of code inside wrapper of the wrapper is independent from
    // the location of the intrinsic call.
    auto localLoc = localBuilder->getUnknownLoc();
    llvm::SmallVector<mlir::Value> localArguments;
    for (mlir::BlockArgument bArg : function.front().getArguments()) {
      auto refType = bArg.getType().dyn_cast<fir::ReferenceType>();
      if (loadRefArguments && refType) {
        auto loaded = localBuilder->create<fir::LoadOp>(localLoc, bArg);
        localArguments.push_back(loaded);
      } else {
        localArguments.push_back(bArg);
      }
    }

    IntrinsicLibrary localLib{*localBuilder, localLoc};

    if constexpr (std::is_same_v<GeneratorType, SubroutineGenerator>) {
      localLib.invokeGenerator(generator, localArguments);
      localBuilder->create<mlir::ReturnOp>(localLoc);
    } else {
      assert(funcType.getNumResults() == 1 &&
             "expect one result for intrinsic function wrapper type");
      auto resultType = funcType.getResult(0);
      auto result =
          localLib.invokeGenerator(generator, resultType, localArguments);
      localBuilder->create<mlir::ReturnOp>(localLoc, result);
    }
  } else {
    // Wrapper was already built, ensure it has the sought type
    assert(function.getType() == funcType &&
           "conflict between intrinsic wrapper types");
  }
  return function;
}

/// Helpers to detect absent optional (not yet supported in outlining).
bool static hasAbsentOptional(llvm::ArrayRef<mlir::Value> args) {
  for (const auto &arg : args)
    if (!arg)
      return true;
  return false;
}
bool static hasAbsentOptional(llvm::ArrayRef<fir::ExtendedValue> args) {
  for (const auto &arg : args)
    if (!fir::getBase(arg))
      return true;
  return false;
}

template <typename GeneratorType>
mlir::Value
IntrinsicLibrary::outlineInWrapper(GeneratorType generator,
                                   llvm::StringRef name, mlir::Type resultType,
                                   llvm::ArrayRef<mlir::Value> args) {
  if (hasAbsentOptional(args)) {
    // TODO: absent optional in outlining is an issue: we cannot just ignore
    // them. Needs a better interface here. The issue is that we cannot easily
    // tell that a value is optional or not here if it is presents. And if it is
    // absent, we cannot tell what it type should be.
    mlir::emitError(loc, "todo: cannot outline call to intrinsic " +
                             llvm::Twine(name) +
                             " with absent optional argument");
    exit(1);
  }

  auto funcType = getFunctionType(resultType, args, builder);
  auto wrapper = getWrapper(generator, name, funcType);
  return builder.create<fir::CallOp>(loc, wrapper, args).getResult(0);
}

template <typename GeneratorType>
fir::ExtendedValue IntrinsicLibrary::outlineInExtendedWrapper(
    GeneratorType generator, llvm::StringRef name,
    llvm::Optional<mlir::Type> resultType,
    llvm::ArrayRef<fir::ExtendedValue> args) {
  if (hasAbsentOptional(args)) {
    // TODO
    mlir::emitError(loc, "todo: cannot outline call to intrinsic " +
                             llvm::Twine(name) +
                             " with absent optional argument");
    exit(1);
  }
  llvm::SmallVector<mlir::Value> mlirArgs;
  for (const auto &extendedVal : args)
    mlirArgs.emplace_back(toValue(extendedVal, builder, loc));
  auto funcType = getFunctionType(resultType, mlirArgs, builder);
  auto wrapper = getWrapper(generator, name, funcType);
  auto call = builder.create<fir::CallOp>(loc, wrapper, mlirArgs);
  if (resultType)
    return toExtendedValue(call.getResult(0), builder, loc);
  // Subroutine calls
  return mlir::Value{};
}

IntrinsicLibrary::RuntimeCallGenerator
IntrinsicLibrary::getRuntimeCallGenerator(llvm::StringRef name,
                                          mlir::FunctionType soughtFuncType) {
  auto funcOp = getRuntimeFunction(loc, builder, name, soughtFuncType);
  if (!funcOp) {
    mlir::emitError(loc,
                    "TODO: missing intrinsic lowering: " + llvm::Twine(name));
    llvm::errs() << "requested type was: " << soughtFuncType << "\n";
    exit(1);
  }

  mlir::FunctionType actualFuncType = funcOp.getType();
  assert(actualFuncType.getNumResults() == soughtFuncType.getNumResults() &&
         actualFuncType.getNumInputs() == soughtFuncType.getNumInputs() &&
         actualFuncType.getNumResults() == 1 && "Bad intrinsic match");

  return [funcOp, actualFuncType,
          soughtFuncType](fir::FirOpBuilder &builder, mlir::Location loc,
                          llvm::ArrayRef<mlir::Value> args) {
    llvm::SmallVector<mlir::Value> convertedArguments;
    for (auto [fst, snd] : llvm::zip(actualFuncType.getInputs(), args))
      convertedArguments.push_back(builder.createConvert(loc, fst, snd));
    auto call = builder.create<fir::CallOp>(loc, funcOp, convertedArguments);
    mlir::Type soughtType = soughtFuncType.getResult(0);
    return builder.createConvert(loc, soughtType, call.getResult(0));
  };
}

mlir::SymbolRefAttr IntrinsicLibrary::getUnrestrictedIntrinsicSymbolRefAttr(
    llvm::StringRef name, mlir::FunctionType signature) {
  // Unrestricted intrinsics signature follows implicit rules: argument
  // are passed by references. But the runtime versions expect values.
  // So instead of duplicating the runtime, just have the wrappers loading
  // this before calling the code generators.
  bool loadRefArguments = true;
  mlir::FuncOp funcOp;
  if (const auto &handler = findIntrinsicHandler(name))
    funcOp = std::visit(
        [&](auto generator) {
          return getWrapper(generator, name, signature, loadRefArguments);
        },
        handler->generator);

  if (!funcOp) {
    llvm::SmallVector<mlir::Type> argTypes;
    for (auto type : signature.getInputs()) {
      if (auto refType = type.dyn_cast<fir::ReferenceType>())
        argTypes.push_back(refType.getEleTy());
      else
        argTypes.push_back(type);
    }
    auto soughtFuncType =
        builder.getFunctionType(argTypes, signature.getResults());
    auto rtCallGenerator = getRuntimeCallGenerator(name, soughtFuncType);
    funcOp = getWrapper(rtCallGenerator, name, signature, loadRefArguments);
  }

  return SymbolRefAttr::get(funcOp);
}

void IntrinsicLibrary::addCleanUpForTemp(mlir::Location loc, mlir::Value temp) {
  assert(stmtCtx);
  auto *bldr = &builder;
  stmtCtx->attachCleanup([=]() { bldr->create<fir::FreeMemOp>(loc, temp); });
}

fir::ExtendedValue
IntrinsicLibrary::readAndAddCleanUp(fir::MutableBoxValue resultMutableBox,
                                    mlir::Type resultType,
                                    llvm::StringRef intrinsicName) {
  auto res = fir::factory::genMutableBoxRead(builder, loc, resultMutableBox);
  return res.match(
      [&](const fir::ArrayBoxValue &box) -> fir::ExtendedValue {
        // Add cleanup code
        addCleanUpForTemp(loc, box.getAddr());
        return box;
      },
      [&](const fir::BoxValue &box) -> fir::ExtendedValue {
        // Add cleanup code
        auto addr =
            builder.create<fir::BoxAddrOp>(loc, box.getMemTy(), box.getAddr());
        addCleanUpForTemp(loc, addr);
        return box;
      },
      [&](const fir::CharArrayBoxValue &box) -> fir::ExtendedValue {
        // Add cleanup code
        addCleanUpForTemp(loc, box.getAddr());
        return box;
      },
      [&](const mlir::Value &tempAddr) -> fir::ExtendedValue {
        // Add cleanup code
        addCleanUpForTemp(loc, tempAddr);
        return builder.create<fir::LoadOp>(loc, resultType, tempAddr);
      },
      [&](const fir::CharBoxValue &box) -> fir::ExtendedValue {
        // Add cleanup code
        addCleanUpForTemp(loc, box.getAddr());
        return box;
      },
      [&](const auto &) -> fir::ExtendedValue {
        fir::emitFatalError(loc, "unexpected result for " + intrinsicName);
      });
}

//===----------------------------------------------------------------------===//
// Code generators for the intrinsic
//===----------------------------------------------------------------------===//

mlir::Value IntrinsicLibrary::genRuntimeCall(llvm::StringRef name,
                                             mlir::Type resultType,
                                             llvm::ArrayRef<mlir::Value> args) {
  mlir::FunctionType soughtFuncType =
      getFunctionType(resultType, args, builder);
  return getRuntimeCallGenerator(name, soughtFuncType)(builder, loc, args);
}

mlir::Value IntrinsicLibrary::genConversion(mlir::Type resultType,
                                            llvm::ArrayRef<mlir::Value> args) {
  // There can be an optional kind in second argument.
  assert(args.size() >= 1);
  return builder.convertWithSemantics(loc, resultType, args[0]);
}

// ABS
mlir::Value IntrinsicLibrary::genAbs(mlir::Type resultType,
                                     llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);
  auto arg = args[0];
  auto type = arg.getType();
  if (fir::isa_real(type)) {
    // Runtime call to fp abs. An alternative would be to use mlir math::AbsFOp
    // but it does not support all fir floating point types.
    return genRuntimeCall("abs", resultType, args);
  }
  if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
    // At the time of this implementation there is no abs op in mlir.
    // So, implement abs here without branching.
    auto shift =
        builder.createIntegerConstant(loc, intType, intType.getWidth() - 1);
    auto mask = builder.create<mlir::arith::ShRSIOp>(loc, arg, shift);
    auto xored = builder.create<mlir::arith::XOrIOp>(loc, arg, mask);
    return builder.create<mlir::arith::SubIOp>(loc, xored, mask);
  }
  if (fir::isa_complex(type)) {
    // Use HYPOT to fulfill the no underflow/overflow requirement.
    auto parts =
        fir::factory::ComplexExprHelper{builder, loc}.extractParts(arg);
    llvm::SmallVector<mlir::Value> args = {parts.first, parts.second};
    return genRuntimeCall("hypot", resultType, args);
  }
  llvm_unreachable("unexpected type in ABS argument");
}

// ADJUSTL & ADJUSTR
template <void (*CallRuntime)(fir::FirOpBuilder &, mlir::Location loc,
                              mlir::Value, mlir::Value)>
fir::ExtendedValue
IntrinsicLibrary::genAdjustRtCall(mlir::Type resultType,
                                  llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  auto string = builder.createBox(loc, args[0]);
  // Create a mutable fir.box to be passed to the runtime for the result.
  auto resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultType);
  auto resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  // Call the runtime -- the runtime will allocate the result.
  CallRuntime(builder, loc, resultIrBox, string);

  // Read result from mutable fir.box and add it to the list of temps to be
  // finalized by the StatementContext.
  auto res = fir::factory::genMutableBoxRead(builder, loc, resultMutableBox);
  return res.match(
      [&](const fir::CharBoxValue &box) -> fir::ExtendedValue {
        addCleanUpForTemp(loc, fir::getBase(box));
        return box;
      },
      [&](const auto &) -> fir::ExtendedValue {
        fir::emitFatalError(loc, "result of ADJUSTL is not a scalar character");
      });
}

// AIMAG
mlir::Value IntrinsicLibrary::genAimag(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);
  return fir::factory::ComplexExprHelper{builder, loc}.extractComplexPart(
      args[0], true /* isImagPart */);
}

// ALL
fir::ExtendedValue
IntrinsicLibrary::genAll(mlir::Type resultType,
                         llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 2);
  // Handle required mask argument
  auto mask = builder.createBox(loc, args[0]);

  fir::BoxValue maskArry = builder.createBox(loc, args[0]);
  int rank = maskArry.rank();
  assert(rank >= 1);

  // Handle optional dim argument
  bool absentDim = isAbsent(args[1]);
  auto dim = absentDim
                 ? builder.createIntegerConstant(loc, builder.getIndexType(), 1)
                 : fir::getBase(args[1]);

  if (rank == 1 || absentDim)
    return builder.createConvert(loc, resultType,
                                 fir::runtime::genAll(builder, loc, mask, dim));

  // else use the result descriptor AllDim() intrinsic

  // Create mutable fir.box to be passed to the runtime for the result.

  auto resultArrayType = builder.getVarLenSeqTy(resultType, rank - 1);
  auto resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  auto resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  // Call runtime. The runtime is allocating the result.
  fir::runtime::genAllDescriptor(builder, loc, resultIrBox, mask, dim);
  return fir::factory::genMutableBoxRead(builder, loc, resultMutableBox)
      .match(
          [&](const fir::ArrayBoxValue &box) -> fir::ExtendedValue {
            addCleanUpForTemp(loc, box.getAddr());
            return box;
          },
          [&](const auto &) -> fir::ExtendedValue {
            fir::emitFatalError(loc, "Invalid result for ALL");
          });
}

// ALLOCATED
fir::ExtendedValue
IntrinsicLibrary::genAllocated(mlir::Type resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  return args[0].match(
      [&](const fir::MutableBoxValue &x) -> fir::ExtendedValue {
        return fir::factory::genIsAllocatedOrAssociatedTest(builder, loc, x);
      },
      [&](const auto &) -> fir::ExtendedValue {
        fir::emitFatalError(loc,
                            "allocated arg not lowered to MutableBoxValue");
      });
}

// ANINT
mlir::Value IntrinsicLibrary::genAnint(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() >= 1 && args.size() <= 2);
  // Skip optional kind argument to search the runtime; it is already reflected
  // in result type.
  return genRuntimeCall("anint", resultType, {args[0]});
}

// ANY
fir::ExtendedValue
IntrinsicLibrary::genAny(mlir::Type resultType,
                         llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 2);
  // Handle required mask argument
  auto mask = builder.createBox(loc, args[0]);

  fir::BoxValue maskArry = builder.createBox(loc, args[0]);
  int rank = maskArry.rank();
  assert(rank >= 1);

  // Handle optional dim argument
  bool absentDim = isAbsent(args[1]);
  auto dim = absentDim
                 ? builder.createIntegerConstant(loc, builder.getIndexType(), 1)
                 : fir::getBase(args[1]);

  if (rank == 1 || absentDim)
    return builder.createConvert(loc, resultType,
                                 fir::runtime::genAny(builder, loc, mask, dim));

  // else use the result descriptor AnyDim() intrinsic

  // Create mutable fir.box to be passed to the runtime for the result.

  auto resultArrayType = builder.getVarLenSeqTy(resultType, rank - 1);
  auto resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  auto resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  // Call runtime. The runtime is allocating the result.
  fir::runtime::genAnyDescriptor(builder, loc, resultIrBox, mask, dim);
  return fir::factory::genMutableBoxRead(builder, loc, resultMutableBox)
      .match(
          [&](const fir::ArrayBoxValue &box) -> fir::ExtendedValue {
            addCleanUpForTemp(loc, box.getAddr());
            return box;
          },
          [&](const auto &) -> fir::ExtendedValue {
            fir::emitFatalError(loc, "Invalid result for ANY");
          });
}

// ASSOCIATED
fir::ExtendedValue
IntrinsicLibrary::genAssociated(mlir::Type resultType,
                                llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  auto *pointer =
      args[0].match([&](const fir::MutableBoxValue &x) { return &x; },
                    [&](const auto &) -> const fir::MutableBoxValue * {
                      fir::emitFatalError(loc, "pointer not a MutableBoxValue");
                    });
  if (isAbsent(args[1]))
    return fir::factory::genIsAllocatedOrAssociatedTest(builder, loc, *pointer);
  auto pointerBoxRef = fir::factory::getMutableIRBox(builder, loc, *pointer);
  auto pointerBox = builder.create<fir::LoadOp>(loc, pointerBoxRef);
  return Fortran::lower::genAssociated(builder, loc, pointerBox,
                                       *args[1].getUnboxed());
}

// AINT
mlir::Value IntrinsicLibrary::genAint(mlir::Type resultType,
                                      llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() >= 1 && args.size() <= 2);
  // Skip optional kind argument to search the runtime; it is already reflected
  // in result type.
  return genRuntimeCall("aint", resultType, {args[0]});
}

// BTEST
mlir::Value IntrinsicLibrary::genBtest(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  // A conformant BTEST(I,POS) call satisfies:
  //     POS >= 0
  //     POS < BIT_SIZE(I)
  // Return:  (I >> POS) & 1
  assert(args.size() == 2);
  auto argType = args[0].getType();
  auto pos = builder.createConvert(loc, argType, args[1]);
  auto shift = builder.create<mlir::arith::ShRUIOp>(loc, args[0], pos);
  auto one = builder.createIntegerConstant(loc, argType, 1);
  auto res = builder.create<mlir::arith::AndIOp>(loc, shift, one);
  return builder.createConvert(loc, resultType, res);
}

// CEILING
mlir::Value IntrinsicLibrary::genCeiling(mlir::Type resultType,
                                         llvm::ArrayRef<mlir::Value> args) {
  // Optional KIND argument.
  assert(args.size() >= 1);
  auto arg = args[0];
  // Use ceil that is not an actual Fortran intrinsic but that is
  // an llvm intrinsic that does the same, but return a floating
  // point.
  auto ceil = genRuntimeCall("ceil", arg.getType(), {arg});
  return builder.createConvert(loc, resultType, ceil);
}

fir::ExtendedValue
IntrinsicLibrary::genChar(mlir::Type type,
                          llvm::ArrayRef<fir::ExtendedValue> args) {
  // Optional KIND argument.
  assert(args.size() >= 1);
  auto *arg = args[0].getUnboxed();
  // expect argument to be a scalar integer
  if (!arg)
    mlir::emitError(loc, "CHAR intrinsic argument not unboxed");
  fir::factory::CharacterExprHelper helper{builder, loc};
  auto kind = helper.getCharacterType(type).getFKind();
  auto cast = helper.createSingletonFromCode(*arg, kind);
  auto len =
      builder.createIntegerConstant(loc, builder.getCharacterLengthType(), 1);
  return fir::CharBoxValue{cast, len};
}

// CONJG
mlir::Value IntrinsicLibrary::genConjg(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);
  if (resultType != args[0].getType())
    llvm_unreachable("argument type mismatch");

  mlir::Value cplx = args[0];
  auto imag = fir::factory::ComplexExprHelper{builder, loc}.extractComplexPart(
      cplx, /*isImagPart=*/true);
  auto negImag = builder.create<mlir::arith::NegFOp>(loc, imag);
  return fir::factory::ComplexExprHelper{builder, loc}.insertComplexPart(
      cplx, negImag, /*isImagPart=*/true);
}

// COUNT
fir::ExtendedValue
IntrinsicLibrary::genCount(mlir::Type resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);

  // Handle mask argument
  fir::BoxValue mask = builder.createBox(loc, args[0]);
  auto maskRank = mask.rank();

  assert(maskRank > 0);

  // Handle optional dim argument
  bool absentDim = isAbsent(args[1]);
  auto dim = absentDim
                 ? builder.createIntegerConstant(loc, builder.getIndexType(), 0)
                 : fir::getBase(args[1]);

  if (absentDim || maskRank == 1) {
    // Result is scalar if no dim argument or mask is rank 1.
    // So, call specialized Count runtime routine.
    return builder.createConvert(
        loc, resultType,
        fir::runtime::genCount(builder, loc, fir::getBase(mask), dim));
  }

  // Call general CountDim runtime routine.

  // Handle optional kind argument
  bool absentKind = isAbsent(args[2]);
  auto kind = absentKind ? builder.createIntegerConstant(
                               loc, builder.getIndexType(),
                               builder.getKindMap().defaultIntegerKind())
                         : fir::getBase(args[2]);

  // Create mutable fir.box to be passed to the runtime for the result.
  auto type = builder.getVarLenSeqTy(resultType, maskRank - 1);
  auto resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, type);

  auto resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  fir::runtime::genCountDim(builder, loc, resultIrBox, fir::getBase(mask), dim,
                            kind);

  // Handle cleanup of allocatable result descriptor and return
  auto res = fir::factory::genMutableBoxRead(builder, loc, resultMutableBox);
  return res.match(
      [&](const fir::ArrayBoxValue &box) -> fir::ExtendedValue {
        // Add cleanup code
        addCleanUpForTemp(loc, box.getAddr());
        return box;
      },
      [&](const auto &) -> fir::ExtendedValue {
        fir::emitFatalError(loc, "unexpected result for COUNT");
      });
}

// CPU_TIME
void IntrinsicLibrary::genCpuTime(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  auto arg = args[0].getUnboxed();
  assert(arg && "nonscalar cpu_time argument");
  auto res1 = Fortran::lower::genCpuTime(builder, loc);
  auto res2 =
      builder.createConvert(loc, fir::dyn_cast_ptrEleTy(arg->getType()), res1);
  builder.create<fir::StoreOp>(loc, res2, *arg);
}

// CSHIFT
fir::ExtendedValue
IntrinsicLibrary::genCshift(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);

  // Handle required ARRAY argument
  fir::BoxValue arrayBox = builder.createBox(loc, args[0]);
  auto array = fir::getBase(arrayBox);
  auto arrayRank = arrayBox.rank();

  // Create mutable fir.box to be passed to the runtime for the result.
  auto resultArrayType = builder.getVarLenSeqTy(resultType, arrayRank);
  auto resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  auto resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  if (arrayRank == 1) {
    // Vector case
    // Handle required SHIFT argument as a scalar
    auto shiftAddr = args[1].getUnboxed();
    assert(shiftAddr && "nonscalar CSHIFT argument");
    auto shift = builder.create<fir::LoadOp>(loc, *shiftAddr);

    fir::runtime::genCshiftVector(builder, loc, resultIrBox, array, shift);
  } else {
    // Non-vector case
    // Handle required SHIFT argument as an array
    auto shift = builder.createBox(loc, args[1]);

    // Handle optional DIM argument
    auto dim =
        isAbsent(args[2])
            ? builder.createIntegerConstant(loc, builder.getIndexType(), 1)
            : fir::getBase(args[2]);
    fir::runtime::genCshift(builder, loc, resultIrBox, array, shift, dim);
  }
  return readAndAddCleanUp(resultMutableBox, resultType, "CSHIFT");
}

// DATE_AND_TIME
void IntrinsicLibrary::genDateAndTime(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4 && "date_and_time has 4 args");
  llvm::SmallVector<llvm::Optional<fir::CharBoxValue>> charArgs(3);
  for (auto i = 0; i < 3; ++i)
    if (auto *charBox = args[i].getCharBox())
      charArgs[i] = *charBox;

  auto values = fir::getBase(args[3]);
  if (!values)
    values = builder.create<fir::AbsentOp>(
        loc, fir::BoxType::get(builder.getNoneType()));

  Fortran::lower::genDateAndTime(builder, loc, charArgs[0], charArgs[1],
                                 charArgs[2], values);
}

// DIM
mlir::Value IntrinsicLibrary::genDim(mlir::Type resultType,
                                     llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  if (resultType.isa<mlir::IntegerType>()) {
    auto zero = builder.createIntegerConstant(loc, resultType, 0);
    auto diff = builder.create<mlir::arith::SubIOp>(loc, args[0], args[1]);
    auto cmp = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::sgt, diff, zero);
    return builder.create<mlir::SelectOp>(loc, cmp, diff, zero);
  }
  assert(fir::isa_real(resultType) && "Only expects real and integer in DIM");
  auto zero = builder.createRealZeroConstant(loc, resultType);
  auto diff = builder.create<mlir::arith::SubFOp>(loc, args[0], args[1]);
  auto cmp = builder.create<mlir::arith::CmpFOp>(
      loc, mlir::arith::CmpFPredicate::OGT, diff, zero);
  return builder.create<mlir::SelectOp>(loc, cmp, diff, zero);
}

// DOT_PRODUCT
fir::ExtendedValue
IntrinsicLibrary::genDotProduct(mlir::Type resultType,
                                llvm::ArrayRef<fir::ExtendedValue> args) {
  return genDotProd(fir::runtime::genDotProduct, resultType, builder, loc,
                    stmtCtx, args);
}

// DPROD
mlir::Value IntrinsicLibrary::genDprod(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  assert(fir::isa_real(resultType) &&
         "Result must be double precision in DPROD");
  auto a = builder.createConvert(loc, resultType, args[0]);
  auto b = builder.createConvert(loc, resultType, args[1]);
  return builder.create<mlir::arith::MulFOp>(loc, a, b);
}

// EOSHIFT
fir::ExtendedValue
IntrinsicLibrary::genEoshift(mlir::Type resultType,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);

  // Handle required ARRAY argument
  fir::BoxValue arrayBox = builder.createBox(loc, args[0]);
  auto array = fir::getBase(arrayBox);
  auto arrayRank = arrayBox.rank();

  // Create mutable fir.box to be passed to the runtime for the result.
  auto resultArrayType = builder.getVarLenSeqTy(resultType, arrayRank);
  auto resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  auto resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  // Handle optional BOUNDARY argument
  auto boundary = isAbsent(args[2])
                      ? builder.create<fir::AbsentOp>(
                            loc, fir::BoxType::get(builder.getNoneType()))
                      : builder.createBox(loc, args[2]);

  if (arrayRank == 1) {
    // Vector case
    // Handle required SHIFT argument as a scalar
    auto shiftAddr = args[1].getUnboxed();
    assert(shiftAddr && "nonscalar EOSHIFT SHIFT argument");
    auto shift = builder.create<fir::LoadOp>(loc, *shiftAddr);
    fir::runtime::genEoshiftVector(builder, loc, resultIrBox, array, shift,
                                   boundary);
  } else {
    // Non-vector case
    // Handle required SHIFT argument as an array
    auto shift = builder.createBox(loc, args[1]);

    // Handle optional DIM argument
    auto dim =
        isAbsent(args[3])
            ? builder.createIntegerConstant(loc, builder.getIndexType(), 1)
            : fir::getBase(args[3]);
    fir::runtime::genEoshift(builder, loc, resultIrBox, array, shift, boundary,
                             dim);
  }
  return readAndAddCleanUp(resultMutableBox, resultType,
                           "unexpected result for EOSHIFT");
}

// EXPONENT
mlir::Value IntrinsicLibrary::genExponent(mlir::Type resultType,
                                          llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genExponent(builder, loc, resultType,
                                fir::getBase(args[0])));
}

// FLOOR
mlir::Value IntrinsicLibrary::genFloor(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  // Optional KIND argument.
  assert(args.size() >= 1);
  auto arg = args[0];
  // Use LLVM floor that returns real.
  auto floor = genRuntimeCall("floor", arg.getType(), {arg});
  return builder.createConvert(loc, resultType, floor);
}

// FRACTION
mlir::Value IntrinsicLibrary::genFraction(mlir::Type resultType,
                                          llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genFraction(builder, loc, fir::getBase(args[0])));
}

// IAND
mlir::Value IntrinsicLibrary::genIand(mlir::Type resultType,
                                      llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  return builder.create<mlir::arith::AndIOp>(loc, args[0], args[1]);
}

// IBCLR
mlir::Value IntrinsicLibrary::genIbclr(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  // A conformant IBCLR(I,POS) call satisfies:
  //     POS >= 0
  //     POS < BIT_SIZE(I)
  // Return:  I & (!(1 << POS))
  assert(args.size() == 2);
  auto pos = builder.createConvert(loc, resultType, args[1]);
  auto one = builder.createIntegerConstant(loc, resultType, 1);
  auto ones = builder.createIntegerConstant(loc, resultType, -1);
  auto mask = builder.create<mlir::arith::ShLIOp>(loc, one, pos);
  auto res = builder.create<mlir::arith::XOrIOp>(loc, ones, mask);
  return builder.create<mlir::arith::AndIOp>(loc, args[0], res);
}

// IBITS
mlir::Value IntrinsicLibrary::genIbits(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  // A conformant IBITS(I,POS,LEN) call satisfies:
  //     POS >= 0
  //     LEN >= 0
  //     POS + LEN <= BIT_SIZE(I)
  // Return:  LEN == 0 ? 0 : (I >> POS) & (-1 >> (BIT_SIZE(I) - LEN))
  // For a conformant call, implementing (I >> POS) with a signed or an
  // unsigned shift produces the same result.  For a nonconformant call,
  // the two choices may produce different results.
  assert(args.size() == 3);
  auto pos = builder.createConvert(loc, resultType, args[1]);
  auto len = builder.createConvert(loc, resultType, args[2]);
  auto bitSize = builder.createIntegerConstant(
      loc, resultType, resultType.cast<mlir::IntegerType>().getWidth());
  auto shiftCount = builder.create<mlir::arith::SubIOp>(loc, bitSize, len);
  auto zero = builder.createIntegerConstant(loc, resultType, 0);
  auto ones = builder.createIntegerConstant(loc, resultType, -1);
  auto mask = builder.create<mlir::arith::ShRUIOp>(loc, ones, shiftCount);
  auto res1 = builder.create<mlir::arith::ShRSIOp>(loc, args[0], pos);
  auto res2 = builder.create<mlir::arith::AndIOp>(loc, res1, mask);
  auto lenIsZero = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, len, zero);
  return builder.create<mlir::SelectOp>(loc, lenIsZero, zero, res2);
}

// IBSET
mlir::Value IntrinsicLibrary::genIbset(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  // A conformant IBSET(I,POS) call satisfies:
  //     POS >= 0
  //     POS < BIT_SIZE(I)
  // Return:  I | (1 << POS)
  assert(args.size() == 2);
  auto pos = builder.createConvert(loc, resultType, args[1]);
  auto one = builder.createIntegerConstant(loc, resultType, 1);
  auto mask = builder.create<mlir::arith::ShLIOp>(loc, one, pos);
  return builder.create<mlir::arith::OrIOp>(loc, args[0], mask);
}

// ICHAR
fir::ExtendedValue
IntrinsicLibrary::genIchar(mlir::Type resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args) {
  // There can be an optional kind in second argument.
  assert(args.size() == 2);
  auto charBox = args[0].getCharBox();
  if (!charBox)
    llvm::report_fatal_error("expected character scalar");

  fir::factory::CharacterExprHelper helper{builder, loc};
  auto buffer = charBox->getBuffer();
  auto bufferTy = buffer.getType();
  mlir::Value charVal;
  if (auto charTy = bufferTy.dyn_cast<fir::CharacterType>()) {
    assert(charTy.singleton());
    charVal = buffer;
  } else {
    // Character is in memory, cast to fir.ref<char> and load.
    auto ty = fir::dyn_cast_ptrEleTy(bufferTy);
    if (!ty)
      llvm::report_fatal_error("expected memory type");
    // The length of in the character type may be unknown. Casting
    // to a singleton ref is required before loading.
    auto eleType = helper.getCharacterType(ty);
    auto charType =
        fir::CharacterType::get(builder.getContext(), eleType.getFKind(), 1);
    auto toTy = builder.getRefType(charType);
    auto cast = builder.createConvert(loc, toTy, buffer);
    charVal = builder.create<fir::LoadOp>(loc, cast);
  }
  LLVM_DEBUG(llvm::dbgs() << "ichar(" << charVal << ")\n");
  auto code = helper.extractCodeFromSingleton(charVal);
  return builder.create<mlir::arith::ExtUIOp>(loc, resultType, code);
}

// IEOR
mlir::Value IntrinsicLibrary::genIeor(mlir::Type resultType,
                                      llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  return builder.create<mlir::arith::XOrIOp>(loc, args[0], args[1]);
}

// INDEX
fir::ExtendedValue
IntrinsicLibrary::genIndex(mlir::Type resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() >= 2 && args.size() <= 4);

  auto stringBase = fir::getBase(args[0]);
  auto kind = fir::factory::CharacterExprHelper{builder, loc}.getCharacterKind(
      stringBase.getType());
  auto stringLen = fir::getLen(args[0]);
  auto substringBase = fir::getBase(args[1]);
  auto substringLen = fir::getLen(args[1]);
  mlir::Value back =
      isAbsent(args, 2)
          ? builder.createIntegerConstant(loc, builder.getI1Type(), 0)
          : fir::getBase(args[2]);
  if (isAbsent(args, 3))
    return builder.createConvert(
        loc, resultType,
        fir::runtime::genIndex(builder, loc, kind, stringBase, stringLen,
                               substringBase, substringLen, back));

  // Call the descriptor-based Index implementation
  auto string = builder.createBox(loc, args[0]);
  auto substring = builder.createBox(loc, args[1]);
  auto makeRefThenEmbox = [&](mlir::Value b) {
    auto logTy = fir::LogicalType::get(
        builder.getContext(), builder.getKindMap().defaultLogicalKind());
    auto temp = builder.createTemporary(loc, logTy);
    auto castb = builder.createConvert(loc, logTy, b);
    builder.create<fir::StoreOp>(loc, castb, temp);
    return builder.createBox(loc, temp);
  };
  auto backOpt = isAbsent(args, 2)
                     ? builder.create<fir::AbsentOp>(
                           loc, fir::BoxType::get(builder.getI1Type()))
                     : makeRefThenEmbox(fir::getBase(args[2]));
  auto kindVal = isAbsent(args, 3)
                     ? builder.createIntegerConstant(
                           loc, builder.getIndexType(),
                           builder.getKindMap().defaultIntegerKind())
                     : fir::getBase(args[3]);
  // Create mutable fir.box to be passed to the runtime for the result.
  auto mutBox = fir::factory::createTempMutableBox(builder, loc, resultType);
  auto resBox = fir::factory::getMutableIRBox(builder, loc, mutBox);
  // Call runtime. The runtime is allocating the result.
  fir::runtime::genIndexDescriptor(builder, loc, resBox, string, substring,
                                   backOpt, kindVal);
  // Read back the result from the mutable box.
  return readAndAddCleanUp(mutBox, resultType, "INDEX");
}

// IOR
mlir::Value IntrinsicLibrary::genIor(mlir::Type resultType,
                                     llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  return builder.create<mlir::arith::OrIOp>(loc, args[0], args[1]);
}

// ISHFT
mlir::Value IntrinsicLibrary::genIshft(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  // A conformant ISHFT(I,SHIFT) call satisfies:
  //     abs(SHIFT) <= BIT_SIZE(I)
  // Return:  abs(SHIFT) >= BIT_SIZE(I)
  //              ? 0
  //              : SHIFT < 0
  //                    ? I >> abs(SHIFT)
  //                    : I << abs(SHIFT)
  assert(args.size() == 2);
  auto bitSize = builder.createIntegerConstant(
      loc, resultType, resultType.cast<mlir::IntegerType>().getWidth());
  auto zero = builder.createIntegerConstant(loc, resultType, 0);
  auto shift = builder.createConvert(loc, resultType, args[1]);
  auto absShift = genAbs(resultType, {shift});
  auto left = builder.create<mlir::arith::ShLIOp>(loc, args[0], absShift);
  auto right = builder.create<mlir::arith::ShRUIOp>(loc, args[0], absShift);
  auto shiftIsLarge = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::sge, absShift, bitSize);
  auto shiftIsNegative = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, shift, zero);
  auto sel = builder.create<mlir::SelectOp>(loc, shiftIsNegative, right, left);
  return builder.create<mlir::SelectOp>(loc, shiftIsLarge, zero, sel);
}

// ISHFTC
mlir::Value IntrinsicLibrary::genIshftc(mlir::Type resultType,
                                        llvm::ArrayRef<mlir::Value> args) {
  // A conformant ISHFTC(I,SHIFT,SIZE) call satisfies:
  //     SIZE > 0
  //     SIZE <= BIT_SIZE(I)
  //     abs(SHIFT) <= SIZE
  // if SHIFT > 0
  //     leftSize = abs(SHIFT)
  //     rightSize = SIZE - abs(SHIFT)
  // else [if SHIFT < 0]
  //     leftSize = SIZE - abs(SHIFT)
  //     rightSize = abs(SHIFT)
  // unchanged = SIZE == BIT_SIZE(I) ? 0 : (I >> SIZE) << SIZE
  // leftMaskShift = BIT_SIZE(I) - leftSize
  // rightMaskShift = BIT_SIZE(I) - rightSize
  // left = (I >> rightSize) & (-1 >> leftMaskShift)
  // right = (I & (-1 >> rightMaskShift)) << leftSize
  // Return:  SHIFT == 0 || SIZE == abs(SHIFT) ? I : (unchanged | left | right)
  assert(args.size() == 3);
  auto bitSize = builder.createIntegerConstant(
      loc, resultType, resultType.cast<mlir::IntegerType>().getWidth());
  auto I = args[0];
  auto shift = builder.createConvert(loc, resultType, args[1]);
  auto size =
      args[2] ? builder.createConvert(loc, resultType, args[2]) : bitSize;
  auto zero = builder.createIntegerConstant(loc, resultType, 0);
  auto ones = builder.createIntegerConstant(loc, resultType, -1);
  auto absShift = genAbs(resultType, {shift});
  auto elseSize = builder.create<mlir::arith::SubIOp>(loc, size, absShift);
  auto shiftIsZero = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, shift, zero);
  auto shiftEqualsSize = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, absShift, size);
  auto shiftIsNop =
      builder.create<mlir::arith::OrIOp>(loc, shiftIsZero, shiftEqualsSize);
  auto shiftIsPositive = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::sgt, shift, zero);
  auto leftSize =
      builder.create<mlir::SelectOp>(loc, shiftIsPositive, absShift, elseSize);
  auto rightSize =
      builder.create<mlir::SelectOp>(loc, shiftIsPositive, elseSize, absShift);
  auto hasUnchanged = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, size, bitSize);
  auto unchangedTmp1 = builder.create<mlir::arith::ShRUIOp>(loc, I, size);
  auto unchangedTmp2 =
      builder.create<mlir::arith::ShLIOp>(loc, unchangedTmp1, size);
  auto unchanged =
      builder.create<mlir::SelectOp>(loc, hasUnchanged, unchangedTmp2, zero);
  auto leftMaskShift =
      builder.create<mlir::arith::SubIOp>(loc, bitSize, leftSize);
  auto leftMask =
      builder.create<mlir::arith::ShRUIOp>(loc, ones, leftMaskShift);
  auto leftTmp = builder.create<mlir::arith::ShRUIOp>(loc, I, rightSize);
  auto left = builder.create<mlir::arith::AndIOp>(loc, leftTmp, leftMask);
  auto rightMaskShift =
      builder.create<mlir::arith::SubIOp>(loc, bitSize, rightSize);
  auto rightMask =
      builder.create<mlir::arith::ShRUIOp>(loc, ones, rightMaskShift);
  auto rightTmp = builder.create<mlir::arith::AndIOp>(loc, I, rightMask);
  auto right = builder.create<mlir::arith::ShLIOp>(loc, rightTmp, leftSize);
  auto resTmp = builder.create<mlir::arith::OrIOp>(loc, unchanged, left);
  auto res = builder.create<mlir::arith::OrIOp>(loc, resTmp, right);
  return builder.create<mlir::SelectOp>(loc, shiftIsNop, I, res);
}

// LEN
// Note that this is only used for an unrestricted intrinsic LEN call.
// Other uses of LEN are rewritten as descriptor inquiries by the front-end.
fir::ExtendedValue
IntrinsicLibrary::genLen(mlir::Type resultType,
                         llvm::ArrayRef<fir::ExtendedValue> args) {
  // Optional KIND argument reflected in result type.
  assert(args.size() == 1); // args.size() != 2
  mlir::Value len;
  if (const auto *charBox = args[0].getCharBox()) {
    len = charBox->getLen();
  } else if (const auto *charBoxArray = args[0].getCharBox()) {
    len = charBoxArray->getLen();
  } else {
    fir::factory::CharacterExprHelper helper{builder, loc};
    len = helper.createUnboxChar(fir::getBase(args[0])).second;
  }

  return builder.createConvert(loc, resultType, len);
}

// LEN_TRIM
fir::ExtendedValue
IntrinsicLibrary::genLenTrim(mlir::Type resultType,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  // Optional KIND argument reflected in result type.
  assert(args.size() >= 1);
  fir::factory::CharacterExprHelper helper{builder, loc};
  auto *charBox = args[0].getCharBox();
  if (!charBox)
    TODO(loc, "character array len_trim");
  auto len = helper.createLenTrim(*charBox);
  return builder.createConvert(loc, resultType, len);
}

// LGE, LGT, LLE, LLT
template <mlir::arith::CmpIPredicate pred>
fir::ExtendedValue
IntrinsicLibrary::genCharacterCompare(mlir::Type type,
                                      llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  return fir::runtime::genCharCompare(
      builder, loc, pred, fir::getBase(args[0]), fir::getLen(args[0]),
      fir::getBase(args[1]), fir::getLen(args[1]));
}

// MATMUL
fir::ExtendedValue
IntrinsicLibrary::genMatmul(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);

  // Handle required matmul arguments
  fir::BoxValue matrixTmpA = builder.createBox(loc, args[0]);
  mlir::Value matrixA = fir::getBase(matrixTmpA);
  fir::BoxValue matrixTmpB = builder.createBox(loc, args[1]);
  mlir::Value matrixB = fir::getBase(matrixTmpB);
  unsigned resultRank =
      (matrixTmpA.rank() == 1 || matrixTmpB.rank() == 1) ? 1 : 2;

  // Create mutable fir.box to be passed to the runtime for the result.
  auto resultArrayType = builder.getVarLenSeqTy(resultType, resultRank);
  auto resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  auto resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);
  // Call runtime. The runtime is allocating the result.
  fir::runtime::genMatmul(builder, loc, resultIrBox, matrixA, matrixB);
  // Read result from mutable fir.box and add it to the list of temps to be
  // finalized by the StatementContext.
  return readAndAddCleanUp(resultMutableBox, resultType,
                           "unexpected result for MATMUL");
}

// MERGE
mlir::Value IntrinsicLibrary::genMerge(mlir::Type,
                                       llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 3);
  auto mask = builder.createConvert(loc, builder.getI1Type(), args[2]);
  return builder.create<mlir::SelectOp>(loc, mask, args[0], args[1]);
}

// MOD
mlir::Value IntrinsicLibrary::genMod(mlir::Type resultType,
                                     llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  if (resultType.isa<mlir::IntegerType>())
    return builder.create<mlir::arith::RemSIOp>(loc, args[0], args[1]);

  // Use runtime. Note that mlir::arith::RemFOp implements floating point
  // remainder, but it does not work with fir::Real type.
  // TODO: consider using mlir::arith::RemFOp when possible, that may help
  // folding and  optimizations.
  return genRuntimeCall("mod", resultType, args);
}

// MODULO
mlir::Value IntrinsicLibrary::genModulo(mlir::Type resultType,
                                        llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  // No floored modulo op in LLVM/MLIR yet. TODO: add one to MLIR.
  // In the meantime, use a simple inlined implementation based on truncated
  // modulo (MOD(A, P) implemented by RemIOp, RemFOp). This avoids making manual
  // division and multiplication from MODULO formula.
  //  - If A/P > 0 or MOD(A,P)=0, then INT(A/P) = FLOOR(A/P), and MODULO = MOD.
  //  - Otherwise, when A/P < 0 and MOD(A,P) !=0, then MODULO(A, P) =
  //    A-FLOOR(A/P)*P = A-(INT(A/P)-1)*P = A-INT(A/P)*P+P = MOD(A,P)+P
  // Note that A/P < 0 if and only if A and P signs are different.
  if (resultType.isa<mlir::IntegerType>()) {
    auto remainder =
        builder.create<mlir::arith::RemSIOp>(loc, args[0], args[1]);
    auto argXor = builder.create<mlir::arith::XOrIOp>(loc, args[0], args[1]);
    auto zero = builder.createIntegerConstant(loc, argXor.getType(), 0);
    auto argSignDifferent = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::slt, argXor, zero);
    auto remainderIsNotZero = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, remainder, zero);
    auto mustAddP = builder.create<mlir::arith::AndIOp>(loc, remainderIsNotZero,
                                                        argSignDifferent);
    auto remPlusP =
        builder.create<mlir::arith::AddIOp>(loc, remainder, args[1]);
    return builder.create<mlir::SelectOp>(loc, mustAddP, remPlusP, remainder);
  }
  // Real case
  auto remainder = builder.create<mlir::arith::RemFOp>(loc, args[0], args[1]);
  auto zero = builder.createRealZeroConstant(loc, remainder.getType());
  auto remainderIsNotZero = builder.create<mlir::arith::CmpFOp>(
      loc, mlir::arith::CmpFPredicate::UNE, remainder, zero);
  auto aLessThanZero = builder.create<mlir::arith::CmpFOp>(
      loc, mlir::arith::CmpFPredicate::OLT, args[0], zero);
  auto pLessThanZero = builder.create<mlir::arith::CmpFOp>(
      loc, mlir::arith::CmpFPredicate::OLT, args[1], zero);
  auto argSignDifferent =
      builder.create<mlir::arith::XOrIOp>(loc, aLessThanZero, pLessThanZero);
  auto mustAddP = builder.create<mlir::arith::AndIOp>(loc, remainderIsNotZero,
                                                      argSignDifferent);
  auto remPlusP = builder.create<mlir::arith::AddFOp>(loc, remainder, args[1]);
  return builder.create<mlir::SelectOp>(loc, mustAddP, remPlusP, remainder);
}

// MVBITS
void IntrinsicLibrary::genMvbits(llvm::ArrayRef<fir::ExtendedValue> args) {
  // A conformant MVBITS(FROM,FROMPOS,LEN,TO,TOPOS) call satisfies:
  //     FROMPOS >= 0
  //     LEN >= 0
  //     TOPOS >= 0
  //     FROMPOS + LEN <= BIT_SIZE(FROM)
  //     TOPOS + LEN <= BIT_SIZE(TO)
  // MASK = -1 >> (BIT_SIZE(FROM) - LEN)
  // TO = LEN == 0 ? TO : ((!(MASK << TOPOS)) & TO) |
  //                      (((FROM >> FROMPOS) & MASK) << TOPOS)
  assert(args.size() == 5);
  auto unbox = [&](fir::ExtendedValue exv) {
    auto arg = exv.getUnboxed();
    assert(arg && "nonscalar mvbits argument");
    return *arg;
  };
  auto from = unbox(args[0]);
  auto resultType = from.getType();
  auto frompos = builder.createConvert(loc, resultType, unbox(args[1]));
  auto len = builder.createConvert(loc, resultType, unbox(args[2]));
  auto toAddr = unbox(args[3]);
  assert(fir::dyn_cast_ptrEleTy(toAddr.getType()) == resultType &&
         "mismatched mvbits types");
  auto to = builder.create<fir::LoadOp>(loc, resultType, toAddr);
  auto topos = builder.createConvert(loc, resultType, unbox(args[4]));
  auto zero = builder.createIntegerConstant(loc, resultType, 0);
  auto ones = builder.createIntegerConstant(loc, resultType, -1);
  auto bitSize = builder.createIntegerConstant(
      loc, resultType, resultType.cast<mlir::IntegerType>().getWidth());
  auto shiftCount = builder.create<mlir::arith::SubIOp>(loc, bitSize, len);
  auto mask = builder.create<mlir::arith::ShRUIOp>(loc, ones, shiftCount);
  auto unchangedTmp1 = builder.create<mlir::arith::ShLIOp>(loc, mask, topos);
  auto unchangedTmp2 =
      builder.create<mlir::arith::XOrIOp>(loc, unchangedTmp1, ones);
  auto unchanged = builder.create<mlir::arith::AndIOp>(loc, unchangedTmp2, to);
  auto frombitsTmp1 = builder.create<mlir::arith::ShRUIOp>(loc, from, frompos);
  auto frombitsTmp2 =
      builder.create<mlir::arith::AndIOp>(loc, frombitsTmp1, mask);
  auto frombits = builder.create<mlir::arith::ShLIOp>(loc, frombitsTmp2, topos);
  auto resTmp = builder.create<mlir::arith::OrIOp>(loc, unchanged, frombits);
  auto lenIsZero = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, len, zero);
  auto res = builder.create<mlir::SelectOp>(loc, lenIsZero, to, resTmp);
  builder.create<fir::StoreOp>(loc, res, toAddr);
}

// NEAREST
mlir::Value IntrinsicLibrary::genNearest(mlir::Type resultType,
                                         llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);

  auto realX = fir::getBase(args[0]);
  auto realS = fir::getBase(args[1]);

  return builder.createConvert(
      loc, resultType, fir::runtime::genNearest(builder, loc, realX, realS));
}

// NINT
mlir::Value IntrinsicLibrary::genNint(mlir::Type resultType,
                                      llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() >= 1);
  // Skip optional kind argument to search the runtime; it is already reflected
  // in result type.
  return genRuntimeCall("nint", resultType, {args[0]});
}

// NOT
mlir::Value IntrinsicLibrary::genNot(mlir::Type resultType,
                                     llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);
  auto allOnes = builder.createIntegerConstant(loc, resultType, -1);
  return builder.create<mlir::arith::XOrIOp>(loc, args[0], allOnes);
}

// NULL
fir::ExtendedValue
IntrinsicLibrary::genNull(mlir::Type, llvm::ArrayRef<fir::ExtendedValue> args) {
  // NULL() without MOLD must be handled in the contexts where it can appear
  // (see table 16.5 of Fortran 2018 standard).
  assert(args.size() == 1 && isPresent(args[0]) &&
         "MOLD argument required to lower NULL outside of any context");
  const auto *mold = args[0].getBoxOf<fir::MutableBoxValue>();
  assert(mold && "MOLD must be a pointer or allocatable");
  auto boxType = mold->getBoxTy();
  auto boxStorage = builder.createTemporary(loc, boxType);
  auto box = fir::factory::createUnallocatedBox(builder, loc, boxType,
                                                mold->nonDeferredLenParams());
  builder.create<fir::StoreOp>(loc, box, boxStorage);
  return fir::MutableBoxValue(boxStorage, mold->nonDeferredLenParams(), {});
}

// PACK
fir::ExtendedValue
IntrinsicLibrary::genPack(mlir::Type resultType,
                          llvm::ArrayRef<fir::ExtendedValue> args) {
  [[maybe_unused]] auto numArgs = args.size();
  assert(numArgs == 2 || numArgs == 3);

  // Handle required array argument
  auto array = builder.createBox(loc, args[0]);

  // Handle required mask argument
  auto mask = builder.createBox(loc, args[1]);

  // Handle optional vector argument
  auto vector = isAbsent(args, 2)
                    ? builder.create<fir::AbsentOp>(
                          loc, fir::BoxType::get(builder.getI1Type()))
                    : builder.createBox(loc, args[2]);

  // Create mutable fir.box to be passed to the runtime for the result.
  auto resultArrayType = builder.getVarLenSeqTy(resultType, 1);
  auto resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  auto resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  fir::runtime::genPack(builder, loc, resultIrBox, array, mask, vector);

  return readAndAddCleanUp(resultMutableBox, resultType,
                           "unexpected result for PACK");
}

// PRESENT
fir::ExtendedValue
IntrinsicLibrary::genPresent(mlir::Type,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  return builder.create<fir::IsPresentOp>(loc, builder.getI1Type(),
                                          fir::getBase(args[0]));
}

// PRODUCT
fir::ExtendedValue
IntrinsicLibrary::genProduct(mlir::Type resultType,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  return genProdOrSum(fir::runtime::genProduct, fir::runtime::genProductDim,
                      resultType, builder, loc, stmtCtx,
                      "unexpected result for Product", args);
}

// RANDOM_INIT
void IntrinsicLibrary::genRandomInit(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  Fortran::lower::genRandomInit(builder, loc, *args[0].getUnboxed(),
                                *args[1].getUnboxed());
}

// RANDOM_NUMBER
void IntrinsicLibrary::genRandomNumber(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  Fortran::lower::genRandomNumber(builder, loc, *args[0].getUnboxed());
}

// RANDOM_SEED
void IntrinsicLibrary::genRandomSeed(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  for (int i = 0; i < 3; ++i)
    if (isPresent(args[i])) {
      Fortran::lower::genRandomSeed(builder, loc, i, *args[i].getUnboxed());
      return;
    }
  Fortran::lower::genRandomSeed(builder, loc, -1, mlir::Value{});
}

// REPEAT
fir::ExtendedValue
IntrinsicLibrary::genRepeat(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  auto string = builder.createBox(loc, args[0]);
  auto ncopies = fir::getBase(args[1]);
  // Create mutable fir.box to be passed to the runtime for the result.
  auto resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultType);
  auto resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);
  // Call runtime. The runtime is allocating the result.
  fir::runtime::genRepeat(builder, loc, resultIrBox, string, ncopies);
  // Read result from mutable fir.box and add it to the list of temps to be
  // finalized by the StatementContext.
  return readAndAddCleanUp(resultMutableBox, resultType, "REPEAT");
}

// RESHAPE
fir::ExtendedValue
IntrinsicLibrary::genReshape(mlir::Type resultType,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);

  // Handle source argument
  auto source = builder.createBox(loc, args[0]);

  // Handle shape argument
  auto shape = builder.createBox(loc, args[1]);
  fir::BoxValue shapeTmp = shape;
  [[maybe_unused]] auto shapeRank = shapeTmp.rank();
  assert(shapeRank == 1);
  auto shapeTy = shape.getType();
  auto shapeArrTy = fir::dyn_cast_ptrOrBoxEleTy(shapeTy);
  auto resultRank = shapeArrTy.cast<fir::SequenceType>().getShape();

  assert(resultRank[0] != fir::SequenceType::getUnknownExtent() &&
         "shape arg must have constant size");

  // Handle optional pad argument
  auto pad = isAbsent(args[2])
                 ? builder.create<fir::AbsentOp>(
                       loc, fir::BoxType::get(builder.getI1Type()))
                 : builder.createBox(loc, args[2]);

  // Handle optional order argument
  auto order = isAbsent(args[3])
                   ? builder.create<fir::AbsentOp>(
                         loc, fir::BoxType::get(builder.getI1Type()))
                   : builder.createBox(loc, args[3]);

  // Create mutable fir.box to be passed to the runtime for the result.
  auto type = builder.getVarLenSeqTy(resultType, resultRank[0]);
  auto resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, type);

  auto resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  fir::runtime::genReshape(builder, loc, resultIrBox, source, shape, pad,
                           order);

  return readAndAddCleanUp(resultMutableBox, resultType,
                           "unexpected result for RESHAPE");
}

// RRSPACING
mlir::Value IntrinsicLibrary::genRRSpacing(mlir::Type resultType,
                                           llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genRRSpacing(builder, loc, fir::getBase(args[0])));
}

// SCALE
mlir::Value IntrinsicLibrary::genScale(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);

  auto realX = fir::getBase(args[0]);
  auto intI = fir::getBase(args[1]);

  return builder.createConvert(
      loc, resultType, fir::runtime::genScale(builder, loc, realX, intI));
}

// SCAN
fir::ExtendedValue
IntrinsicLibrary::genScan(mlir::Type resultType,
                          llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 4);

  if (isAbsent(args[3])) {
    // Kind not specified, so call scan/verify runtime routine that is
    // specialized on the kind of characters in string.

    // Handle required string base arg
    auto stringBase = fir::getBase(args[0]);

    // Handle required set string base arg
    auto setBase = fir::getBase(args[1]);

    // Handle kind argument; it is the kind of character in this case
    auto kind =
        fir::factory::CharacterExprHelper{builder, loc}.getCharacterKind(
            stringBase.getType());

    // Get string length argument
    auto stringLen = fir::getLen(args[0]);

    // Get set string length argument
    auto setLen = fir::getLen(args[1]);

    // Handle optional back argument
    auto back = isAbsent(args[2])
                    ? builder.createIntegerConstant(loc, builder.getI1Type(), 0)
                    : fir::getBase(args[2]);

    return builder.createConvert(loc, resultType,
                                 fir::runtime::genScan(builder, loc, kind,
                                                       stringBase, stringLen,
                                                       setBase, setLen, back));
  }
  // else use the runtime descriptor version of scan/verify

  // Handle optional argument, back
  auto makeRefThenEmbox = [&](mlir::Value b) {
    auto logTy = fir::LogicalType::get(
        builder.getContext(), builder.getKindMap().defaultLogicalKind());
    auto temp = builder.createTemporary(loc, logTy);
    auto castb = builder.createConvert(loc, logTy, b);
    builder.create<fir::StoreOp>(loc, castb, temp);
    return builder.createBox(loc, temp);
  };
  auto back = fir::isUnboxedValue(args[2])
                  ? makeRefThenEmbox(*args[2].getUnboxed())
                  : builder.create<fir::AbsentOp>(
                        loc, fir::BoxType::get(builder.getI1Type()));

  // Handle required string argument
  auto string = builder.createBox(loc, args[0]);

  // Handle required set argument
  auto set = builder.createBox(loc, args[1]);

  // Handle kind argument
  auto kind = fir::getBase(args[3]);

  // Create result descriptor
  auto resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultType);
  auto resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  fir::runtime::genScanDescriptor(builder, loc, resultIrBox, string, set, back,
                                  kind);

  // Handle cleanup of allocatable result descriptor and return
  return readAndAddCleanUp(resultMutableBox, resultType, "SCAN");
}

// SET_EXPONENT
mlir::Value IntrinsicLibrary::genSetExponent(mlir::Type resultType,
                                             llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genSetExponent(builder, loc, fir::getBase(args[0]),
                                   fir::getBase(args[1])));
}

// SIGN
mlir::Value IntrinsicLibrary::genSign(mlir::Type resultType,
                                      llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  auto abs = genAbs(resultType, {args[0]});
  if (resultType.isa<mlir::IntegerType>()) {
    auto zero = builder.createIntegerConstant(loc, resultType, 0);
    auto neg = builder.create<mlir::arith::SubIOp>(loc, zero, abs);
    auto cmp = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::slt, args[1], zero);
    return builder.create<mlir::SelectOp>(loc, cmp, neg, abs);
  }
  // TODO: Requirements when second argument is +0./0.
  auto zero = builder.createRealZeroConstant(loc, resultType);
  auto neg = builder.create<mlir::arith::NegFOp>(loc, abs);
  auto cmp = builder.create<mlir::arith::CmpFOp>(
      loc, mlir::arith::CmpFPredicate::OLT, args[1], zero);
  return builder.create<mlir::SelectOp>(loc, cmp, neg, abs);
}

// SIZE
fir::ExtendedValue
IntrinsicLibrary::genSize(mlir::Type resultType,
                          llvm::ArrayRef<fir::ExtendedValue> args) {
  // TODO: handle assumed-rank arrays, especially a dummy whose actual argument
  // is an assumed-size array
  assert(args.size() == 3);

  // Calls to SIZE that don't have the DIM argument are handled elsewhere
  assert(!isAbsent(args[1]));

  // Handle the ARRAY argument
  auto array = builder.createBox(loc, args[0]);

  // Handle the DIM argument.  Note that the lowering code never sees a call
  // to SIZE that doesn't have a DIM argument.  If the Fortran source has such
  // a call, the front end changes it to an expression that consists of the
  // product of a set of calls to SIZE(ARRAY, DIM) where DIM goes from 1 to N
  // where N is the rank of an array.  For N == 2, for example, a call to
  // "SIZE(ARRAY)" gets changed to "SIZE(ARRAY, 1) * SIZE(ARRAY, 2)".

  // Convert the Fortran 1-based index to the FIR 0-based index
  auto indexType = builder.getIndexType();
  auto oneBasedDim =
      builder.createConvert(loc, indexType, fir::getBase(args[1]));
  auto one = builder.createIntegerConstant(loc, indexType, 1);
  auto zeroBasedDim =
      builder.create<mlir::arith::SubIOp>(loc, oneBasedDim, one);

  // The value of the KIND argument is already reflected in the resultType

  auto result = builder
                    .create<fir::BoxDimsOp>(loc, indexType, indexType,
                                            indexType, array, zeroBasedDim)
                    .getResult(1);
  return builder.createConvert(loc, resultType, result);
}

// SPACING
mlir::Value IntrinsicLibrary::genSpacing(mlir::Type resultType,
                                         llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genSpacing(builder, loc, fir::getBase(args[0])));
}

// SPREAD
fir::ExtendedValue
IntrinsicLibrary::genSpread(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 3);

  // Handle source argument
  auto source = builder.createBox(loc, args[0]);
  fir::BoxValue sourceTmp = source;
  auto sourceRank = sourceTmp.rank();

  // Handle Dim argument
  auto dim = fir::getBase(args[1]);

  // Handle ncopies argument
  auto ncopies = fir::getBase(args[2]);

  // Generate result descriptor
  auto resultArrayType = builder.getVarLenSeqTy(resultType, sourceRank + 1);
  auto resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  auto resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  fir::runtime::genSpread(builder, loc, resultIrBox, source, dim, ncopies);

  return readAndAddCleanUp(resultMutableBox, resultType,
                           "unexpected result for SPREAD");
}

// SUM
fir::ExtendedValue
IntrinsicLibrary::genSum(mlir::Type resultType,
                         llvm::ArrayRef<fir::ExtendedValue> args) {
  return genProdOrSum(fir::runtime::genSum, fir::runtime::genSumDim, resultType,
                      builder, loc, stmtCtx, "unexpected result for Sum", args);
}

// SYSTEM_CLOCK
void IntrinsicLibrary::genSystemClock(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  Fortran::lower::genSystemClock(builder, loc, *args[0].getUnboxed(),
                                 *args[1].getUnboxed(), *args[2].getUnboxed());
}

// TRANSFER
fir::ExtendedValue
IntrinsicLibrary::genTransfer(mlir::Type resultType,
                              llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() >= 2); // args.size() == 2 when size argument is omitted.

  // Handle source argument
  auto source = builder.createBox(loc, args[0]);

  // Handle mold argument
  auto mold = builder.createBox(loc, args[1]);
  fir::BoxValue moldTmp = mold;
  auto moldRank = moldTmp.rank();

  bool absentSize = (args.size() == 2);

  // Create mutable fir.box to be passed to the runtime for the result.
  auto type = (moldRank == 0 && absentSize)
                  ? resultType
                  : builder.getVarLenSeqTy(resultType, 1);
  auto resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, type);

  if (moldRank == 0 && absentSize) {
    // This result is a scalar in this case.
    auto resultIrBox =
        fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

    Fortran::lower::genTransfer(builder, loc, resultIrBox, source, mold);
  } else {
    // The result is a rank one array in this case.
    auto resultIrBox =
        fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

    if (absentSize) {
      Fortran::lower::genTransfer(builder, loc, resultIrBox, source, mold);
    } else {
      auto sizeArg = fir::getBase(args[2]);
      Fortran::lower::genTransferSize(builder, loc, resultIrBox, source, mold,
                                      sizeArg);
    }
  }
  return readAndAddCleanUp(resultMutableBox, resultType,
                           "unexpected result for TRANSFER");
}

// TRANSPOSE
fir::ExtendedValue
IntrinsicLibrary::genTranspose(mlir::Type resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 1);

  // Handle source argument
  auto source = builder.createBox(loc, args[0]);

  // Create mutable fir.box to be passed to the runtime for the result.
  auto resultArrayType = builder.getVarLenSeqTy(resultType, 2);
  auto resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  auto resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);
  // Call runtime. The runtime is allocating the result.
  fir::runtime::genTranspose(builder, loc, resultIrBox, source);
  // Read result from mutable fir.box and add it to the list of temps to be
  // finalized by the StatementContext.
  return readAndAddCleanUp(resultMutableBox, resultType,
                           "unexpected result for TRANSPOSE");
}

// TRIM
fir::ExtendedValue
IntrinsicLibrary::genTrim(mlir::Type resultType,
                          llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  auto string = builder.createBox(loc, args[0]);
  // Create mutable fir.box to be passed to the runtime for the result.
  auto resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultType);
  auto resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);
  // Call runtime. The runtime is allocating the result.
  fir::runtime::genTrim(builder, loc, resultIrBox, string);
  // Read result from mutable fir.box and add it to the list of temps to be
  // finalized by the StatementContext.
  return readAndAddCleanUp(resultMutableBox, resultType, "TRIM");
}

// Compare two FIR values and return boolean result as i1.
template <Extremum extremum, ExtremumBehavior behavior>
static mlir::Value createExtremumCompare(mlir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         mlir::Value left, mlir::Value right) {
  static constexpr auto integerPredicate =
      extremum == Extremum::Max ? mlir::arith::CmpIPredicate::sgt
                                : mlir::arith::CmpIPredicate::slt;
  static constexpr auto orderedCmp = extremum == Extremum::Max
                                         ? mlir::arith::CmpFPredicate::OGT
                                         : mlir::arith::CmpFPredicate::OLT;
  auto type = left.getType();
  mlir::Value result;
  if (fir::isa_real(type)) {
    // Note: the signaling/quit aspect of the result required by IEEE
    // cannot currently be obtained with LLVM without ad-hoc runtime.
    if constexpr (behavior == ExtremumBehavior::IeeeMinMaximumNumber) {
      // Return the number if one of the inputs is NaN and the other is
      // a number.
      auto leftIsResult =
          builder.create<mlir::arith::CmpFOp>(loc, orderedCmp, left, right);
      auto rightIsNan = builder.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::UNE, right, right);
      result =
          builder.create<mlir::arith::OrIOp>(loc, leftIsResult, rightIsNan);
    } else if constexpr (behavior == ExtremumBehavior::IeeeMinMaximum) {
      // Always return NaNs if one the input is NaNs
      auto leftIsResult =
          builder.create<mlir::arith::CmpFOp>(loc, orderedCmp, left, right);
      auto leftIsNan = builder.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::UNE, left, left);
      result = builder.create<mlir::arith::OrIOp>(loc, leftIsResult, leftIsNan);
    } else if constexpr (behavior == ExtremumBehavior::MinMaxss) {
      // If the left is a NaN, return the right whatever it is.
      result =
          builder.create<mlir::arith::CmpFOp>(loc, orderedCmp, left, right);
    } else if constexpr (behavior == ExtremumBehavior::PgfortranLlvm) {
      // If one of the operand is a NaN, return left whatever it is.
      static constexpr auto unorderedCmp =
          extremum == Extremum::Max ? mlir::arith::CmpFPredicate::UGT
                                    : mlir::arith::CmpFPredicate::ULT;
      result =
          builder.create<mlir::arith::CmpFOp>(loc, unorderedCmp, left, right);
    } else {
      // TODO: ieeeMinNum/ieeeMaxNum
      static_assert(behavior == ExtremumBehavior::IeeeMinMaxNum,
                    "ieeeMinNum/ieeeMaxNum behavior not implemented");
    }
  } else if (fir::isa_integer(type)) {
    result =
        builder.create<mlir::arith::CmpIOp>(loc, integerPredicate, left, right);
  } else if (fir::isa_char(type)) {
    // TODO: ! character min and max is tricky because the result
    // length is the length of the longest argument!
    // So we may need a temp.
    TODO(loc, "CHARACTER min and max");
  }
  assert(result && "result must be defined");
  return result;
}

// UNPACK
fir::ExtendedValue
IntrinsicLibrary::genUnpack(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);

  // Handle required vector argument
  auto vector = builder.createBox(loc, args[0]);
  fir::BoxValue vectorTmp = builder.createBox(loc, args[0]);

  // Handle required mask argument
  fir::BoxValue maskBox = builder.createBox(loc, args[1]);
  auto mask = fir::getBase(maskBox);
  auto maskRank = maskBox.rank();

  // Handle required field argument
  auto field = builder.createBox(loc, args[2]);

  // Create mutable fir.box to be passed to the runtime for the result.
  auto resultArrayType = builder.getVarLenSeqTy(resultType, maskRank);
  auto resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  auto resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  fir::runtime::genUnpack(builder, loc, resultIrBox, vector, mask, field);

  return readAndAddCleanUp(resultMutableBox, resultType,
                           "unexpected result for UNPACK");
}

// VERIFY
fir::ExtendedValue
IntrinsicLibrary::genVerify(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 4);

  if (isAbsent(args[3])) {
    // Kind not specified, so call scan/verify runtime routine that is
    // specialized on the kind of characters in string.

    // Handle required string base arg
    auto stringBase = fir::getBase(args[0]);

    // Handle required set string base arg
    auto setBase = fir::getBase(args[1]);

    // Handle kind argument; it is the kind of character in this case
    auto kind =
        fir::factory::CharacterExprHelper{builder, loc}.getCharacterKind(
            stringBase.getType());

    // Get string length argument
    auto stringLen = fir::getLen(args[0]);

    // Get set string length argument
    auto setLen = fir::getLen(args[1]);

    // Handle optional back argument
    auto back = isAbsent(args[2])
                    ? builder.createIntegerConstant(loc, builder.getI1Type(), 0)
                    : fir::getBase(args[2]);

    return builder.createConvert(
        loc, resultType,
        fir::runtime::genVerify(builder, loc, kind, stringBase, stringLen,
                                setBase, setLen, back));
  }
  // else use the runtime descriptor version of scan/verify

  // Handle optional argument, back
  auto makeRefThenEmbox = [&](mlir::Value b) {
    auto logTy = fir::LogicalType::get(
        builder.getContext(), builder.getKindMap().defaultLogicalKind());
    auto temp = builder.createTemporary(loc, logTy);
    auto castb = builder.createConvert(loc, logTy, b);
    builder.create<fir::StoreOp>(loc, castb, temp);
    return builder.createBox(loc, temp);
  };
  auto back = fir::isUnboxedValue(args[2])
                  ? makeRefThenEmbox(*args[2].getUnboxed())
                  : builder.create<fir::AbsentOp>(
                        loc, fir::BoxType::get(builder.getI1Type()));

  // Handle required string argument
  auto string = builder.createBox(loc, args[0]);

  // Handle required set argument
  auto set = builder.createBox(loc, args[1]);

  // Handle kind argument
  auto kind = fir::getBase(args[3]);

  // Create result descriptor
  auto resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultType);
  auto resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  fir::runtime::genVerifyDescriptor(builder, loc, resultIrBox, string, set,
                                    back, kind);

  // Handle cleanup of allocatable result descriptor and return
  return readAndAddCleanUp(resultMutableBox, resultType, "VERIFY");
}

// MAXLOC
fir::ExtendedValue
IntrinsicLibrary::genMaxloc(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  return genExtremumloc(fir::runtime::genMaxloc, fir::runtime::genMaxlocDim,
                        resultType, builder, loc, stmtCtx,
                        "unexpected result for Maxloc", args);
}

// MAXVAL
fir::ExtendedValue
IntrinsicLibrary::genMaxval(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  return genExtremumVal(fir::runtime::genMaxval, fir::runtime::genMaxvalDim,
                        fir::runtime::genMaxvalChar, resultType, builder, loc,
                        stmtCtx, "unexpected result for Maxval", args);
}

// MINLOC
fir::ExtendedValue
IntrinsicLibrary::genMinloc(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  return genExtremumloc(fir::runtime::genMinloc, fir::runtime::genMinlocDim,
                        resultType, builder, loc, stmtCtx,
                        "unexpected result for Minloc", args);
}

// MINVAL
fir::ExtendedValue
IntrinsicLibrary::genMinval(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  return genExtremumVal(fir::runtime::genMinval, fir::runtime::genMinvalDim,
                        fir::runtime::genMinvalChar, resultType, builder, loc,
                        stmtCtx, "unexpected result for Minval", args);
}

// MIN and MAX
template <Extremum extremum, ExtremumBehavior behavior>
mlir::Value IntrinsicLibrary::genExtremum(mlir::Type,
                                          llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() >= 1);
  mlir::Value result = args[0];
  for (auto arg : args.drop_front()) {
    auto mask =
        createExtremumCompare<extremum, behavior>(loc, builder, result, arg);
    result = builder.create<mlir::SelectOp>(loc, mask, result, arg);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Argument lowering rules interface
//===----------------------------------------------------------------------===//

const Fortran::lower::IntrinsicArgumentLoweringRules *
Fortran::lower::getIntrinsicArgumentLowering(llvm::StringRef intrinsicName) {
  if (const auto &handler = findIntrinsicHandler(intrinsicName))
    if (!handler->argLoweringRules.hasDefaultRules())
      return &handler->argLoweringRules;
  return nullptr;
}

/// Return how argument \p argName should be lowered given the rules for the
/// intrinsic function.
Fortran::lower::LowerIntrinsicArgAs Fortran::lower::lowerIntrinsicArgumentAs(
    mlir::Location loc, const IntrinsicArgumentLoweringRules &rules,
    llvm::StringRef argName) {
  for (const auto &arg : rules.args) {
    if (arg.name && arg.name == argName)
      return arg.lowerAs;
  }
  fir::emitFatalError(
      loc, "internal: unknown intrinsic argument name in lowering '" + argName +
               "'");
}

//===----------------------------------------------------------------------===//
// Public intrinsic call helpers
//===----------------------------------------------------------------------===//

fir::ExtendedValue
Fortran::lower::genIntrinsicCall(fir::FirOpBuilder &builder, mlir::Location loc,
                                 llvm::StringRef name,
                                 llvm::Optional<mlir::Type> resultType,
                                 llvm::ArrayRef<fir::ExtendedValue> args,
                                 Fortran::lower::StatementContext &stmtCtx) {
  return IntrinsicLibrary{builder, loc, &stmtCtx}.genIntrinsicCall(
      name, resultType, args);
}

mlir::Value Fortran::lower::genMax(fir::FirOpBuilder &builder,
                                   mlir::Location loc,
                                   llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() > 0 && "max requires at least one argument");
  return IntrinsicLibrary{builder, loc}
      .genExtremum<Extremum::Max, ExtremumBehavior::MinMaxss>(args[0].getType(),
                                                              args);
}

mlir::Value Fortran::lower::genMin(fir::FirOpBuilder &builder,
                                   mlir::Location loc,
                                   llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() > 0 && "min requires at least one argument");
  return IntrinsicLibrary{builder, loc}
      .genExtremum<Extremum::Min, ExtremumBehavior::MinMaxss>(args[0].getType(),
                                                              args);
}

mlir::Value Fortran::lower::genPow(fir::FirOpBuilder &builder,
                                   mlir::Location loc, mlir::Type type,
                                   mlir::Value x, mlir::Value y) {
  return IntrinsicLibrary{builder, loc}.genRuntimeCall("pow", type, {x, y});
}

mlir::SymbolRefAttr Fortran::lower::getUnrestrictedIntrinsicSymbolRefAttr(
    fir::FirOpBuilder &builder, mlir::Location loc, llvm::StringRef name,
    mlir::FunctionType signature) {
  return IntrinsicLibrary{builder, loc}.getUnrestrictedIntrinsicSymbolRefAttr(
      name, signature);
}
