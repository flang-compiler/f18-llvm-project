//===-- IO.cpp -- I/O statement lowering ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/IO.h"
#include "../../runtime/io-api.h"
#include "RTBuilder.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Runtime.h"
#include "flang/Lower/Utils.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#define TODO() llvm_unreachable("not yet implemented")

using namespace Fortran::runtime::io;

#define NAMIFY_HELPER(X) #X
#define NAMIFY(X) NAMIFY_HELPER(IONAME(X))
#define mkIOKey(X) mkKey(IONAME(X))

namespace Fortran::lower {
/// Static table of IO runtime calls
///
/// This logical map contains the name and type builder function for each IO
/// runtime function listed in the tuple. This table is fully constructed at
/// compile-time. Use the `mkIOKey` macro to access the table.
static constexpr std::tuple<
    mkIOKey(BeginInternalArrayListOutput), mkIOKey(BeginInternalArrayListInput),
    mkIOKey(BeginInternalArrayFormattedOutput),
    mkIOKey(BeginInternalArrayFormattedInput), mkIOKey(BeginInternalListOutput),
    mkIOKey(BeginInternalListInput), mkIOKey(BeginInternalFormattedOutput),
    mkIOKey(BeginInternalFormattedInput), mkIOKey(BeginInternalNamelistOutput),
    mkIOKey(BeginInternalNamelistInput), mkIOKey(BeginExternalListOutput),
    mkIOKey(BeginExternalListInput), mkIOKey(BeginExternalFormattedOutput),
    mkIOKey(BeginExternalFormattedInput), mkIOKey(BeginUnformattedOutput),
    mkIOKey(BeginUnformattedInput), mkIOKey(BeginExternalNamelistOutput),
    mkIOKey(BeginExternalNamelistInput), mkIOKey(BeginAsynchronousOutput),
    mkIOKey(BeginAsynchronousInput), mkIOKey(BeginWait), mkIOKey(BeginWaitAll),
    mkIOKey(BeginClose), mkIOKey(BeginFlush), mkIOKey(BeginBackspace),
    mkIOKey(BeginEndfile), mkIOKey(BeginRewind), mkIOKey(BeginOpenUnit),
    mkIOKey(BeginOpenNewUnit), mkIOKey(BeginInquireUnit),
    mkIOKey(BeginInquireFile), mkIOKey(BeginInquireIoLength),
    mkIOKey(EnableHandlers), mkIOKey(SetAdvance), mkIOKey(SetBlank),
    mkIOKey(SetDecimal), mkIOKey(SetDelim), mkIOKey(SetPad), mkIOKey(SetPos),
    mkIOKey(SetRec), mkIOKey(SetRound), mkIOKey(SetSign),
    mkIOKey(OutputDescriptor), mkIOKey(InputDescriptor),
    mkIOKey(OutputUnformattedBlock), mkIOKey(InputUnformattedBlock),
    mkIOKey(OutputInteger64), mkIOKey(InputInteger), mkIOKey(OutputReal32),
    mkIOKey(InputReal32), mkIOKey(OutputReal64), mkIOKey(InputReal64),
    mkIOKey(OutputComplex64), mkIOKey(OutputComplex32), mkIOKey(OutputAscii),
    mkIOKey(InputAscii), mkIOKey(OutputLogical), mkIOKey(InputLogical),
    mkIOKey(SetAccess), mkIOKey(SetAction), mkIOKey(SetAsynchronous),
    mkIOKey(SetEncoding), mkIOKey(SetForm), mkIOKey(SetPosition),
    mkIOKey(SetRecl), mkIOKey(SetStatus), mkIOKey(SetFile), mkIOKey(GetNewUnit),
    mkIOKey(GetSize), mkIOKey(GetIoLength), mkIOKey(GetIoMsg),
    mkIOKey(InquireCharacter), mkIOKey(InquireLogical),
    mkIOKey(InquirePendingId), mkIOKey(InquireInteger64),
    mkIOKey(EndIoStatement)>
    newIOTable;
} // namespace Fortran::lower

namespace {
struct ErrorHandling {
  bool hasErrorHandlers() const { return ioStat || err || end || eor || ioMsg; }
  bool ioStat{};
  bool err{};
  bool end{};
  bool eor{};
  bool ioMsg{};
  const Fortran::semantics::SomeExpr *ioStatExpr{};
  const Fortran::semantics::SomeExpr *ioMsgExpr{};
};
} // namespace

using namespace Fortran::lower;

/// Helper function to retrieve the name of the IO function given the key `A`
template <typename A>
static constexpr const char *getName() {
  return std::get<A>(newIOTable).name;
}

/// Helper function to retrieve the type model signature builder of the IO
/// function as defined by the key `A`
template <typename A>
static constexpr FuncTypeBuilderFunc getTypeModel() {
  return std::get<A>(newIOTable).getTypeModel();
}

inline int64_t getLength(mlir::Type argTy) {
  return argTy.cast<fir::SequenceType>().getShape()[0];
}

/// Get (or generate) the MLIR FuncOp for a given IO runtime function.
template <typename E>
static mlir::FuncOp getIORuntimeFunc(Fortran::lower::FirOpBuilder &builder) {
  auto name = getName<E>();
  auto func = builder.getNamedFunction(name);
  if (func)
    return func;
  auto funTy = getTypeModel<E>()(builder.getContext());
  func = builder.createFunction(name, funTy);
  func.setAttr("fir.runtime", builder.getUnitAttr());
  func.setAttr("fir.io", builder.getUnitAttr());
  return func;
}

/// Generate a call to end an IO statement
static mlir::Value genEndIO(Fortran::lower::AbstractConverter &converter,
                            mlir::Location loc, mlir::Value cookie,
                            const ErrorHandling &eh) {
  // Terminate IO
  auto &builder = converter.getFirOpBuilder();
  if (eh.ioMsg) {
    TODO(); // eh.ioMsgExpr is never set
    auto getMsg = getIORuntimeFunc<mkIOKey(GetIoMsg)>(builder);
    auto var = converter.genExprAddr(eh.ioMsgExpr, loc);
    mlir::Value varLen{}; // FIXME
    llvm::SmallVector<mlir::Value, 3> args{cookie, var, varLen};
    builder.create<mlir::CallOp>(loc, getMsg, args);
  }
  auto endIOFunc = getIORuntimeFunc<mkIOKey(EndIoStatement)>(builder);
  llvm::SmallVector<mlir::Value, 1> endArgs{cookie};
  auto call = builder.create<mlir::CallOp>(loc, endIOFunc, endArgs);
  if (eh.ioStat) {
    TODO(); // eh.ioStatExpr is never set
    auto ioStatVar = converter.genExprAddr(eh.ioStatExpr, loc);
    builder.create<fir::StoreOp>(loc, call.getResult(0), ioStatVar);
  }
  return eh.hasErrorHandlers() ? call.getResult(0) : mlir::Value{};
}

/// Make the next `call` in the IO statement conditional on the result `ok`.  If
/// the current call returns `ok==false`, we do not continue to call the IO
/// statement's suboperations. This will lend itself to deeply nested
/// conditionals (and very branch heavy code) for statements with a large number
/// of suboperations. New conditional regions are created lazily.
static void makeNextConditionalOn(Fortran::lower::FirOpBuilder &builder,
                                  mlir::Location loc, mlir::Value ok,
                                  mlir::OpBuilder::InsertPoint &insertPt) {
  if (!ok)
    return;
  // Last call returned the bool `ok`.
  auto where = builder.create<fir::WhereOp>(loc, ok, /*withOtherwise=*/false);
  if (!insertPt.isSet())
    insertPt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(&where.whereRegion().front());
}

/// Get the OutputXyz routine to output a value with type `type`.
static mlir::FuncOp getOutputRuntimeFunc(Fortran::lower::FirOpBuilder &builder,
                                         mlir::Type type) {
  if (auto ty = type.dyn_cast<mlir::IntegerType>()) {
    if (ty.getWidth() == 1)
      return getIORuntimeFunc<mkIOKey(OutputLogical)>(builder);
    return getIORuntimeFunc<mkIOKey(OutputInteger64)>(builder);
  }
  if (auto ty = type.dyn_cast<mlir::FloatType>()) {
    if (ty.getWidth() <= 32)
      return getIORuntimeFunc<mkIOKey(OutputReal32)>(builder);
    return getIORuntimeFunc<mkIOKey(OutputReal64)>(builder);
  }
  if (auto ty = type.dyn_cast<fir::CplxType>()) {
    if (ty.getFKind() <= 4)
      return getIORuntimeFunc<mkIOKey(OutputComplex32)>(builder);
    return getIORuntimeFunc<mkIOKey(OutputComplex64)>(builder);
  }
  if (auto ty = type.dyn_cast<fir::LogicalType>())
    return getIORuntimeFunc<mkIOKey(OutputLogical)>(builder);
  if (auto ty = type.dyn_cast<fir::BoxType>())
    return getIORuntimeFunc<mkIOKey(OutputDescriptor)>(builder);
  return getIORuntimeFunc<mkIOKey(OutputAscii)>(builder);
}

/// The I/O library interface requires that COMPLEX and CHARACTER typed values
/// be extracted and passed as separate values.
static llvm::SmallVector<mlir::Value, 4>
splitArguments(Fortran::lower::FirOpBuilder &builder, mlir::Location loc,
               mlir::Value arg) {
  auto type = arg.getType();
  if (builder.isCharacter(type)) {
    auto dataLen = builder.materializeCharacter(arg);
    return {dataLen.first, dataLen.second};
  }
  if (fir::isa_complex(type)) {
    auto parts = builder.extractParts(arg);
    return {parts.first, parts.second};
  }
  return {arg};
}

/// Generate a call to an Output I/O function.
/// The specific call is determined dynamically by the argument type.
static void genOutputRuntimeFunc(Fortran::lower::FirOpBuilder &builder,
                                 mlir::Location loc, mlir::Type argType,
                                 mlir::Value cookie,
                                 llvm::SmallVector<mlir::Value, 4> &operands,
                                 mlir::OpBuilder::InsertPoint &insertPt,
                                 mlir::Value &lastOk) {
  int i = 1;
  llvm::SmallVector<mlir::Value, 4> actuals{cookie};
  auto outputFunc = getOutputRuntimeFunc(builder, argType);

  for (auto &op : operands)
    actuals.emplace_back(
        builder.createConvert(loc, outputFunc.getType().getInput(i++), op));
  makeNextConditionalOn(builder, loc, lastOk, insertPt);
  auto ok = builder.create<mlir::CallOp>(loc, outputFunc, actuals);
  lastOk = ok.getResult(0);
}

/// Get the InputXyz routine to input a value with type `type`.
static mlir::FuncOp getInputRuntimeFunc(Fortran::lower::FirOpBuilder &builder,
                                        mlir::Type type, int &intKind) {
  if (auto ty = type.dyn_cast<mlir::IntegerType>()) {
    if (ty.getWidth() == 1)
      return getIORuntimeFunc<mkIOKey(InputLogical)>(builder);
    intKind = ty.getWidth() / 8;
    return getIORuntimeFunc<mkIOKey(InputInteger)>(builder);
  }
  if (auto ty = type.dyn_cast<mlir::FloatType>()) {
    if (ty.getWidth() <= 32)
      return getIORuntimeFunc<mkIOKey(InputReal32)>(builder);
    return getIORuntimeFunc<mkIOKey(InputReal64)>(builder);
  }
  if (auto ty = type.dyn_cast<fir::CplxType>()) {
    if (ty.getFKind() <= 4)
      return getIORuntimeFunc<mkIOKey(InputReal32)>(builder);
    return getIORuntimeFunc<mkIOKey(InputReal64)>(builder);
  }
  if (auto ty = type.dyn_cast<fir::LogicalType>())
    return getIORuntimeFunc<mkIOKey(InputLogical)>(builder);
  if (auto ty = type.dyn_cast<fir::BoxType>())
    return getIORuntimeFunc<mkIOKey(InputDescriptor)>(builder);
  return getIORuntimeFunc<mkIOKey(InputAscii)>(builder);
}

/// Generate a call to an Output I/O function.
/// The specific call is determined dynamically by the argument type.
static void genInputRuntimeFunc(Fortran::lower::FirOpBuilder &builder,
                                mlir::Location loc, mlir::Type refType,
                                mlir::Value cookie, mlir::Value &var,
                                mlir::OpBuilder::InsertPoint &insertPt,
                                mlir::Value &lastOk) {
  auto argType = refType.cast<fir::ReferenceType>().getEleTy();
  int intKind = 0;
  auto inputFunc = getInputRuntimeFunc(builder, argType, intKind);
  if (auto ty = argType.dyn_cast<fir::CplxType>()) {
    var = builder.create<fir::CoordinateOp>(
        loc, builder.getRefType(builder.getComplexPartType(ty)), var,
        llvm::SmallVector<mlir::Value, 1>{builder.create<mlir::ConstantOp>(
            loc, builder.getI32IntegerAttr(0))});
  }
  auto cvtTy = inputFunc.getType().getInput(1);
  auto cvt = builder.createConvert(loc, cvtTy, var);
  if (intKind != 0) {
    auto kind = builder.create<mlir::ConstantOp>(
        loc, builder.getI32IntegerAttr(intKind));
    makeNextConditionalOn(builder, loc, lastOk, insertPt);
    auto ok = builder.create<mlir::CallOp>(
        loc, inputFunc, llvm::SmallVector<mlir::Value, 3>{cookie, cvt, kind});
    lastOk = ok.getResult(0);
  } else {
    makeNextConditionalOn(builder, loc, lastOk, insertPt);
    auto ok = builder.create<mlir::CallOp>(
        loc, inputFunc, llvm::SmallVector<mlir::Value, 2>{cookie, cvt});
    lastOk = ok.getResult(0);
  }
  if (auto ty = argType.dyn_cast<fir::CplxType>()) {
    auto ptr = builder.create<fir::CoordinateOp>(
        loc, builder.getRefType(builder.getComplexPartType(ty)), var,
        llvm::SmallVector<mlir::Value, 1>{builder.create<mlir::ConstantOp>(
            loc, builder.getI32IntegerAttr(1))});
    auto cst = builder.createConvert(loc, cvtTy, ptr);
    makeNextConditionalOn(builder, loc, lastOk, insertPt);
    auto ok = builder.create<mlir::CallOp>(
        loc, inputFunc, llvm::SmallVector<mlir::Value, 2>{cookie, cst});
    lastOk = ok.getResult(0);
  }
}

//===----------------------------------------------------------------------===//
// Default argument generation.
//===----------------------------------------------------------------------===//

static mlir::Value getDefaultFilename(Fortran::lower::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Type toType) {
  mlir::Value null =
      builder.create<mlir::ConstantOp>(loc, builder.getI64IntegerAttr(0));
  return builder.createConvert(loc, toType, null);
}

static mlir::Value getDefaultLineNo(Fortran::lower::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Type toType) {
  return builder.create<mlir::ConstantOp>(loc,
                                          builder.getIntegerAttr(toType, 0));
}

static mlir::Value getDefaultScratch(Fortran::lower::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Type toType) {
  mlir::Value null =
      builder.create<mlir::ConstantOp>(loc, builder.getI64IntegerAttr(0));
  return builder.createConvert(loc, toType, null);
}

static mlir::Value getDefaultScratchLen(Fortran::lower::FirOpBuilder &builder,
                                        mlir::Location loc, mlir::Type toType) {
  return builder.create<mlir::ConstantOp>(loc,
                                          builder.getIntegerAttr(toType, 0));
}

/// Lower a string literal. Many arguments to the runtime are conveyed as
/// Fortran CHARACTER literals.
template <typename A>
static std::tuple<mlir::Value, mlir::Value, mlir::Value>
lowerStringLit(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
               const A &syntax, mlir::Type ty0, mlir::Type ty1,
               mlir::Type ty2 = {}) {
  auto &builder = converter.getFirOpBuilder();
  auto *expr = Fortran::semantics::GetExpr(syntax);
  auto str = converter.genExprValue(expr, loc);
  auto dataLen = builder.materializeCharacter(str);
  auto buff = builder.createConvert(loc, ty0, dataLen.first);
  auto len = builder.createConvert(loc, ty1, dataLen.second);
  if (ty2) {
    auto kindVal = builder.getCharacterKind(str.getType());
    auto kind = builder.create<mlir::ConstantOp>(
        loc, builder.getIntegerAttr(ty2, kindVal));
    return {buff, len, kind};
  }
  return {buff, len, mlir::Value{}};
}

/// Pass the body of the FORMAT statement in as if it were a CHARACTER literal
/// constant. TODO: This feels dumb. Isn't there a front-end method to get this?
/// If not, why not?
static std::tuple<mlir::Value, mlir::Value, mlir::Value>
lowerSourceTextAsStringLit(Fortran::lower::AbstractConverter &converter,
                           mlir::Location loc, llvm::StringRef text,
                           mlir::Type ty0, mlir::Type ty1) {
  text = text.drop_front(text.find('('));
  text = text.take_front(text.rfind(')') + 1);
  auto &builder = converter.getFirOpBuilder();
  auto lit = builder.createStringLit(
      loc, /*FIXME*/ fir::CharacterType::get(builder.getContext(), 1), text);
  auto data = builder.materializeCharacter(lit);
  auto buff = builder.createConvert(loc, ty0, data.first);
  auto len = builder.createConvert(loc, ty1, data.second);
  return {buff, len, mlir::Value{}};
}

//===----------------------------------------------------------------------===//
// Handle I/O statement specifiers.
// These are threaded together for a single statement via the passed cookie.
//===----------------------------------------------------------------------===//

/// Generic to build an integral argument to the runtime.
template <typename A, typename B>
mlir::Value genIntIOOption(Fortran::lower::AbstractConverter &converter,
                           mlir::Location loc, mlir::Value cookie,
                           const B &spec) {
  auto &builder = converter.getFirOpBuilder();
  mlir::FuncOp ioFunc = getIORuntimeFunc<A>(builder);
  mlir::FunctionType ioFuncTy = ioFunc.getType();
  auto expr = converter.genExprValue(Fortran::semantics::GetExpr(spec.v), loc);
  auto val = builder.createConvert(loc, ioFuncTy.getInput(1), expr);
  llvm::SmallVector<mlir::Value, 4> ioArgs = {cookie, val};
  return builder.create<mlir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
}

/// Generic to build a string argument to the runtime. This passes a CHARACTER
/// as a pointer to the buffer and a LEN parameter.
template <typename A, typename B>
mlir::Value genCharIOOption(Fortran::lower::AbstractConverter &converter,
                            mlir::Location loc, mlir::Value cookie,
                            const B &spec) {
  auto &builder = converter.getFirOpBuilder();
  mlir::FuncOp ioFunc = getIORuntimeFunc<A>(builder);
  mlir::FunctionType ioFuncTy = ioFunc.getType();
  auto tup = lowerStringLit(converter, loc, spec, ioFuncTy.getInput(1),
                            ioFuncTy.getInput(2));
  llvm::SmallVector<mlir::Value, 4> ioArgs = {cookie, std::get<0>(tup),
                                              std::get<1>(tup)};
  return builder.create<mlir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
}

template <typename A>
mlir::Value genIOOption(Fortran::lower::AbstractConverter &converter,
                        mlir::Location loc, mlir::Value cookie, const A &spec) {
  // default case: do nothing
  return {};
}

template <>
mlir::Value genIOOption<Fortran::parser::FileNameExpr>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::FileNameExpr &spec) {
  auto &builder = converter.getFirOpBuilder();
  // has an extra KIND argument
  auto ioFunc = getIORuntimeFunc<mkIOKey(SetFile)>(builder);
  mlir::FunctionType ioFuncTy = ioFunc.getType();
  auto tup = lowerStringLit(converter, loc, spec, ioFuncTy.getInput(1),
                            ioFuncTy.getInput(2), ioFuncTy.getInput(3));
  llvm::SmallVector<mlir::Value, 4> ioArgs{cookie, std::get<0>(tup),
                                           std::get<1>(tup), std::get<2>(tup)};
  return builder.create<mlir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
}

template <>
mlir::Value genIOOption<Fortran::parser::ConnectSpec::CharExpr>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::ConnectSpec::CharExpr &spec) {
  auto &builder = converter.getFirOpBuilder();
  mlir::FuncOp ioFunc;
  switch (std::get<Fortran::parser::ConnectSpec::CharExpr::Kind>(spec.t)) {
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Access:
    ioFunc = getIORuntimeFunc<mkIOKey(SetAccess)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Action:
    ioFunc = getIORuntimeFunc<mkIOKey(SetAction)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Asynchronous:
    ioFunc = getIORuntimeFunc<mkIOKey(SetAsynchronous)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Blank:
    ioFunc = getIORuntimeFunc<mkIOKey(SetBlank)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Decimal:
    ioFunc = getIORuntimeFunc<mkIOKey(SetDecimal)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Delim:
    ioFunc = getIORuntimeFunc<mkIOKey(SetDelim)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Encoding:
    ioFunc = getIORuntimeFunc<mkIOKey(SetEncoding)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Form:
    ioFunc = getIORuntimeFunc<mkIOKey(SetForm)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Pad:
    ioFunc = getIORuntimeFunc<mkIOKey(SetPad)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Position:
    ioFunc = getIORuntimeFunc<mkIOKey(SetPosition)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Round:
    ioFunc = getIORuntimeFunc<mkIOKey(SetRound)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Sign:
    ioFunc = getIORuntimeFunc<mkIOKey(SetSign)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Convert:
    llvm_unreachable("CONVERT not part of the runtime::io interface");
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Dispose:
    llvm_unreachable("DISPOSE not part of the runtime::io interface");
  }
  mlir::FunctionType ioFuncTy = ioFunc.getType();
  auto tup = lowerStringLit(
      converter, loc, std::get<Fortran::parser::ScalarDefaultCharExpr>(spec.t),
      ioFuncTy.getInput(1), ioFuncTy.getInput(2));
  llvm::SmallVector<mlir::Value, 4> ioArgs = {cookie, std::get<0>(tup),
                                              std::get<1>(tup)};
  return builder.create<mlir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
}

template <>
mlir::Value genIOOption<Fortran::parser::ConnectSpec::Recl>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::ConnectSpec::Recl &spec) {
  return genIntIOOption<mkIOKey(SetRecl)>(converter, loc, cookie, spec);
}

template <>
mlir::Value genIOOption<Fortran::parser::StatusExpr>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::StatusExpr &spec) {
  return genCharIOOption<mkIOKey(SetStatus)>(converter, loc, cookie, spec.v);
}

template <>
mlir::Value
genIOOption<Fortran::parser::Name>(Fortran::lower::AbstractConverter &converter,
                                   mlir::Location loc, mlir::Value cookie,
                                   const Fortran::parser::Name &spec) {
  // namelist
  llvm_unreachable("not implemented");
}

template <>
mlir::Value genIOOption<Fortran::parser::IoControlSpec::CharExpr>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::IoControlSpec::CharExpr &spec) {
  auto &builder = converter.getFirOpBuilder();
  mlir::FuncOp ioFunc;
  switch (std::get<Fortran::parser::IoControlSpec::CharExpr::Kind>(spec.t)) {
  case Fortran::parser::IoControlSpec::CharExpr::Kind::Advance:
    ioFunc = getIORuntimeFunc<mkIOKey(SetAdvance)>(builder);
    break;
  case Fortran::parser::IoControlSpec::CharExpr::Kind::Blank:
    ioFunc = getIORuntimeFunc<mkIOKey(SetBlank)>(builder);
    break;
  case Fortran::parser::IoControlSpec::CharExpr::Kind::Decimal:
    ioFunc = getIORuntimeFunc<mkIOKey(SetDecimal)>(builder);
    break;
  case Fortran::parser::IoControlSpec::CharExpr::Kind::Delim:
    ioFunc = getIORuntimeFunc<mkIOKey(SetDelim)>(builder);
    break;
  case Fortran::parser::IoControlSpec::CharExpr::Kind::Pad:
    ioFunc = getIORuntimeFunc<mkIOKey(SetPad)>(builder);
    break;
  case Fortran::parser::IoControlSpec::CharExpr::Kind::Round:
    ioFunc = getIORuntimeFunc<mkIOKey(SetRound)>(builder);
    break;
  case Fortran::parser::IoControlSpec::CharExpr::Kind::Sign:
    ioFunc = getIORuntimeFunc<mkIOKey(SetSign)>(builder);
    break;
  }
  mlir::FunctionType ioFuncTy = ioFunc.getType();
  auto tup = lowerStringLit(
      converter, loc, std::get<Fortran::parser::ScalarDefaultCharExpr>(spec.t),
      ioFuncTy.getInput(1), ioFuncTy.getInput(2));
  llvm::SmallVector<mlir::Value, 4> ioArgs = {cookie, std::get<0>(tup),
                                              std::get<1>(tup)};
  return builder.create<mlir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
}

template <>
mlir::Value genIOOption<Fortran::parser::IoControlSpec::Asynchronous>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie,
    const Fortran::parser::IoControlSpec::Asynchronous &spec) {
  return genCharIOOption<mkIOKey(SetAsynchronous)>(converter, loc, cookie,
                                                   spec.v);
}

template <>
mlir::Value genIOOption<Fortran::parser::IdVariable>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::IdVariable &spec) {
  llvm_unreachable("asynchronous ID not implemented");
}

template <>
mlir::Value genIOOption<Fortran::parser::IoControlSpec::Pos>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::IoControlSpec::Pos &spec) {
  return genIntIOOption<mkIOKey(SetPos)>(converter, loc, cookie, spec);
}
template <>
mlir::Value genIOOption<Fortran::parser::IoControlSpec::Rec>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::IoControlSpec::Rec &spec) {
  return genIntIOOption<mkIOKey(SetRec)>(converter, loc, cookie, spec);
}

//===----------------------------------------------------------------------===//
// Gathering of I/O statement error specifiers (if any).
//===----------------------------------------------------------------------===//

template <typename A>
void genErrorOption(ErrorHandling &handle, const A &) {}
template <>
void genErrorOption<Fortran::parser::MsgVariable>(
    ErrorHandling &eh, const Fortran::parser::MsgVariable &) {
  eh.ioMsg = true;
}
template <>
void genErrorOption<Fortran::parser::StatVariable>(
    ErrorHandling &eh, const Fortran::parser::StatVariable &) {
  eh.ioStat = true;
}
template <>
void genErrorOption<Fortran::parser::ErrLabel>(
    ErrorHandling &eh, const Fortran::parser::ErrLabel &) {
  eh.err = true;
}
template <>
void genErrorOption<Fortran::parser::EndLabel>(
    ErrorHandling &eh, const Fortran::parser::EndLabel &) {
  eh.end = true;
}
template <>
void genErrorOption<Fortran::parser::EorLabel>(
    ErrorHandling &eh, const Fortran::parser::EorLabel &) {
  eh.eor = true;
}

template <typename SEEK, typename A>
static bool hasX(const A &list) {
  for (const auto &spec : list)
    if (std::holds_alternative<SEEK>(spec.u))
      return true;
  return false;
}

template <typename SEEK, typename A>
static bool hasMem(const A &stmt) {
  return hasX<SEEK>(stmt.v);
}

/// Get the sought expression from the specifier list.
template <typename SEEK, typename A>
static const Fortran::semantics::SomeExpr *getExpr(const A &stmt) {
  for (const auto &spec : stmt.v)
    if (auto *f = std::get_if<SEEK>(&spec.u))
      return Fortran::semantics::GetExpr(f->v);
  llvm_unreachable("must have a file unit");
}

/// For each specifier, build the appropriate call, threading the cookie, and
/// returning the insertion point as to the initial context. If there are no
/// specs, the insertion point is undefined.
template <typename A>
static mlir::OpBuilder::InsertPoint
threadSpecs(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
            mlir::Value cookie, const A &list, mlir::Value &ok) {
  auto &builder = converter.getFirOpBuilder();
  mlir::OpBuilder::InsertPoint insertPt;
  for (const auto &spec : list) {
    makeNextConditionalOn(builder, loc, ok, insertPt);
    ok = std::visit(Fortran::common::visitors{[&](const auto &x) {
                      return genIOOption(converter, loc, cookie, x);
                    }},
                    spec.u);
  }
  return insertPt;
}

template <typename A>
static void threadErrors(Fortran::lower::AbstractConverter &converter,
                         mlir::Location loc, mlir::Value cookie, const A &list,
                         ErrorHandling &eh) {
  for (const auto &spec : list)
    std::visit(Fortran::common::visitors{[&](const auto &x) {
                 genErrorOption(eh, x);
               }},
               spec.u);
  if (eh.hasErrorHandlers()) {
    auto &builder = converter.getFirOpBuilder();
    mlir::FuncOp ioFunc = getIORuntimeFunc<mkIOKey(EnableHandlers)>(builder);
    mlir::FunctionType ioFuncTy = ioFunc.getType();
    auto hasIoStat = builder.create<mlir::ConstantOp>(
        loc, builder.getIntegerAttr(ioFuncTy.getInput(1), eh.ioStat));
    auto hasErr = builder.create<mlir::ConstantOp>(
        loc, builder.getIntegerAttr(ioFuncTy.getInput(2), eh.err));
    auto hasEnd = builder.create<mlir::ConstantOp>(
        loc, builder.getIntegerAttr(ioFuncTy.getInput(3), eh.end));
    auto hasEor = builder.create<mlir::ConstantOp>(
        loc, builder.getIntegerAttr(ioFuncTy.getInput(4), eh.eor));
    auto hasIoMsg = builder.create<mlir::ConstantOp>(
        loc, builder.getIntegerAttr(ioFuncTy.getInput(5), eh.ioMsg));
    llvm::SmallVector<mlir::Value, 4> ioArgs = {cookie, hasIoStat, hasErr,
                                                hasEnd, hasEor,    hasIoMsg};
    builder.create<mlir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
  }
}

//===----------------------------------------------------------------------===//
// Data transfer helpers
//===----------------------------------------------------------------------===//

template <typename SEEK, typename A>
static bool hasIOControl(const A &stmt) {
  return hasX<SEEK>(stmt.controls);
}

template <typename SEEK, typename A>
static const auto *getIOControl(const A &stmt) {
  for (const auto &spec : stmt.controls)
    if (const auto *result = std::get_if<SEEK>(&spec.u))
      return result;
  return static_cast<const SEEK *>(nullptr);
}

/// returns true iff the expression in the parse tree is not really a format but
/// rather a namelist variable.
template <typename A>
static bool formatIsActuallyNamelist(const A &format) {
  if (auto *e = std::get_if<Fortran::parser::Expr>(&format.u)) {
    auto *expr = Fortran::semantics::GetExpr(*e);
    if (const Fortran::semantics::Symbol *y =
            Fortran::evaluate::UnwrapWholeSymbolDataRef(*expr))
      return y->has<Fortran::semantics::NamelistDetails>();
  }
  return false;
}

template <typename A>
static bool isDataTransferFormatted(const A &stmt) {
  if (stmt.format)
    return !formatIsActuallyNamelist(*stmt.format);
  return hasIOControl<Fortran::parser::Format>(stmt);
}
template <>
constexpr bool isDataTransferFormatted<Fortran::parser::PrintStmt>(
    const Fortran::parser::PrintStmt &) {
  return true; // PRINT is always formatted
}

template <typename A>
static bool isDataTransferList(const A &stmt) {
  if (stmt.format)
    return std::holds_alternative<Fortran::parser::Star>(stmt.format->u);
  if (auto *mem = getIOControl<Fortran::parser::Format>(stmt))
    return std::holds_alternative<Fortran::parser::Star>(mem->u);
  return false;
}
template <>
bool isDataTransferList<Fortran::parser::PrintStmt>(
    const Fortran::parser::PrintStmt &stmt) {
  return std::holds_alternative<Fortran::parser::Star>(
      std::get<Fortran::parser::Format>(stmt.t).u);
}

template <typename A>
static bool isDataTransferInternal(const A &stmt) {
  if (stmt.iounit.has_value())
    return std::holds_alternative<Fortran::parser::Variable>(stmt.iounit->u);
  if (auto *unit = getIOControl<Fortran::parser::IoUnit>(stmt))
    return std::holds_alternative<Fortran::parser::Variable>(unit->u);
  return false;
}
template <>
constexpr bool isDataTransferInternal<Fortran::parser::PrintStmt>(
    const Fortran::parser::PrintStmt &) {
  return false;
}

static bool hasNonDefaultCharKind(const Fortran::parser::Variable &var) {
  // TODO
  return false;
}

template <typename A>
static bool isDataTransferInternalNotDefaultKind(const A &stmt) {
  // same as isDataTransferInternal, but the KIND of the expression is not the
  // default KIND.
  if (stmt.iounit.has_value())
    if (auto *var = std::get_if<Fortran::parser::Variable>(&stmt.iounit->u))
      return hasNonDefaultCharKind(*var);
  if (auto *unit = getIOControl<Fortran::parser::IoUnit>(stmt))
    if (auto *var = std::get_if<Fortran::parser::Variable>(&unit->u))
      return hasNonDefaultCharKind(*var);
  return false;
}
template <>
constexpr bool isDataTransferInternalNotDefaultKind<Fortran::parser::PrintStmt>(
    const Fortran::parser::PrintStmt &) {
  return false;
}

template <typename A>
static bool isDataTransferAsynchronous(const A &stmt) {
  if (auto *asynch =
          getIOControl<Fortran::parser::IoControlSpec::Asynchronous>(stmt)) {
    // FIXME: should contain a string of YES or NO
    llvm_unreachable("asynchronous transfers not implemented in runtime");
  }
  return false;
}
template <>
constexpr bool isDataTransferAsynchronous<Fortran::parser::PrintStmt>(
    const Fortran::parser::PrintStmt &) {
  return false;
}

template <typename A>
static bool isDataTransferNamelist(const A &stmt) {
  if (stmt.format)
    return formatIsActuallyNamelist(*stmt.format);
  return hasIOControl<Fortran::parser::Name>(stmt);
}
template <>
constexpr bool isDataTransferNamelist<Fortran::parser::PrintStmt>(
    const Fortran::parser::PrintStmt &) {
  return false;
}

static std::tuple<mlir::Value, mlir::Value, mlir::Value>
genFormat(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
          const Fortran::parser::Format &format, mlir::Type ty0, mlir::Type ty1,
          Fortran::lower::pft::LabelEvalMap &labelMap) {
  if (auto *e = std::get_if<Fortran::parser::Expr>(&format.u))
    return lowerStringLit(converter, loc, *e, ty0, ty1);
  if (auto *lab = std::get_if<Fortran::parser::Label>(&format.u)) {
    auto iter = labelMap.find(*lab);
    assert(iter != labelMap.end() && "FORMAT not found in PROCEDURE");
    return lowerSourceTextAsStringLit(
        converter, loc, toStringRef(iter->second->position), ty0, ty1);
  }
  llvm_unreachable("* case should be handled elsewhere as transfer list");
}

template <typename A>
std::tuple<mlir::Value, mlir::Value, mlir::Value>
getFormat(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
          const A &stmt, mlir::Type ty0, mlir::Type ty1,
          Fortran::lower::pft::LabelEvalMap &labelMap) {
  if (stmt.format && !formatIsActuallyNamelist(*stmt.format))
    return genFormat(converter, loc, *stmt.format, ty0, ty1, labelMap);
  return genFormat(converter, loc, *getIOControl<Fortran::parser::Format>(stmt),
                   ty0, ty1, labelMap);
}
template <>
std::tuple<mlir::Value, mlir::Value, mlir::Value>
getFormat<Fortran::parser::PrintStmt>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::parser::PrintStmt &stmt, mlir::Type ty0, mlir::Type ty1,
    Fortran::lower::pft::LabelEvalMap &labelMap) {
  return genFormat(converter, loc, std::get<Fortran::parser::Format>(stmt.t),
                   ty0, ty1, labelMap);
}

static std::tuple<mlir::Value, mlir::Value, mlir::Value>
genBuffer(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
          const Fortran::parser::IoUnit &iounit, mlir::Type ty0,
          mlir::Type ty1) {
  [[maybe_unused]] auto &var = std::get<Fortran::parser::Variable>(iounit.u);
  TODO();
}
template <typename A>
std::tuple<mlir::Value, mlir::Value, mlir::Value>
getBuffer(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
          const A &stmt, mlir::Type ty0, mlir::Type ty1) {
  if (stmt.iounit)
    return genBuffer(converter, loc, *stmt.iounit, ty0, ty1);
  return genBuffer(converter, loc, *getIOControl<Fortran::parser::IoUnit>(stmt),
                   ty0, ty1);
}

template <typename A>
mlir::Value getDescriptor(Fortran::lower::AbstractConverter &converter,
                          mlir::Location loc, const A &stmt,
                          mlir::Type toType) {
  TODO();
}

static mlir::Value genIOUnit(Fortran::lower::AbstractConverter &converter,
                             mlir::Location loc,
                             const Fortran::parser::IoUnit &iounit,
                             mlir::Type ty) {
  auto &builder = converter.getFirOpBuilder();
  if (auto *e = std::get_if<Fortran::parser::FileUnitNumber>(&iounit.u)) {
    auto ex = converter.genExprValue(Fortran::semantics::GetExpr(*e), loc);
    return builder.createConvert(loc, ty, ex);
  }
  return builder.create<mlir::ConstantOp>(
      loc, builder.getIntegerAttr(ty, Fortran::runtime::io::DefaultUnit));
}

template <typename A>
mlir::Value getIOUnit(Fortran::lower::AbstractConverter &converter,
                      mlir::Location loc, const A &stmt, mlir::Type ty) {
  if (stmt.iounit)
    return genIOUnit(converter, loc, *stmt.iounit, ty);
  return genIOUnit(converter, loc, *getIOControl<Fortran::parser::IoUnit>(stmt),
                   ty);
}

//===----------------------------------------------------------------------===//
// Generators for each I/O statement type.
//===----------------------------------------------------------------------===//

template <typename K, typename S>
static mlir::Value genBasicIOStmt(Fortran::lower::AbstractConverter &converter,
                                  const S &stmt) {
  auto &builder = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  auto beginFunc = getIORuntimeFunc<K>(builder);
  mlir::FunctionType beginFuncTy = beginFunc.getType();
  auto unit = converter.genExprValue(
      getExpr<Fortran::parser::FileUnitNumber>(stmt), loc);
  auto un = builder.createConvert(loc, beginFuncTy.getInput(0), unit);
  auto file = getDefaultFilename(builder, loc, beginFuncTy.getInput(1));
  auto line = getDefaultLineNo(builder, loc, beginFuncTy.getInput(2));
  llvm::SmallVector<mlir::Value, 4> args{un, file, line};
  auto cookie = builder.create<mlir::CallOp>(loc, beginFunc, args).getResult(0);
  ErrorHandling eh{};
  threadErrors(converter, loc, cookie, stmt.v, eh);
  mlir::Value ok{};
  auto insertPt = threadSpecs(converter, loc, cookie, stmt.v, ok);
  if (insertPt.isSet())
    builder.restoreInsertionPoint(insertPt);
  return genEndIO(converter, converter.getCurrentLocation(), cookie, eh);
}

mlir::Value Fortran::lower::genBackspaceStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::BackspaceStmt &stmt) {
  return genBasicIOStmt<mkIOKey(BeginBackspace)>(converter, stmt);
}

mlir::Value Fortran::lower::genEndfileStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::EndfileStmt &stmt) {
  return genBasicIOStmt<mkIOKey(BeginEndfile)>(converter, stmt);
}

mlir::Value
Fortran::lower::genFlushStatement(Fortran::lower::AbstractConverter &converter,
                                  const Fortran::parser::FlushStmt &stmt) {
  return genBasicIOStmt<mkIOKey(BeginFlush)>(converter, stmt);
}

mlir::Value
Fortran::lower::genRewindStatement(Fortran::lower::AbstractConverter &converter,
                                   const Fortran::parser::RewindStmt &stmt) {
  return genBasicIOStmt<mkIOKey(BeginRewind)>(converter, stmt);
}

mlir::Value
Fortran::lower::genOpenStatement(Fortran::lower::AbstractConverter &converter,
                                 const Fortran::parser::OpenStmt &stmt) {
  auto &builder = converter.getFirOpBuilder();
  mlir::FuncOp beginFunc;
  llvm::SmallVector<mlir::Value, 4> beginArgs;
  auto loc = converter.getCurrentLocation();
  if (hasMem<Fortran::parser::FileUnitNumber>(stmt)) {
    beginFunc = getIORuntimeFunc<mkIOKey(BeginOpenUnit)>(builder);
    mlir::FunctionType beginFuncTy = beginFunc.getType();
    auto unit = converter.genExprValue(
        getExpr<Fortran::parser::FileUnitNumber>(stmt), loc);
    beginArgs.push_back(
        builder.createConvert(loc, beginFuncTy.getInput(0), unit));
    beginArgs.push_back(
        getDefaultFilename(builder, loc, beginFuncTy.getInput(1)));
    beginArgs.push_back(
        getDefaultLineNo(builder, loc, beginFuncTy.getInput(2)));
  } else {
    assert(hasMem<Fortran::parser::ConnectSpec::Newunit>(stmt));
    beginFunc = getIORuntimeFunc<mkIOKey(BeginOpenNewUnit)>(builder);
    mlir::FunctionType beginFuncTy = beginFunc.getType();
    beginArgs.push_back(
        getDefaultFilename(builder, loc, beginFuncTy.getInput(0)));
    beginArgs.push_back(
        getDefaultLineNo(builder, loc, beginFuncTy.getInput(1)));
  }
  auto cookie =
      builder.create<mlir::CallOp>(loc, beginFunc, beginArgs).getResult(0);
  ErrorHandling eh{};
  threadErrors(converter, loc, cookie, stmt.v, eh);
  mlir::Value ok{};
  auto insertPt = threadSpecs(converter, loc, cookie, stmt.v, ok);
  if (insertPt.isSet())
    builder.restoreInsertionPoint(insertPt);
  return genEndIO(converter, loc, cookie, eh);
}

mlir::Value
Fortran::lower::genCloseStatement(Fortran::lower::AbstractConverter &converter,
                                  const Fortran::parser::CloseStmt &stmt) {
  return genBasicIOStmt<mkIOKey(BeginClose)>(converter, stmt);
}

mlir::Value
Fortran::lower::genWaitStatement(Fortran::lower::AbstractConverter &converter,
                                 const Fortran::parser::WaitStmt &stmt) {
  auto &builder = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  bool hasId = hasMem<Fortran::parser::IdExpr>(stmt);
  mlir::FuncOp beginFunc =
      hasId ? getIORuntimeFunc<mkIOKey(BeginWait)>(builder)
            : getIORuntimeFunc<mkIOKey(BeginWaitAll)>(builder);
  mlir::FunctionType beginFuncTy = beginFunc.getType();
  auto unit = converter.genExprValue(
      getExpr<Fortran::parser::FileUnitNumber>(stmt), loc);
  auto un = builder.createConvert(loc, beginFuncTy.getInput(0), unit);
  llvm::SmallVector<mlir::Value, 4> args{un};
  if (hasId) {
    auto id =
        converter.genExprValue(getExpr<Fortran::parser::IdExpr>(stmt), loc);
    args.push_back(builder.createConvert(loc, beginFuncTy.getInput(1), id));
  }
  auto cookie = builder.create<mlir::CallOp>(loc, beginFunc, args).getResult(0);
  ErrorHandling eh{};
  threadErrors(converter, loc, cookie, stmt.v, eh);
  return genEndIO(converter, converter.getCurrentLocation(), cookie, eh);
}

//===----------------------------------------------------------------------===//
// Data transfer statements.
//
// There are several dimensions to the API with regard to data transfer
// statements that need to be considered.
//
//   - input (READ) vs. output (WRITE, PRINT)
//   - formatted vs. list vs. unformatted
//   - synchronous vs. asynchronous
//   - namelist vs. list
//   - external vs. internal + default KIND vs. internal + other KIND
//===----------------------------------------------------------------------===//

// Determine the correct BeginXyz{In|Out}put api to invoke.
template <bool isInput>
mlir::FuncOp getBeginDataTransfer(FirOpBuilder &builder, bool isFormatted,
                                  bool isList, bool isIntern,
                                  bool isOtherIntern, bool isAsynch,
                                  bool isNml) {
  if constexpr (isInput) {
    if (isAsynch)
      return getIORuntimeFunc<mkIOKey(BeginAsynchronousInput)>(builder);
    if (isFormatted) {
      if (isIntern) {
        if (isNml)
          return getIORuntimeFunc<mkIOKey(BeginInternalNamelistInput)>(builder);
        if (isOtherIntern) {
          if (isList)
            return getIORuntimeFunc<mkIOKey(BeginInternalArrayListInput)>(
                builder);
          return getIORuntimeFunc<mkIOKey(BeginInternalArrayFormattedInput)>(
              builder);
        }
        if (isList)
          return getIORuntimeFunc<mkIOKey(BeginInternalListInput)>(builder);
        return getIORuntimeFunc<mkIOKey(BeginInternalFormattedInput)>(builder);
      }
      if (isNml)
        return getIORuntimeFunc<mkIOKey(BeginExternalNamelistInput)>(builder);
      if (isList)
        return getIORuntimeFunc<mkIOKey(BeginExternalListInput)>(builder);
      return getIORuntimeFunc<mkIOKey(BeginExternalFormattedInput)>(builder);
    }
    return getIORuntimeFunc<mkIOKey(BeginUnformattedInput)>(builder);
  } else {
    if (isAsynch)
      return getIORuntimeFunc<mkIOKey(BeginAsynchronousOutput)>(builder);
    if (isFormatted) {
      if (isIntern) {
        if (isNml)
          return getIORuntimeFunc<mkIOKey(BeginInternalNamelistOutput)>(
              builder);
        if (isOtherIntern) {
          if (isList)
            return getIORuntimeFunc<mkIOKey(BeginInternalArrayListOutput)>(
                builder);
          return getIORuntimeFunc<mkIOKey(BeginInternalArrayFormattedOutput)>(
              builder);
        }
        if (isList)
          return getIORuntimeFunc<mkIOKey(BeginInternalListOutput)>(builder);
        return getIORuntimeFunc<mkIOKey(BeginInternalFormattedOutput)>(builder);
      }
      if (isNml)
        return getIORuntimeFunc<mkIOKey(BeginExternalNamelistOutput)>(builder);
      if (isList)
        return getIORuntimeFunc<mkIOKey(BeginExternalListOutput)>(builder);
      return getIORuntimeFunc<mkIOKey(BeginExternalFormattedOutput)>(builder);
    }
    return getIORuntimeFunc<mkIOKey(BeginUnformattedOutput)>(builder);
  }
}

template <bool hasIOCtrl, typename A>
void populateArguments(llvm::SmallVector<mlir::Value, 8> &ioArgs,
                       Fortran::lower::AbstractConverter &converter,
                       mlir::Location loc, const A &stmt,
                       mlir::FunctionType ioFuncTy, bool isFormatted,
                       bool isList, bool isIntern, bool isOtherIntern,
                       bool isAsynch, bool isNml,
                       Fortran::lower::pft::LabelEvalMap &labelMap) {
  auto &builder = converter.getFirOpBuilder();
  if constexpr (hasIOCtrl) {
    // READ/WRITE cases have a wide variety of argument permutations
    if (isAsynch || !isFormatted) {
      // unit (always first), ...
      ioArgs.push_back(
          getIOUnit(converter, loc, stmt, ioFuncTy.getInput(ioArgs.size())));
      if (isAsynch) {
        // unknown-thingy, [buff, LEN]
        llvm_unreachable("not implemented");
      }
      return;
    }
    assert(isFormatted && "formatted data transfer");
    if (!isIntern) {
      if (isNml) {
        // namelist group, ...
        llvm_unreachable("not implemented");
      } else if (!isList) {
        // | [format, LEN], ...
        auto pair =
            getFormat(converter, loc, stmt, ioFuncTy.getInput(ioArgs.size()),
                      ioFuncTy.getInput(ioArgs.size() + 1), labelMap);
        ioArgs.push_back(std::get<0>(pair));
        ioArgs.push_back(std::get<1>(pair));
      }
      // unit (always last)
      ioArgs.push_back(
          getIOUnit(converter, loc, stmt, ioFuncTy.getInput(ioArgs.size())));
      return;
    }
    assert(isIntern && "internal data transfer");
    if (isNml || isOtherIntern) {
      // descriptor, ...
      ioArgs.push_back(getDescriptor(converter, loc, stmt,
                                     ioFuncTy.getInput(ioArgs.size())));
      if (isNml) {
        // namelist group, ...
        llvm_unreachable("not implemented");
      } else if (isOtherIntern && !isList) {
        // | [format, LEN], ...
        auto pair =
            getFormat(converter, loc, stmt, ioFuncTy.getInput(ioArgs.size()),
                      ioFuncTy.getInput(ioArgs.size() + 1), labelMap);
        ioArgs.push_back(std::get<0>(pair));
        ioArgs.push_back(std::get<1>(pair));
      }
    } else {
      // | [buff, LEN], ...
      auto pair =
          getBuffer(converter, loc, stmt, ioFuncTy.getInput(ioArgs.size()),
                    ioFuncTy.getInput(ioArgs.size() + 1));
      ioArgs.push_back(std::get<0>(pair));
      ioArgs.push_back(std::get<1>(pair));
      if (!isList) {
        // [format, LEN], ...
        auto pair =
            getFormat(converter, loc, stmt, ioFuncTy.getInput(ioArgs.size()),
                      ioFuncTy.getInput(ioArgs.size() + 1), labelMap);
        ioArgs.push_back(std::get<0>(pair));
        ioArgs.push_back(std::get<1>(pair));
      }
    }
    // [scratch, LEN] (always last)
    ioArgs.push_back(
        getDefaultScratch(builder, loc, ioFuncTy.getInput(ioArgs.size())));
    ioArgs.push_back(
        getDefaultScratchLen(builder, loc, ioFuncTy.getInput(ioArgs.size())));
  } else {
    if (!isList) {
      // [format, LEN], ...
      auto pair =
          getFormat(converter, loc, stmt, ioFuncTy.getInput(ioArgs.size()),
                    ioFuncTy.getInput(ioArgs.size() + 1), labelMap);
      ioArgs.push_back(std::get<0>(pair));
      ioArgs.push_back(std::get<1>(pair));
    }
    // unit (always last)
    ioArgs.push_back(builder.create<mlir::ConstantOp>(
        loc, builder.getIntegerAttr(ioFuncTy.getInput(ioArgs.size()),
                                    Fortran::runtime::io::DefaultUnit)));
  }
}

template <typename A>
const std::list<Fortran::parser::OutputItem> &getOutputItems(const A &stmt) {
  if constexpr (std::is_same_v<A, Fortran::parser::PrintStmt>) {
    return std::get<std::list<Fortran::parser::OutputItem>>(stmt.t);
  } else {
    return stmt.items;
  }
}

template <bool isInput, bool hasIOCtrl = true, typename A>
static mlir::Value
genDataTransfer(Fortran::lower::AbstractConverter &converter, const A &stmt,
                Fortran::lower::pft::LabelEvalMap &labelMap) {
  auto &builder = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  mlir::Value ok{};
  const bool isFormatted = isDataTransferFormatted(stmt);
  const bool isList = isFormatted ? isDataTransferList(stmt) : false;
  const bool isIntern = isDataTransferInternal(stmt);
  const bool isOtherIntern =
      isIntern ? isDataTransferInternalNotDefaultKind(stmt) : false;
  const bool isAsynch = isDataTransferAsynchronous(stmt);
  const bool isNml = isDataTransferNamelist(stmt);

  // Determine the correct IO API call to make.
  mlir::FuncOp ioFunc = getBeginDataTransfer<isInput>(
      builder, isFormatted, isList, isIntern, isOtherIntern, isAsynch, isNml);
  mlir::FunctionType ioFuncTy = ioFunc.getType();

  // append BeginXyz arguments
  llvm::SmallVector<mlir::Value, 8> ioArgs;
  populateArguments<hasIOCtrl>(ioArgs, converter, loc, stmt, ioFuncTy,
                               isFormatted, isList, isIntern, isOtherIntern,
                               isAsynch, isNml, labelMap);
  // filename and line are universal and always last
  ioArgs.push_back(
      getDefaultFilename(builder, loc, ioFuncTy.getInput(ioArgs.size())));
  ioArgs.push_back(
      getDefaultLineNo(builder, loc, ioFuncTy.getInput(ioArgs.size())));

  // args are done; call the BeginXyz function
  mlir::Value cookie =
      builder.create<mlir::CallOp>(loc, ioFunc, ioArgs).getResult(0);

  ErrorHandling eh{};
  mlir::OpBuilder::InsertPoint insertPt;
  if constexpr (hasIOCtrl) {
    threadErrors(converter, loc, cookie, stmt.controls, eh);
    insertPt = threadSpecs(converter, loc, cookie, stmt.controls, ok);
  }
  // process the arguments
  llvm::SmallVector<mlir::Value, 32> args;
  if constexpr (isInput) {
    for (auto &item : stmt.items) {
      if (auto *pe = std::get_if<Fortran::parser::Variable>(&item.u)) {
        auto loc = converter.genLocation(pe->GetSource());
        args.push_back(
            converter.genExprAddr(Fortran::semantics::GetExpr(*pe), loc));
      } else {
        TODO(); // TODO implied do
      }
    }
    for (mlir::Value arg : args)
      genInputRuntimeFunc(builder, loc, arg.getType(), cookie, arg, insertPt,
                          ok);
  } else {
    for (auto &item : getOutputItems(stmt)) {
      if (auto *pe = std::get_if<Fortran::parser::Expr>(&item.u)) {
        auto loc = converter.genLocation(pe->source);
        args.push_back(
            converter.genExprValue(Fortran::semantics::GetExpr(*pe), loc));
      } else {
        TODO(); // TODO implied do
      }
    }
    for (mlir::Value arg : args) {
      // need some special handling for COMPLEX and CHARACTER
      auto operands = splitArguments(builder, loc, arg);
      genOutputRuntimeFunc(builder, loc, arg.getType(), cookie, operands,
                           insertPt, ok);
    }
  }

  if (insertPt.isSet())
    builder.restoreInsertionPoint(insertPt);
  return genEndIO(converter, loc, cookie, eh);
}

void Fortran::lower::genPrintStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::PrintStmt &stmt,
    Fortran::lower::pft::LabelEvalMap &labelMap) {
  // PRINT does not take an io-control-spec. It only has a format specifier, so
  // it is a simplified case of WRITE.
  genDataTransfer</*isInput=*/false, /*ioCtrl=*/false>(converter, stmt,
                                                       labelMap);
}

mlir::Value
Fortran::lower::genWriteStatement(Fortran::lower::AbstractConverter &converter,
                                  const Fortran::parser::WriteStmt &stmt,
                                  Fortran::lower::pft::LabelEvalMap &labelMap) {
  return genDataTransfer</*isInput=*/false>(converter, stmt, labelMap);
}

mlir::Value
Fortran::lower::genReadStatement(Fortran::lower::AbstractConverter &converter,
                                 const Fortran::parser::ReadStmt &stmt,
                                 Fortran::lower::pft::LabelEvalMap &labelMap) {
  return genDataTransfer</*isInput=*/true>(converter, stmt, labelMap);
}

mlir::Value Fortran::lower::genInquireStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::InquireStmt &) {
  auto &builder = converter.getFirOpBuilder();
  mlir::FuncOp beginFunc;
  // if (...
  beginFunc = getIORuntimeFunc<mkIOKey(BeginInquireUnit)>(builder);
  // else if (...
  beginFunc = getIORuntimeFunc<mkIOKey(BeginInquireFile)>(builder);
  // else
  beginFunc = getIORuntimeFunc<mkIOKey(BeginInquireIoLength)>(builder);
  TODO();
  return {};
}
