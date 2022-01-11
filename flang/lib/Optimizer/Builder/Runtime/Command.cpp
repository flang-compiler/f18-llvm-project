//===-- Command.cpp -- generate command line runtime API calls ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Command.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/command.h"
#include "llvm/ADT/Optional.h"
using namespace Fortran::runtime;

mlir::Value fir::runtime::genCommandArgumentCount(fir::FirOpBuilder &builder,
                                                  mlir::Location loc) {
  auto argumentCountFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(ArgumentCount)>(loc, builder);
  return builder.create<fir::CallOp>(loc, argumentCountFunc).getResult(0);
}

void fir::runtime::genGetCommandArgument(fir::FirOpBuilder &builder,
                                         mlir::Location loc, mlir::Value number,
                                         mlir::Value value, mlir::Value length,
                                         mlir::Value status,
                                         mlir::Value errmsg) {
  auto argumentValueFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(ArgumentValue)>(loc, builder);
  auto argumentLengthFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(ArgumentLength)>(loc, builder);

  auto isPresent = [&](mlir::Value val) -> bool {
    auto definingOp = val.getDefiningOp();
    if (auto cst = mlir::dyn_cast<fir::AbsentOp>(definingOp))
      return false;
    return true;
  };

  if (isPresent(value) || status || isPresent(errmsg)) {
    llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
        builder, loc, argumentValueFunc.getType(), number, value, errmsg);
    mlir::Value result =
        builder.create<fir::CallOp>(loc, argumentValueFunc, args).getResult(0);

    if (status) {
      const mlir::Value statusLoaded = builder.create<fir::LoadOp>(loc, status);
      mlir::Value resultCast =
          builder.createConvert(loc, statusLoaded.getType(), result);
      builder.create<fir::StoreOp>(loc, resultCast, status);
    }
  }

  if (length) {
    llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
        builder, loc, argumentLengthFunc.getType(), number);
    mlir::Value result =
        builder.create<fir::CallOp>(loc, argumentLengthFunc, args).getResult(0);
    const mlir::Value valueLoaded = builder.create<fir::LoadOp>(loc, length);
    mlir::Value resultCast =
        builder.createConvert(loc, valueLoaded.getType(), result);
    builder.create<fir::StoreOp>(loc, resultCast, length);
  }
}
