//===-- Coarray.cpp ---------------------------------------------*- C++ -*-===//
// Generate coarray intrinsic runtime API calls.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Coarray.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/coarray.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace Fortran::runtime;

/// Generate call to num_images intrinsic function
mlir::Value fir::runtime::genNumImages(fir::FirOpBuilder &builder,
                                       mlir::Location loc,
                                       mlir::Value teamBox) {
  mlir::FuncOp func = fir::runtime::getRuntimeFunc<mkRTKey(NumImages)>(loc, builder);
  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine = fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
  auto args = fir::runtime::createArguments(builder, loc, fTy, teamBox,
                                            sourceFile, sourceLine);
  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}
