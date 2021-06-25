//===-- TransformationalRuntime.cpp -- runtime for transformational intrinsics//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/TransformationalRuntime.h"
#include "../../runtime/transformational.h"
#include "RTBuilder.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/CharacterExpr.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/Support/BoxValue.h"
#include "flang/Lower/Todo.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace Fortran::runtime;

/// Generate call to Reshape intrinsic runtime routine.
void Fortran::lower::genReshape(Fortran::lower::FirOpBuilder &builder,
                                mlir::Location loc, mlir::Value resultBox,
                                mlir::Value sourceBox, mlir::Value shapeBox,
                                mlir::Value padBox, mlir::Value orderBox) {
  auto func = Fortran::lower::getRuntimeFunc<mkRTKey(Reshape)>(loc, builder);
  auto fTy = func.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(6));
  auto args = Fortran::lower::createArguments(builder, loc, fTy, resultBox,
                                              sourceBox, shapeBox, padBox,
                                              orderBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}
