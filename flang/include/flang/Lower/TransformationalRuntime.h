//===-- Lower/TransformationalRuntime.h -- lower transformational intrinsics==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_TRANSFORMATIONALRUNTIME_H
#define FORTRAN_LOWER_TRANSFORMATIONALRUNTIME_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace fir {
class ExtendedValue;
}

namespace Fortran::lower {
class FirOpBuilder;

void genReshape(FirOpBuilder &builder, mlir::Location loc,
                mlir::Value resultBox, mlir::Value sourceBox,
                mlir::Value shapeBox, mlir::Value padBox, mlir::Value orderBox);

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_TRANSFORMATIONALRUNTIME_H
