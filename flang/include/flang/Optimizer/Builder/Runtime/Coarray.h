//===-- Coarray.h -- generate calls to coarray runtime API ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COARRAY_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COARRAY_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace fir {
class ExtendedValue;
class FirOpBuilder;
} // namespace fir

namespace fir::runtime {

/// Generate call to num_images runtime.
///   \p teamBox must be a fir.box describing the team_number argument.
mlir::Value genNumImages(fir::FirOpBuilder &builder, mlir::Location loc,
                         mlir::Value teamBox);

} // namespace fir::runtime

#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COARRAY_H
