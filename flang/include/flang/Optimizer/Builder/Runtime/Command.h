//===-- Command.cpp -- generate command line runtime API calls ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COMMAND_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COMMAND_H

namespace mlir {
class Value;
class Location;
} // namespace mlir

namespace fir {
class FirOpBuilder;
} // namespace fir

namespace fir::runtime {

/// Generate call to COMMAND_ARGUMENT_COUNT intrinsic runtime routine.
mlir::Value genCommandArgumentCount(fir::FirOpBuilder &, mlir::Location);

/// Generate a call to ArgumentValue runtime function which implements
/// the part of GET_COMMAND_ARGUMENT related to VALUE, ERRMSG, and STATUS.
/// \p value and \p errmsg must be fir.box that can be absent (but not null
/// mlir values). The status value is returned.
mlir::Value genArgumentValue(fir::FirOpBuilder &, mlir::Location,
                             mlir::Value number, mlir::Value value,
                             mlir::Value errmsg);

/// Generate a call to ArgumentLength runtime function which implements
/// the part of GET_COMMAND_ARGUMENT related to LENGTH.
/// It returns the length of the \p number command arguments.
mlir::Value genArgumentLength(fir::FirOpBuilder &, mlir::Location,
                              mlir::Value number);

/// Generate call to GET_ENVIRONMENT_VARIABLE intrinsic runtime routine.
/// Note that GET_ENVIRONMENT_ARGUMENT intrinsic is split between 2 functions in
/// implementation; EnvVariableValue and EnvVariableLength. So we handle each
/// seperately.
void genGetEnvironmentVariable(fir::FirOpBuilder &, mlir::Location,
                               mlir::Value number, mlir::Value value,
                               mlir::Value length, mlir::Value status,
                               mlir::Value trimName, mlir::Value errmsg);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COMMAND_H
