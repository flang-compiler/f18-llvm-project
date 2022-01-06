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
class CharBoxValue;
}

namespace llvm {
    template<typename T>
    class Optional;
}

namespace fir::runtime {

/// Generate call to COMMAND_ARGUMENT_COUNT intrinsic runtime routine.
mlir::Value genCommandArgumentCount(fir::FirOpBuilder &, mlir::Location);

/// Generate call to GET_COMMAND_ARGUMENT intrinsic runtime routine.
void genGetCommandArgument(fir::FirOpBuilder &, mlir::Location, mlir::Value number,
                           llvm::Optional<fir::CharBoxValue> valueBox, mlir::Value length,
                           mlir::Value status, llvm::Optional<fir::CharBoxValue> errmsgBox);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COMMAND_H
