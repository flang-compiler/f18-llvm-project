//===-- Lower/Support/Verifier.h -- verify pass for lowering ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://github.com/flang-compiler/f18-llvm-project/blob/fir-dev/flang/docs/LLVMCodingStyle.md
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_SUPPORT_VERIFIER_H
#define FORTRAN_LOWER_SUPPORT_VERIFIER_H

#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

namespace Fortran::lower {

/// A verification pass to verify the output from the bridge. This provides a
/// little bit of glue to run a verifier pass directly.
class VerifierPass
    : public mlir::PassWrapper<VerifierPass, mlir::OperationPass<>> {
  void runOnOperation() override final {
    if (mlir::failed(mlir::verify(getOperation())))
      signalPassFailure();
    markAllAnalysesPreserved();
  }
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_SUPPORT_VERIFIER_H
