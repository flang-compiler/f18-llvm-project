//===--- TargetOptions.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the flang::TargetOptions class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_TARGETOPTIONS_H
#define LLVM_FLANG_TARGETOPTIONS_H

namespace Fortran::frontend {

/// Options for controlling the target.
class TargetOptions {
public:
  /// The name of the target triple to compile for.
  std::string triple;
};

} // end namespace Fortran::frontend

#endif
