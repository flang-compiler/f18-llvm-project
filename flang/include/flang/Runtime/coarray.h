//===-- include/flang/Runtime/coarray.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_COARRAY_H_
#define FORTRAN_RUNTIME_COARRAY_H_

// Defines API between compiled code and the coarray
// support functions in the runtime library.

#include "flang/Runtime/entry-names.h"

namespace Fortran::runtime {

extern "C" {
// 16.9.145 NUM_IMAGES
// The default value team_number=0 is not a valid team number and thus chosen
// when NUM_IMAGES is called with no actual arguments, see:
//   1. 9.6 paragraph 3: TEAM_NUMBER shall be from a FORM TEAM statement or
//      the initial team.
//   2. 16.9.189 TEAM_NUMBER paragraph 5: initial team shall have the value -1.
//   3. 11.6.9 FORM TEAM statement: TEAM_NUMBER shall be greater than 0.
int RTNAME(NumImages)(
    int team_number = 0, const char *sourceFile = nullptr, int sourceLine = 0);
}
} // namespace Fortran::runtime

#endif // FORTRAN_RUNTIME_COARRAY_H_
