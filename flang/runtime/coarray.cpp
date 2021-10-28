//===-- runtime/coarray.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/coarray.h"
#include <cstdio>

extern "C" {

int RTNAME(NumImages)(int team,
    const char *sourceFile, int sourceLine) {
  std::fputs("Fortran NUM_IMAGES: single image value is 1", stderr);
  // return value for single image (default coarray runtime)
  return 1;
}

}
