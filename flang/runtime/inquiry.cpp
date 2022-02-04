//===-- runtime/inquiry.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements the inquiry intrinsic functions of Fortran 2018 that
// inquire about shape information of arrays -- LBOUND and SIZE.

#include "flang/Runtime/inquiry.h"
#include "copy.h"
#include "terminator.h"
#include "tools.h"
#include "flang/Runtime/descriptor.h"
#include <algorithm>
#include <stdio.h>

namespace Fortran::runtime {

extern "C" {
std::int64_t RTNAME(LboundDim)(
    const Descriptor &array, int dim, const char *sourceFile, int line) {
  if (dim < 1 || dim > array.rank()) {
    Terminator terminator{sourceFile, line};
    terminator.Crash("SIZE: bad DIM=%d", dim);
  }
  const Dimension &dimension{array.GetDimension(dim - 1)};
  return static_cast<std::int64_t>(dimension.LowerBound());
}

std::int64_t RTNAME(Size)(
    const Descriptor &array, const char *sourceFile, int line) {
  SubscriptValue result{1};
  for (int i = 0; i < array.rank(); ++i) {
    const Dimension &dimension{array.GetDimension(i)};
    result *= dimension.Extent();
  }
  return result;
}

std::int64_t RTNAME(SizeDim)(
    const Descriptor &array, int dim, const char *sourceFile, int line) {
  if (dim < 1 || dim > array.rank()) {
    Terminator terminator{sourceFile, line};
    terminator.Crash("SIZE: bad DIM=%d", dim);
  }
  const Dimension &dimension{array.GetDimension(dim - 1)};
  return static_cast<std::int64_t>(dimension.Extent());
}

} // extern "C"
} // namespace Fortran::runtime
