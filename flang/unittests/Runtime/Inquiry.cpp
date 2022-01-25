//===-- flang/unittests/RuntimeGTest/Inquiry.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/inquiry.h"
#include "gtest/gtest.h"
#include "tools.h"
#include "flang/Runtime/type-code.h"

using namespace Fortran::runtime;
using Fortran::common::TypeCategory;

TEST(Inquiry, Lbound) {
  // ARRAY  1 3 5
  //        2 4 6
  auto array{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{2, 3}, std::vector<std::int32_t>{1, 2, 3, 4, 5, 6})};
  array->GetDimension(0).SetLowerBound(0);
  array->GetDimension(1).SetLowerBound(-1);
  StaticDescriptor<2, true> statDesc;

  EXPECT_EQ(RTNAME(LboundDim)(*array, 1, __FILE__, __LINE__), std::int64_t{0});
  EXPECT_EQ(RTNAME(LboundDim)(*array, 2, __FILE__, __LINE__), std::int64_t{-1});
}

TEST(Inquiry, Size) {
  // ARRAY  1 3 5
  //        2 4 6
  auto array{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{2, 3}, std::vector<std::int32_t>{1, 2, 3, 4, 5, 6})};
  array->GetDimension(0).SetLowerBound(0); // shouldn't matter
  array->GetDimension(1).SetLowerBound(-1);
  StaticDescriptor<2, true> statDesc;

  EXPECT_EQ(RTNAME(SizeDim)(*array, 1, __FILE__, __LINE__), std::int64_t{2});
  EXPECT_EQ(RTNAME(SizeDim)(*array, 2, __FILE__, __LINE__), std::int64_t{3});
  EXPECT_EQ(RTNAME(Size)(*array, __FILE__, __LINE__), std::int64_t{6});
}
