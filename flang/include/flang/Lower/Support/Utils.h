//===-- Lower/Support/Utils.h -- utilities ----------------------*- C++ -*-===//
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

#ifndef FORTRAN_LOWER_SUPPORT_UTILS_H
#define FORTRAN_LOWER_SUPPORT_UTILS_H

#include "flang/Common/indirection.h"
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Parser/char-block.h"
#include "flang/Semantics/tools.h"
#include "llvm/ADT/StringRef.h"

//===----------------------------------------------------------------------===//
// Small inline helper functions to deal with repetitive, clumsy conversions.
//===----------------------------------------------------------------------===//

/// Convert an F18 CharBlock to an LLVM StringRef.
inline llvm::StringRef toStringRef(const Fortran::parser::CharBlock &cb) {
  return {cb.begin(), cb.size()};
}

/// Template helper to remove Fortran::common::Indirection wrappers.
template <typename A>
const A &removeIndirection(const A &a) {
  return a;
}
template <typename A>
const A &removeIndirection(const Fortran::common::Indirection<A> &a) {
  return a.value();
}

/// Clone subexpression and wrap it as a generic `Fortran::evaluate::Expr`.
template <typename A>
static Fortran::evaluate::Expr<Fortran::evaluate::SomeType>
toEvExpr(const A &x) {
  return Fortran::evaluate::AsGenericExpr(Fortran::common::Clone(x));
}

template <Fortran::common::TypeCategory FROM>
static Fortran::evaluate::Expr<Fortran::evaluate::SomeType> ignoreEvConvert(
    const Fortran::evaluate::Convert<
        Fortran::evaluate::Type<Fortran::common::TypeCategory::Integer, 8>,
        FROM> &x) {
  return toEvExpr(x.left());
}
template <typename A>
static Fortran::evaluate::Expr<Fortran::evaluate::SomeType>
ignoreEvConvert(const A &x) {
  return toEvExpr(x);
}
/// A vector subscript expression may be wrapped with a cast to INTEGER*8.
/// Get rid of it here so the vector can be loaded. Add it back when
/// generating the elemental evaluation (inside the loop nest).
inline Fortran::evaluate::Expr<Fortran::evaluate::SomeType>
ignoreEvConvert(const Fortran::evaluate::Expr<Fortran::evaluate::Type<
                    Fortran::common::TypeCategory::Integer, 8>> &x) {
  return std::visit([](const auto &v) { return ignoreEvConvert(v); }, x.u);
}

#endif // FORTRAN_LOWER_SUPPORT_UTILS_H
