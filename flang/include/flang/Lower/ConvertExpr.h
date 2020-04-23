//===-- Lower/ConvertExpr.h -- lowering of expressions ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_CONVERT_EXPR_H
#define FORTRAN_LOWER_CONVERT_EXPR_H

#include "Intrinsics.h"

/// [Coding style](https://llvm.org/docs/CodingStandards.html)

namespace mlir {
class Location;
class OpBuilder;
class Type;
class Value;
} // namespace mlir

namespace fir {
class AllocaExpr;
} // namespace fir

namespace Fortran {
namespace common {
class IntrinsicTypeDefaultKinds;
} // namespace common

namespace evaluate {
template <typename>
class Expr;
struct SomeType;
} // namespace evaluate

namespace semantics {
class Symbol;
} // namespace semantics

namespace lower {

class AbstractConverter;
class FirOpBuilder;
class SymMap;

/// Create an expression. Lower \p expr to the FIR dialect of MLIR as a
/// value result.
mlir::Value createSomeExpression(mlir::Location loc,
                                 AbstractConverter &converter,
                                 const evaluate::Expr<evaluate::SomeType> &expr,
                                 SymMap &symMap,
                                 const IntrinsicLibrary &intrinsics);

/// Create an expression. Like createSomeExpression, but the result is an
/// I1 logical value suitable for use as a conditional value.
mlir::Value
createI1LogicalExpression(mlir::Location loc, AbstractConverter &converter,
                          const evaluate::Expr<evaluate::SomeType> &expr,
                          SymMap &symMap, const IntrinsicLibrary &intrinsics);

/// Create an expression to call a subroutine with an alternate return value.
/// The return value is an integer index that the caller can use to select
/// alternate call successor code.
mlir::Value
createAltReturnCallExpression(mlir::Location loc, AbstractConverter &converter,
                              const evaluate::Expr<evaluate::SomeType> &expr,
                              SymMap &symMap,
                              const IntrinsicLibrary &intrinsics);

/// Create an address. Lower \p expr to the FIR dialect of MLIR. The expression
/// must be an entity. The address of the entity is returned.
mlir::Value createSomeAddress(mlir::Location loc, AbstractConverter &converter,
                              const evaluate::Expr<evaluate::SomeType> &expr,
                              SymMap &symMap,
                              const IntrinsicLibrary &intrinsics);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_CONVERT_EXPR_H
