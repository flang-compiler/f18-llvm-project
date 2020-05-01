//===-- Lower/Support/BoxValue.h -- FIR box values --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LOWER_SUPPORT_BOXVALUE_H
#define LOWER_SUPPORT_BOXVALUE_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace fir {

//===----------------------------------------------------------------------===//
//
// Boxed values
//
// Define a set of containers to use internally to keep track of extended values
// associated with a Fortran subexpression. These associations are maintained
// during the construction of FIR.
//
//===----------------------------------------------------------------------===//

/// Most expressions of intrinsic type can be passed unboxed. Their properties are known statically.
using UnboxedValue = mlir::Value;

/// Abstract base class.
struct AbstractBox {
  AbstractBox() = delete;
  AbstractBox(mlir::Value addr) : baseAddr{addr} {}
  mlir::Value getAddr() const { return baseAddr; }
  
  mlir::Value baseAddr;
};

/// Expressions of CHARACTER type have an associated, possibly dynamic LEN value.
struct CharBoxValue : public AbstractBox {
  CharBoxValue(mlir::Value addr, mlir::Value len)
      : AbstractBox{addr}, len{len} {}

  mlir::Value getLen() const { return len; }
  
  mlir::Value len;
};

/// Abstract base class.
/// Expressions of type array have at minimum a shape. These expressions may have lbound attributes (dynamic values) that affect the interpretation of indexing expressions.
struct AbstractArrayBox {
  AbstractArrayBox() = default;
  AbstractArrayBox(llvm::ArrayRef<mlir::Value> extents,
                   llvm::ArrayRef<mlir::Value> lbounds)
      : extents{extents.begin(), extents.end()}, lbounds{lbounds.begin(),
                                                         lbounds.end()} {}

  // Every array has extents that describe its shape.
  const llvm::SmallVectorImpl<mlir::Value> &getExtends() const {
    return extents;
  }

  // An array expression may have user-defined lower bound values.
  // If this vector is empty, the default in all dimensions in `1`.
  const llvm::SmallVectorImpl<mlir::Value> &getLBounds() const {
    return lbounds;
  }

  bool lboundsAllOne() const { return lbounds.empty(); }

  llvm::SmallVector<mlir::Value, 4> extents;
  llvm::SmallVector<mlir::Value, 4> lbounds;
};

/// Expressions with rank > 0 have extents. They may also have lbounds that are
/// not 1.
struct ArrayBoxValue : public AbstractBox, public AbstractArrayBox {
  ArrayBoxValue(mlir::Value addr, llvm::ArrayRef<mlir::Value> extents,
                llvm::ArrayRef<mlir::Value> lbounds = {})
      : AbstractBox{addr}, AbstractArrayBox{extents, lbounds} {}
};

/// Expressions of type CHARACTER and with rank > 0.
struct CharArrayBoxValue : public CharBoxValue, public AbstractArrayBox {
  CharArrayBoxValue(mlir::Value addr, mlir::Value len,
                    llvm::ArrayRef<mlir::Value> extents,
                    llvm::ArrayRef<mlir::Value> lbounds = {})
      : CharBoxValue{addr, len}, AbstractArrayBox{extents, lbounds} {}
};

/// Expressions that are procedure POINTERs may need a set of references to
/// variables in the host scope.
struct ProcBoxValue : public AbstractBox {
  ProcBoxValue(mlir::Value addr, mlir::Value context)
      : AbstractBox{addr}, hostContext{context} {}

  mlir::Value hostContext;
};

/// In the generalized form, a boxed value can have a dynamic size, be an array
/// with dynamic extents and lbounds, and take dynamic type parameters.
struct BoxValue : public AbstractBox, public AbstractArrayBox {
  BoxValue(mlir::Value addr) : AbstractBox{addr}, AbstractArrayBox{} {}
  BoxValue(mlir::Value addr, mlir::Value len)
      : AbstractBox{addr}, AbstractArrayBox{}, len{len} {}
  BoxValue(mlir::Value addr, llvm::ArrayRef<mlir::Value> extents,
           llvm::ArrayRef<mlir::Value> lbounds = {})
      : AbstractBox{addr}, AbstractArrayBox{extents, lbounds} {}
  BoxValue(mlir::Value addr, mlir::Value len,
           llvm::ArrayRef<mlir::Value> params,
           llvm::ArrayRef<mlir::Value> extents,
           llvm::ArrayRef<mlir::Value> lbounds = {})
      : AbstractBox{addr}, AbstractArrayBox{extents, lbounds}, len{len},
        params{params.begin(), params.end()} {}

  mlir::Value len;
  llvm::SmallVector<mlir::Value, 2> params;
};

} // namespace fir

#endif // LOWER_SUPPORT_BOXVALUE_H
