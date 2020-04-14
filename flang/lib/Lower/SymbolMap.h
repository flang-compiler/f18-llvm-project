//===-- SymbolMap.h -- lowering internal symbol map -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_SYMBOLMAP_H
#define FORTRAN_LOWER_SYMBOLMAP_H

#include "flang/Common/idioms.h"
#include "flang/Common/reference.h"
#include "flang/Semantics/symbol.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"

namespace Fortran::lower {

/// An index of ssa-values that together compose a variable referenced by a
/// Symbol. For example, the declaration
///
///   CHARACTER(LEN=i) :: c(j1,j2)
///
/// is a single variable `c`. This variable is a two-dimensional array of
/// CHARACTER. It has a starting address and three dynamic properties: the LEN
/// parameter `i` a runtime value describing the length of the CHARACTER, and
/// the `j1` and `j2` runtime values, which describe the shape of the array.
///
/// The lowering bridge needs to be able to record all four of these ssa-values
/// in the lookup table to be able to correctly lower Fortran to FIR.
struct SymIndex {
  // For lookups that fail, have a monostate
  using None = std::monostate;

  // Capture triple notation ssa-values
  using Triple = std::tuple<mlir::Value, mlir::Value, mlir::Value>;

  // Trivial intrinsic type
  struct Intrinsic {
    explicit Intrinsic(mlir::Value addr) : addr{addr} {}
    mlir::Value addr;
  };

  // Array variable that has a simple shape
  struct Shaped {
    explicit Shaped(mlir::Value addr, llvm::ArrayRef<mlir::Value> s)
        : addr{addr}, shape{s.begin(), s.end()} {}
    mlir::Value addr;
    std::vector<mlir::Value> shape;
  };

  // Array variable that uses triple notation
  struct FullDim {
    explicit FullDim(mlir::Value addr, llvm::ArrayRef<Triple> s)
        : addr{addr}, shape{s.begin(), s.end()} {}
    mlir::Value addr;
    std::vector<Triple> shape;
  };

  // CHARACTER type variable with its dependent type LEN parameter
  struct Char {
    explicit Char(mlir::Value addr, mlir::Value len) : addr{addr}, len{len} {}
    mlir::Value addr;
    mlir::Value len;
  };

  // CHARACTER array variable that has a simple shape
  struct CharShaped {
    explicit CharShaped(mlir::Value addr, mlir::Value len,
                        llvm::ArrayRef<mlir::Value> s)
        : addr{addr}, len{len}, shape{s.begin(), s.end()} {}
    mlir::Value addr;
    mlir::Value len;
    std::vector<mlir::Value> shape;
  };

  // CHARACTER array variable using triple notation
  struct CharFullDim {
    explicit CharFullDim(mlir::Value addr, mlir::Value len,
                         llvm::ArrayRef<Triple> s)
        : addr{addr}, len{len}, shape{s.begin(), s.end()} {}
    mlir::Value addr;
    mlir::Value len;
    std::vector<Triple> shape;
  };

  // Generalized derived type variable
  struct Derived {
    explicit Derived(mlir::Value addr, mlir::Value size,
                     llvm::ArrayRef<Triple> s,
                     llvm::ArrayRef<mlir::Value> parameters)
        : addr{addr}, size{size}, shape{s.begin(), s.end()},
          params{parameters.begin(), parameters.end()} {}
    mlir::Value addr;
    mlir::Value size;                // element size or null
    std::vector<Triple> shape;       // empty for scalar
    std::vector<mlir::Value> params; // LEN type parameters, if any
  };

  //===--------------------------------------------------------------------===//
  // Constructors
  //===--------------------------------------------------------------------===//

  SymIndex() : v{None{}} {}
  template <typename A>
  SymIndex(const A &x) : v{x} {}

  operator bool() const { return !std::holds_alternative<None>(v); }
  operator mlir::Value() const { return getAddr(); }

  //===--------------------------------------------------------------------===//
  // Accessors
  //===--------------------------------------------------------------------===//

  mlir::Value getAddr() const {
    return std::visit(common::visitors{
                          [](const None &) { return mlir::Value{}; },
                          [](const auto &x) { return x.addr; },
                      },
                      v);
  }

  llvm::Optional<mlir::Value> getCharLen() const {
    using T = llvm::Optional<mlir::Value>;
    return std::visit(common::visitors{
                          [](const Char &x) { return T{x.len}; },
                          [](const CharShaped &x) { return T{x.len}; },
                          [](const CharFullDim &x) { return T{x.len}; },
                          [](const auto &) { return T{}; },
                      },
                      v);
  }

  bool hasRank() const {
    return std::visit(common::visitors{
                          [](const Intrinsic &) { return false; },
                          [](const Char &) { return false; },
                          [](const None &) { return false; },
                          [](const auto &x) { return x.shape.size() > 0; },
                      },
                      v);
  }

  bool hasSimpleShape() const {
    return std::holds_alternative<Shaped>(v) ||
           std::holds_alternative<CharShaped>(v);
  }

  std::variant<Intrinsic, Shaped, FullDim, Char, CharShaped, CharFullDim,
               Derived, None>
      v;
};

/// Helper class to map front-end symbols to their MLIR representation. This
/// provides a way to lookup the ssa-values that comprise a Fortran symbol's
/// runtime attributes. These attributes include its address, its dynamic size,
/// dynamic bounds information for non-scalar entities, dynamic type parameters,
/// etc.
class SymMap {
public:
  /// Add `symbol` to the current map and bind an `index`.
  void addSymbol(semantics::SymbolRef sym, const SymIndex &index,
                 bool force = false) {
    if (force)
      erase(sym);
    assert(index && "cannot add an undefined symbol index");
    symbolMap.try_emplace(&*sym, index);
  }

  /// Add a trivial symbol mapping to an address.
  void addSymbol(semantics::SymbolRef sym, mlir::Value value,
                 bool force = false) {
    addSymbol(sym, SymIndex::Intrinsic(value), force);
  }

  /// Add a scalar CHARACTER mapping to an (address, len).
  void addCharSymbol(semantics::SymbolRef sym, mlir::Value value,
                     mlir::Value len, bool force = false) {
    addSymbol(sym, SymIndex::Char(value, len), force);
  }

  /// Add an array mapping with (address, shape).
  void addSymbolWithShape(semantics::SymbolRef sym, mlir::Value value,
                          llvm::ArrayRef<mlir::Value> shape,
                          bool force = false) {
    addSymbol(sym, SymIndex::Shaped(value, shape), force);
  }

  /// Add an array of CHARACTER mapping.
  void addCharSymbolWithShape(semantics::SymbolRef sym, mlir::Value value,
                              mlir::Value len,
                              llvm::ArrayRef<mlir::Value> shape,
                              bool force = false) {
    addSymbol(sym, SymIndex::CharShaped(value, len, shape), force);
  }

  /// Add an array mapping with triple notation.
  void addSymbolWithTriples(semantics::SymbolRef sym, mlir::Value value,
                            llvm::ArrayRef<SymIndex::Triple> shape,
                            bool force = false) {
    addSymbol(sym, SymIndex::FullDim(value, shape), force);
  }

  /// Add an array of CHARACTER with triple notation.
  void addCharSymbolWithTriples(semantics::SymbolRef sym, mlir::Value value,
                                mlir::Value len,
                                llvm::ArrayRef<SymIndex::Triple> shape,
                                bool force = false) {
    addSymbol(sym, SymIndex::CharFullDim(value, len, shape), force);
  }

  /// Generalized derived type mapping.
  void addDerivedSymbol(semantics::SymbolRef sym, mlir::Value value,
                        mlir::Value size,
                        llvm::ArrayRef<SymIndex::Triple> shape,
                        llvm::ArrayRef<mlir::Value> params,
                        bool force = false) {
    addSymbol(sym, SymIndex::Derived(value, size, shape, params), force);
  }

  /// Find `symbol` and return its value if it appears in the current mappings.
  SymIndex lookupSymbol(semantics::SymbolRef sym) {
    auto iter = symbolMap.find(&*sym);
    return (iter == symbolMap.end()) ? SymIndex() : iter->second;
  }

  void erase(semantics::SymbolRef sym) { symbolMap.erase(&*sym); }
  void clear() { symbolMap.clear(); }

private:
  llvm::DenseMap<const semantics::Symbol *, SymIndex> symbolMap;
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_FIRBUILDER_H
