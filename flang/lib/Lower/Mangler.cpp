//===-- Mangler.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/Mangler.h"
#include "flang/Common/reference.h"
#include "flang/Lower/Utils.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Semantics/tools.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

// recursively build the vector of module scopes
static void moduleNames(const Fortran::semantics::Scope &scope,
                        llvm::SmallVector<llvm::StringRef, 2> &result) {
  if (scope.kind() == Fortran::semantics::Scope::Kind::Global) {
    return;
  }
  moduleNames(scope.parent(), result);
  if (scope.kind() == Fortran::semantics::Scope::Kind::Module)
    if (auto *symbol = scope.symbol())
      result.emplace_back(toStringRef(symbol->name()));
}

static llvm::SmallVector<llvm::StringRef, 2>
moduleNames(const Fortran::semantics::Symbol &symbol) {
  const auto &scope = symbol.owner();
  llvm::SmallVector<llvm::StringRef, 2> result;
  moduleNames(scope, result);
  return result;
}

static llvm::Optional<llvm::StringRef>
hostName(const Fortran::semantics::Symbol &symbol) {
  const auto &scope = symbol.owner();
  if (scope.kind() == Fortran::semantics::Scope::Kind::Subprogram) {
    assert(scope.symbol() && "subprogram scope must have a symbol");
    return {toStringRef(scope.symbol()->name())};
  }
  return {};
}

static const Fortran::semantics::Symbol *
findInterfaceIfSeperateMP(const Fortran::semantics::Symbol &symbol) {
  const auto &scope = symbol.owner();
  if (symbol.attrs().test(Fortran::semantics::Attr::MODULE) &&
      scope.IsSubmodule()) {
    // FIXME symbol from MpSubprogramStmt do not seem to have
    // Attr::MODULE set.
    const auto *iface = scope.parent().FindSymbol(symbol.name());
    assert(iface && "Separate module proc must be declared");
    return iface;
  }
  return nullptr;
}

// Mangle the name of `symbol` to make it unique within FIR's symbol table using
// the FIR name mangler, `mangler`
std::string
Fortran::lower::mangle::mangleName(fir::NameUniquer &uniquer,
                                   const Fortran::semantics::Symbol &symbol) {
  // Resolve host and module association before mangling
  const auto &ultimateSymbol = symbol.GetUltimate();
  auto symbolName = ultimateSymbol.name();
  return std::visit(
      Fortran::common::visitors{
          [&](const Fortran::semantics::MainProgramDetails &) {
            return uniquer.doProgramEntry().str();
          },
          [&](const Fortran::semantics::SubprogramDetails &) {
            // Separate module subprograms must be mangled according to the
            // scope where they were declared (the symbol we have is the
            // definition).
            const auto *interface = &ultimateSymbol;
            if (const auto *mpIface = findInterfaceIfSeperateMP(ultimateSymbol))
              interface = mpIface;
            auto modNames = moduleNames(*interface);
            return uniquer.doProcedure(modNames, hostName(*interface),
                                       toStringRef(symbolName));
          },
          [&](const Fortran::semantics::ProcEntityDetails &) {
            auto modNames = moduleNames(ultimateSymbol);
            return uniquer.doProcedure(modNames, hostName(ultimateSymbol),
                                       toStringRef(symbolName));
          },
          [](const auto &) -> std::string {
            assert(false);
            return {};
          },
      },
      ultimateSymbol.details());
}

std::string Fortran::lower::mangle::demangleName(llvm::StringRef name) {
  auto result = fir::NameUniquer::deconstruct(name);
  return result.second.name;
}
