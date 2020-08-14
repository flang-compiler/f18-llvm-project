//===-- KindMapping.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "mlir/IR/Module.h"
#include "llvm/Support/Host.h"

static constexpr const char *tripleName = "fir.triple";

void fir::setTargetTriple(mlir::ModuleOp mod, llvm::Triple &triple) {
  mod.setAttr(tripleName, fir::OpaqueAttr::get(mod.getContext(), &triple));
}

llvm::Triple *getTargetTriple(mlir::ModuleOp mod) {
  if (auto triple = mod.getAttrOfType<fir::OpaqueAttr>(tripleName))
    return static_cast<llvm::Triple *>(triple.getPointer());
  return nullptr;
}

static constexpr const char *uniquerName = "fir.uniquer";

void fir::setNameUniquer(mlir::ModuleOp mod, fir::NameUniquer &uniquer) {
  mod.setAttr(uniquerName, fir::OpaqueAttr::get(mod.getContext(), &uniquer));
}

fir::NameUniquer *getNameUniquer(mlir::ModuleOp mod) {
  if (auto triple = mod.getAttrOfType<fir::OpaqueAttr>(uniquerName))
    return static_cast<fir::NameUniquer *>(triple.getPointer());
  return nullptr;
}

static constexpr const char *kindMapName = "fir.kindmap";

void fir::setKindMapping(mlir::ModuleOp mod, fir::KindMapping &kindMap) {
  mod.setAttr(kindMapName, fir::OpaqueAttr::get(mod.getContext(), &kindMap));
}

fir::KindMapping *getKindMapping(mlir::ModuleOp mod) {
  if (auto triple = mod.getAttrOfType<fir::OpaqueAttr>(kindMapName))
    return static_cast<fir::KindMapping *>(triple.getPointer());
  return nullptr;
}

std::string fir::determineTargetTriple(llvm::StringRef triple) {
  // Treat "native" and "" as stand-ins for the host machine.
  if (triple.empty() || (triple == "native"))
    return llvm::sys::getHostCPUName().str();
  // TODO: normalize the triple?
  return triple.str();
}
