//===-- TypeConverter.h -- type conversion ----------------------*- C++ -*-===//
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

#ifndef FORTRAN_OPTIMIZER_CODEGEN_TYPECONVERTER_H
#define FORTRAN_OPTIMIZER_CODEGEN_TYPECONVERTER_H

#include "Target.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

namespace fir {

/// FIR type converter
/// This converts FIR types to LLVM types (for now)
class LLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  LLVMTypeConverter(mlir::ModuleOp module);

  // i32 is used here because LLVM wants i32 constants when indexing into struct
  // types. Indexing into other aggregate types is more flexible.
  mlir::Type offsetType();

  // i64 can be used to index into aggregates like arrays
  mlir::Type indexType();

  // Is an extended descriptor needed given the element type of a fir.box type ?
  // Extended descriptors are required for derived types.
  bool requiresExtendedDesc(mlir::Type boxElementType);

  // Magic value to indicate we do not know the rank of an entity, either
  // because it is assumed rank or because we have not determined it yet.
  static constexpr int unknownRank() { return -1; }
  // This corresponds to the descriptor as defined ISO_Fortran_binding.h and the
  // addendum defined in descriptor.h.
  mlir::Type convertBoxType(BoxType box, int rank = unknownRank());

  /// Convert fir.box type to the corresponding llvm struct type instead of a
  /// pointer to this struct type.
  mlir::Type convertBoxTypeAsStruct(BoxType box);

  // fir.boxproc<any>  -->  llvm<"{ any*, i8* }">
  mlir::Type convertBoxProcType(BoxProcType boxproc);

  unsigned characterBitsize(fir::CharacterType charTy);

  // fir.char<n>  -->  llvm<"ix*">   where ix is scaled by kind mapping
  mlir::Type convertCharType(fir::CharacterType charTy);

  // Convert a complex value's element type based on its Fortran kind.
  mlir::Type convertComplexPartType(fir::KindTy kind);

  // Use the target specifics to figure out how to map complex to LLVM IR. The
  // use of complex values in function signatures is handled before conversion
  // to LLVM IR dialect here.
  //
  // fir.complex<T> | std.complex<T>    --> llvm<"{t,t}">
  template <typename C>
  mlir::Type convertComplexType(C cmplx) {
    LLVM_DEBUG(llvm::dbgs() << "type convert: " << cmplx << '\n');
    auto eleTy = cmplx.getElementType();
    return convertType(specifics->complexMemoryType(eleTy));
  }

  // Get the default size of INTEGER. (The default size might have been set on
  // the command line.)
  mlir::Type getDefaultInt();

  template <typename A>
  mlir::Type convertPointerLike(A &ty) {
    mlir::Type eleTy = ty.getEleTy();
    // A sequence type is a special case. A sequence of runtime size on its
    // interior dimensions lowers to a memory reference. In that case, we
    // degenerate the array and do not want a the type to become `T**` but
    // merely `T*`.
    if (auto seqTy = eleTy.dyn_cast<fir::SequenceType>()) {
      if (!seqTy.hasConstantShape() ||
          characterWithDynamicLen(seqTy.getEleTy())) {
        if (seqTy.hasConstantInterior())
          return convertType(seqTy);
        eleTy = seqTy.getEleTy();
      }
    }
    // fir.ref<fir.box> is a special case because fir.box type is already
    // a pointer to a Fortran descriptor at the LLVM IR level. This implies
    // that a fir.ref<fir.box>, that is the address of fir.box is actually
    // the same as a fir.box at the LLVM level.
    // The distinction is kept in fir to denote when a descriptor is expected
    // to be mutable (fir.ref<fir.box>) and when it is not (fir.box).
    if (eleTy.isa<fir::BoxType>())
      return convertType(eleTy);

    return mlir::LLVM::LLVMPointerType::get(convertType(eleTy));
  }

  // convert a front-end kind value to either a std or LLVM IR dialect type
  // fir.real<n>  -->  llvm.anyfloat  where anyfloat is a kind mapping
  mlir::Type convertRealType(fir::KindTy kind);

  // fir.type<name(p : TY'...){f : TY...}>  -->  llvm<"%name = { ty... }">
  mlir::Type convertRecordType(fir::RecordType derived);

  // fir.array<c ... :any>  -->  llvm<"[...[c x any]]">
  mlir::Type convertSequenceType(SequenceType seq);

  // fir.tdesc<any>  -->  llvm<"i8*">
  // FIXME: for now use a void*, however pointer identity is not sufficient for
  // the f18 object v. class distinction
  mlir::Type convertTypeDescType(mlir::MLIRContext *ctx);

  /// Convert llvm::Type::TypeID to mlir::Type
  mlir::Type fromRealTypeID(llvm::Type::TypeID typeID, fir::KindTy kind);

  KindMapping &getKindMap();

private:
  KindMapping kindMapping;
  std::unique_ptr<CodeGenSpecifics> specifics;
  static StringMap<mlir::Type> identStructCache;
};

} // namespace fir

#endif // FORTRAN_OPTIMIZER_CODEGEN_TYPECONVERTER_H
