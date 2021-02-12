//===- FIRTypes.td - FIR types -----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the FIR dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef FIR_DIALECT_FIR_TYPES
#define FIR_DIALECT_FIR_TYPES

//===----------------------------------------------------------------------===//
// FIR Types
//===----------------------------------------------------------------------===//

class FIR_Type<string name, string typeMnemonic> : TypeDef<fir_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def fir_ComplexType : FIR_Type<"Complex", "complex"> {
  let summary = "Complex type";

  let description = [{
    Model of a Fortran COMPLEX intrinsic type, including the KIND type
    parameter. COMPLEX is a floating point type with a real and imaginary
    member.
  }];

  let parameters = (ins "KindTy":$kind);

  let printer = [{
    $_printer << "complex<" << getFKind() << '>';
  }];

  let genAccessors = 1;

  let extraClassDeclaration = [{
    using KindTy = unsigned;

    KindTy getFKind() const { return getKind(); }

    mlir::Type getElementType() const;
  }];
}

def ShapeType : FIR_Type<"Shape", "shape"> {
  let summary = "shape of a multidimensional array object";

  let description = [{
    Type of a vector of runtime values that define the shape of a
    multidimensional array object. The vector is the extents of each array
    dimension. The rank of a ShapeType must be at least 1.
  }];

  let parameters = (ins "unsigned":$rank);

  let printer = [{
    $_printer << "shape<" << getImpl()->rank << ">";
  }];

  let parser = [{
    if ($_parser.parseLess())
      return Type();
    int rank;
    if ($_parser.parseInteger(rank))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get(context, rank);
  }];
}

#endif // FIR_DIALECT_FIR_TYPES
