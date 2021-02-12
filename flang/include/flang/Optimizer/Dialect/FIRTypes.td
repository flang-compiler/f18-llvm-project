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

include "flang/Optimizer/Dialect/FIRDialect.td"

//===----------------------------------------------------------------------===//
// FIR Types
//===----------------------------------------------------------------------===//

class FIR_Type<string name, string typeMnemonic> : TypeDef<fir_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def fir_FieldType : FIR_Type<"Field", "field"> {
  let summary = "A field (in a RecordType) argument's type";

  let description = [{
    The type of a field name. Implementations may defer the layout of a Fortran
    derived type until runtime. This implies that the runtime must be able to
    determine the offset of fields within the entity.
  }];

  let printer = [{
    $_printer << "field";
  }];

  let parser = [{
    return get(context);
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
