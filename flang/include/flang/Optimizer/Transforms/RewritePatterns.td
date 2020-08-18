//===-- RewritePatterns.td - FIR Rewrite Patterns -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines pattern rewrites for fir optimizations
///
//===----------------------------------------------------------------------===//

#ifndef FIR_REWRITE_PATTERNS
#define FIR_REWRITE_PATTERNS

include "mlir/IR/OpBase.td"
include "flang/Optimizer/Dialect/FIROps.td"

def IntegerType : Constraint<CPred<"$0.getType().isa<fir::IntType>()">>;
def SmallerWidth : Constraint<CPred<"$0.getType().getIntOrFloatBitWidth() < $1.getType().getIntOrFloatBitWidth()">>;

def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;

def ConvertConvertOptPattern : Pat<(fir_ConvertOp (fir_ConvertOp $arg)),
                                   (fir_ConvertOp $arg),
                                   [(IntegerType $arg)]>;

def RedundantConvertOptPattern : Pat<(fir_ConvertOp:$res $arg),
                                     (replaceWithValue $arg),
                                     [(TypesAreIdentical $res, $arg),
                                      (IntegerType $arg)]>;

def CombineConvertOptPattern : Pat<(fir_ConvertOp:$res (fir_ConvertOp:$irm $arg)),
                                     (replaceWithValue $arg),
                                     [(TypesAreIdentical $res, $arg),
                                     (IntegerType $arg), (IntegerType $irm),
                                     (SmallerWidth $arg $irm)]>;

#endif // FIR_REWRITE_PATTERNS
