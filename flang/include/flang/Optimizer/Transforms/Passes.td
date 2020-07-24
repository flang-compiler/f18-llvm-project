//===-- Passes.td - Transforms pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes within the Optimizer/Transforms/
//  directory.
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_OPTIMIZER_TRANSFORMS_PASSES
#define FLANG_OPTIMIZER_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def AffineDialectPromotion : FunctionPass<"promote-to-affine"> {
  let summary = "Promotes fir.do_loop and fir.where to affine.for and affine.if where possible";
  let description = [{
    Convert fir operations which satisfy affine constraints to affine dialect.

    `fir.do_loop` to `affine.for` if the loops inside the body can be converted,
    index for memory loads and stores satisfy affine.apply criteria for symbols
    and dimensions.
    `fir.if` to `affine.if`, affine's condition uses an integer set (==, >=) and
    we try analyze the fir condition's parent operations to construct the integer
    set.
    `fir.load/store` to `affine.load/store` this includes also adding a dummy
    convert to cast `fir.array` to `memref` as affine only works with memref type.
  }];
  let constructor = "fir::createPromoteToAffinePass()";
}

def AffineDialectDemotion : FunctionPass<"demote-affine"> {
  let summary = "Converts affine.load and affine.store back to fir operations";
  let description = [{
    Affine dialect's default lowering for loads and stores is different from fir as
    it uses memref and thus we convert the memory operations back to fir loads and
    stores.
  }];
  let constructor = "fir::createAffineDemotionPass()";
}

def BasicCSE : FunctionPass<"basic-cse"> {
  let summary = "Basic common sub-expression elimination";
  let description = [{
      TODO
  }];
  let constructor = "fir::createCSEPass()";
}

def ControlFlowLowering : FunctionPass<"lower-control-flow"> {
  let summary = "Convert affine dialect, fir.select_type to standard dialect";
  let description = [{
      TODO
  }];
  let constructor = "fir::createControlFlowLoweringPass()";
}

def CFGConversion : FunctionPass<"cfg-conversion"> {
  let summary = "Convert FIR structured control flow ops to CFG ops.";
  let description = [{
      TODO
  }];
  let constructor = "fir::createFirToCfgPass()";
}

#endif // FLANG_OPTIMIZER_TRANSFORMS_PASSES
