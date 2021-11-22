// RUN: fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s 2>&1 | FileCheck %s
// XFAIL:*

// Test `fir.select_type` conversion to llvm.
// Not implemented yet.

func @bar_select_type(%arg : !fir.box<!fir.type<derived3{f:f32}>>) -> i32 {
  %0 = arith.constant 1 : i32
  %2 = arith.constant 3 : i32

  fir.select_type %arg : !fir.box<!fir.type<derived3{f:f32}>> [
    #fir.instance<!fir.int<4>>,^bb1(%0:i32),
    #fir.instance<!fir.int<8>>,^bb2(%2:i32),
    unit,^bb5 ]
^bb1(%a : i32) :
  return %a : i32
^bb2(%b : i32) :
  return %b : i32
^bb5 :
  %zero = arith.constant 0 : i32
  return %zero : i32
}

// CHECK: not yet implemented fir.select_type codegen
