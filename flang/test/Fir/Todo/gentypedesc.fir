// RUN: fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s 2>&1 | FileCheck %s
// XFAIL:*

// Test `fir.gentypedesc` conversion to llvm.
// Not implemented yet.

func @gentypedesc() {
  %0 = fir.gentypedesc !fir.type<derived3>
  return
}

// CHECK: not yet implemented fir.gentypedesc codegen
