// RUN: fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s 2>&1 | FileCheck %s
// XFAIL: *
// UpstreamDiff: We want to retain the conversion. Difficult to test since if
// we erase the op then there is no terminator in a function and currently no
// other op accepts fir.end.

// Test `fir.end` conversion to llvm.

// CHECK-LABEL: end_test
func @end_test(%i : i1) {
  "fir.end"() : () -> ()
}
