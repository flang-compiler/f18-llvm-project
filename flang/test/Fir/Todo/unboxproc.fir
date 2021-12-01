// RUN: fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s 2>&1 | FileCheck %s
// XFAIL:*
// UpstreamDiff: We want to keep the conversion for further development.

// Test `fir.unboxproc` conversion to llvm.
// Not implemented yet.
// Currently fails since coversion for boxproc type is not implemented.

// CHECK: failed to legalize operation 'builtin.func'
func @boxing_match(%bproc: !fir.boxproc<(i32) -> ()>) {
  %ubproc:2 = fir.unboxproc %bproc : (!fir.boxproc<(i32) -> ()>) -> ((i32) -> (), !fir.ref<tuple<i32,f64>>)
  return
}
