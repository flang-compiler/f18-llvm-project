// RUN: tco %s | FileCheck %s

// CHECK-LABEL: @f
func @f(%x : i1) -> i1 {
  %1 = fir.convert %x : (i1) -> !fir.logical<1>
  %2 = fir.convert %1 : (!fir.logical<1>) -> i1
  // CHECK: ret i1 %0
  return %2 : i1
}

// CHECK-LABEL: @g
func @g(%x : !fir.logical<2>) -> !fir.logical<2> {
  %1 = fir.convert %x : (!fir.logical<2>) -> i1
  %2 = fir.convert %1 : (i1) -> !fir.logical<2>
  // CHECK: ret i16 %0
  return %2 : !fir.logical<2>
}

// CHECK-LABEL: @h
func @h(%x : !fir.int<4>) -> !fir.int<4> {
  %1 = fir.convert %x : (!fir.int<4>) -> !fir.int<4>
  %2 = fir.convert %1 : (!fir.int<4>) -> !fir.int<4>
  // CHECK-NEXT: ret i32 %0
  return %2 : !fir.int<4>
}
