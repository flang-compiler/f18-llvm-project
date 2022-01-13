// RUN: fir-opt %s | tco | FileCheck %s

// CHECK-LABEL: define i32* @f1()
func @f1() -> !fir.ref<i32> {
  // CHECK: alloca i32, i64 1
  %1 = fir.alloca i32
  return %1 : !fir.ref<i32>
}

// CHECK-LABEL: define i32* @f2()
func @f2() -> !fir.ref<i32> {
  %0 = arith.constant 100 : index
  // CHECK: alloca i32, i64 100
  %1 = fir.alloca i32, %0
  return %1 : !fir.ref<i32>
}

// CHECK-LABEL: define i32* @f3()
func @f3() -> !fir.heap<i32> {
  // CHECK: call i8* @malloc(i64 4)
  %1 = fir.allocmem i32
  return %1 : !fir.heap<i32>
}

// CHECK-LABEL: define i32* @f4()
func @f4() -> !fir.heap<i32> {
  %0 = arith.constant 100 : index
  // CHECK: call i8* @malloc(i64 400)
  %1 = fir.allocmem i32, %0
  return %1 : !fir.heap<i32>
}

// CHECK-LABEL: define i32** @f5()
func @f5() -> !fir.ref<!fir.ptr<!fir.array<?xi32>>> {
  // CHECK: alloca i32*, i64 1
  %1 = fir.alloca !fir.ptr<!fir.array<?xi32>>
  return %1 : !fir.ref<!fir.ptr<!fir.array<?xi32>>>
}

// CHECK-LABEL: define i8* @char_array_alloca(
// CHECK-SAME: i32 %[[l:.*]], i64 %[[e:.*]])
func @char_array_alloca(%l: i32, %e : index) -> !fir.ref<!fir.array<?x?x!fir.char<1,?>>> {
  // CHECK: %[[lcast:.*]] = sext i32 %[[l]] to i64
  // CHECK: %[[prod:.*]] = mul i64 %[[lcast]], %[[e]]
  // CHECK: %[[size:.*]] = mul i64 %[[prod]], %[[e]]
  // CHECK: alloca i8, i64 %[[size]]
  %a = fir.alloca !fir.array<?x?x!fir.char<1,?>>(%l : i32), %e, %e
  return %a :  !fir.ref<!fir.array<?x?x!fir.char<1,?>>>
}

// Constant factor of 60 (4*3*5) must be included.
// CHECK-LABEL: define i32* @array_with_holes(
// CHECK-SAME: i64 %[[a:.*]], i64 %[[b:.*]])
func @array_with_holes(%0 : index, %1 : index) -> !fir.ref<!fir.array<4x?x3x?x5xi32>> {
  // CHECK: %[[prod1:.*]] = mul i64 60, %[[a]]
  // CHECK: %[[prod2:.*]] = mul i64 %[[prod1]], %[[b]]
  // CHECK: alloca i32, i64 %[[prod2]]
  %a = fir.alloca !fir.array<4x?x3x?x5xi32>, %0, %1
  return %a : !fir.ref<!fir.array<4x?x3x?x5xi32>>
}
