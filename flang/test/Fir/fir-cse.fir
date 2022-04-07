// Test the FIR CSE pass

// RUN: fir-opt %s | tco | FileCheck %s

// CHECK-LABEL: @fun
func @fun(%a : !fir.ref<i64>) -> i64 {
  // CHECK: load i64
  %1 = fir.load %a : !fir.ref<i64>
  %2 = fir.load %a : !fir.ref<i64>
  // CHECK-NOT: load i64
  // CHECK-COUNT-6: add i64
  %3 = arith.addi %1, %2 : i64
  %4 = fir.load %a : !fir.ref<i64>
  %5 = arith.addi %3, %4 : i64
  %6 = fir.load %a : !fir.ref<i64>
  %7 = arith.addi %5, %6 : i64
  %8 = fir.load %a : !fir.ref<i64>
  %9 = arith.addi %7, %8 : i64
  %10 = fir.load %a : !fir.ref<i64>
  %11 = arith.addi %10, %9 : i64
  %12 = fir.load %a : !fir.ref<i64>
  %13 = arith.addi %11, %12 : i64
  // CHECK-NEXT: ret i64
  return %13 : i64
}

// CHECK-LABEL: @bar
func private @bar(%a : !fir.ref<i64>) -> i64

// CHECK-LABEL: @fun2
func @fun2(%a : !fir.ref<i64>) -> i64 {
  // CHECK: load i64
  %1 = fir.load %a : !fir.ref<i64>
  // CHECK-NEXT: call i64
  %2 = fir.call @bar(%a) { pure = true } : (!fir.ref<i64>) -> i64
  // CHECK-COUNT-6: add i64
  %3 = arith.addi %1, %2 : i64
  %4 = fir.call @bar(%a) { pure = true } : (!fir.ref<i64>) -> i64
  %5 = arith.addi %3, %4 : i64
  %6 = fir.call @bar(%a) { pure = true } : (!fir.ref<i64>) -> i64
  %7 = arith.addi %5, %6 : i64
  %8 = fir.call @bar(%a) { pure = true } : (!fir.ref<i64>) -> i64
  %9 = arith.addi %7, %8 : i64
  %10 = fir.call @bar(%a) { pure = true } : (!fir.ref<i64>) -> i64
  %11 = arith.addi %10, %9 : i64
  %12 = fir.call @bar(%a) { pure = true } : (!fir.ref<i64>) -> i64
  %13 = arith.addi %11, %12 : i64
  // CHECK-NEXT: ret i64
  return %13 : i64
}

// Negative test: do not merge loads when an op with regions is between
// CHECK-LABEL: @foo
func @foo(%arg0: !fir.ref<f32>) -> f32 {
  // CHECK: %[[var:.*]] = alloca float
  %0 = fir.alloca f32 {name = "x"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  fir.store %1 to %0 : !fir.ref<f32>
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: load float, float* %[[var]],
  %2 = fir.load %0 : !fir.ref<f32>
  %3 = arith.cmpf olt, %2, %cst : f32
  fir.if %3 {
    // CHECK: load float, float* %[[var]]
    %7 = fir.load %0 : !fir.ref<f32>
    %8 = arith.negf %7 : f32
    fir.store %8 to %0 : !fir.ref<f32>
  }
  %cst_0 = arith.constant 1.000000e+00 : f32
  // CHECK: load float, float* %[[var]]
  %4 = fir.load %0 : !fir.ref<f32>
  %5 = arith.addf %4, %cst_0 : f32
  fir.store %5 to %0 : !fir.ref<f32>
  // CHECK: load float, float* %[[var]]
  %6 = fir.load %0 : !fir.ref<f32>
  return %6 : f32
}
