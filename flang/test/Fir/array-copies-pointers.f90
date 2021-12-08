// Test array-copy-value pass (copy elision) with array assignment
// involving Fortran pointers. Focus in only on wether copy ellision
// is made or not.
// RUN: fir-opt %s --array-value-copy | FileCheck %s

// Test `pointer(:) = array(:)`
// TODO: array should have target attribute.
// CHECK-LABEL: func @maybe_overlap
// CHECK: fir.allocmem !fir.array<100xf32>
func @maybe_overlap(%arg0: !fir.ptr<!fir.array<100xf32>>, %arg1: !fir.ref<!fir.array<100xf32>>) {
  %c100 = arith.constant 100 : index
  %c99 = arith.constant 99 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.alloca f32
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2 = fir.array_load %arg0(%1) : (!fir.ptr<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %3 = fir.array_load %arg1(%1) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %4 = fir.do_loop %arg2 = %c0 to %c99 step %c1 unordered iter_args(%arg3 = %2) -> (!fir.array<100xf32>) {
    %5 = fir.array_fetch %3, %arg2 : (!fir.array<100xf32>, index) -> f32
    %6 = fir.array_update %arg3, %5, %arg2 : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
    fir.result %6 : !fir.array<100xf32>
  }
  fir.array_merge_store %2, %4 to %arg0 : !fir.array<100xf32>, !fir.array<100xf32>, !fir.ptr<!fir.array<100xf32>>
  return
}

// Test `pointer(:) = pointer(:)`
// CHECK-LABEL: func @no_overlap
// CHECK-NOT: fir.allocmem
func @no_overlap(%arg0: !fir.ptr<!fir.array<100xf32>>, %arg1: !fir.ref<!fir.array<100xf32>>) {
  %c100 = arith.constant 100 : index
  %c99 = arith.constant 99 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.alloca f32
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2 = fir.array_load %arg0(%1) : (!fir.ptr<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %3 = fir.do_loop %arg2 = %c0 to %c99 step %c1 unordered iter_args(%arg3 = %2) -> (!fir.array<100xf32>) {
    %4 = fir.array_fetch %2, %arg2 : (!fir.array<100xf32>, index) -> f32
    %5 = fir.array_update %arg3, %4, %arg2 : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
    fir.result %5 : !fir.array<100xf32>
  }
  fir.array_merge_store %2, %3 to %arg0 : !fir.array<100xf32>, !fir.array<100xf32>, !fir.ptr<!fir.array<100xf32>>
  return
}

// Test `array(:) = pointer(:)`
// TODO: array should have target attribute.
// CHECK-LABEL: func @maybe_overlap_2
// CHECK: fir.allocmem !fir.array<100xf32>
func @maybe_overlap_2(%arg0: !fir.ptr<!fir.array<100xf32>>, %arg1: !fir.ref<!fir.array<100xf32>>) {
  %c100 = arith.constant 100 : index
  %c99 = arith.constant 99 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.alloca f32
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2 = fir.array_load %arg0(%1) : (!fir.ptr<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %3 = fir.array_load %arg1(%1) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %4 = fir.do_loop %arg2 = %c0 to %c99 step %c1 unordered iter_args(%arg3 = %3) -> (!fir.array<100xf32>) {
    %5 = fir.array_fetch %2, %arg2 : (!fir.array<100xf32>, index) -> f32
    %6 = fir.array_update %arg3, %5, %arg2 : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
    fir.result %6 : !fir.array<100xf32>
  }
  fir.array_merge_store %3, %4 to %arg1 : !fir.array<100xf32>, !fir.array<100xf32>, !fir.ref<!fir.array<100xf32>>
  return
}

// Test `pointer1(:) = pointer2(:)`
// CHECK-LABEL: func @maybe_overlap_3
// CHECK: fir.allocmem !fir.array<100xf32>
func @maybe_overlap_3(%arg0: !fir.ptr<!fir.array<100xf32>>, %arg1: !fir.ptr<!fir.array<100xf32>>) {
  %c100 = arith.constant 100 : index
  %c99 = arith.constant 99 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.alloca f32
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2 = fir.array_load %arg0(%1) : (!fir.ptr<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %3 = fir.array_load %arg1(%1) : (!fir.ptr<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %4 = fir.do_loop %arg2 = %c0 to %c99 step %c1 unordered iter_args(%arg3 = %3) -> (!fir.array<100xf32>) {
    %5 = fir.array_fetch %2, %arg2 : (!fir.array<100xf32>, index) -> f32
    %6 = fir.array_update %arg3, %5, %arg2 : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
    fir.result %6 : !fir.array<100xf32>
  }
  fir.array_merge_store %3, %4 to %arg1 : !fir.array<100xf32>, !fir.array<100xf32>, !fir.ptr<!fir.array<100xf32>>
  return
}
