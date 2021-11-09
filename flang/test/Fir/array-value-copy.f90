// Test array-copy-value pass
// RUN: fir-opt %s --array-value-copy | FileCheck %s

// CHECK-LABEL:   func @derived_type_component_overlap(
// CHECK-SAME:              %[[VAL_0:.*]]: !fir.ref<!fir.array<100x!fir.type<t{i:i32}>>>) {
func @derived_type_component_overlap(%arg0: !fir.ref<!fir.array<100x!fir.type<t{i:i32}>>>) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c1_0 = arith.constant 1 : index
  %c99 = arith.constant 99 : index
  %c100 = arith.constant 100 : index
  %0 = fir.shape %c100 : (index) -> !fir.shape<1>
  %1 = fir.field_index i, !fir.type<t{i:i32}>
  %2 = fir.slice %c1_0, %c100, %c1_0 path %1 : (index, index, index, !fir.field) -> !fir.slice<1>
  %3 = fir.array_load %arg0(%0) [%2] : (!fir.ref<!fir.array<100x!fir.type<t{i:i32}>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<100xi32>
  %4 = fir.slice %c100, %c1_0, %c1 path %1 : (index, index, index, !fir.field) -> !fir.slice<1>
  %5 = fir.array_load %arg0(%0) [%4] : (!fir.ref<!fir.array<100x!fir.type<t{i:i32}>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<100xi32>
  %6 = fir.do_loop %arg1 = %c0 to %c99 step %c1_0 unordered iter_args(%arg2 = %3) -> (!fir.array<100xi32>) {
    %7 = fir.array_fetch %5, %arg1 : (!fir.array<100xi32>, index) -> i32
    %8 = fir.array_update %arg2, %7, %arg1 : (!fir.array<100xi32>, i32, index) -> !fir.array<100xi32>
    fir.result %8 : !fir.array<100xi32>
  }
  fir.array_merge_store %3, %6 to %arg0[%2] : !fir.array<100xi32>, !fir.array<100xi32>, !fir.ref<!fir.array<100x!fir.type<t{i:i32}>>>, !fir.slice<1>

// Test the copy-in of a derived type array LHS in a temp 

// CHECK:           %[[VAL_5:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_9:.*]] = fir.allocmem !fir.array<100x!fir.type<t{i:i32}>>
// CHECK:           fir.do_loop %[[VAL_14:.*]] = %{{.*}} {
// CHECK:             %[[VAL_15:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_16:.*]] = arith.addi %[[VAL_14]], %[[VAL_15]] : index
// CHECK:             %[[VAL_17:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_6]]) %[[VAL_16]] : (!fir.ref<!fir.array<100x!fir.type<t{i:i32}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<t{i:i32}>>
// CHECK:             %[[VAL_18:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_19:.*]] = arith.addi %[[VAL_14]], %[[VAL_18]] : index
// CHECK:             %[[VAL_20:.*]] = fir.array_coor %[[VAL_9]](%[[VAL_6]]) %[[VAL_19]] : (!fir.heap<!fir.array<100x!fir.type<t{i:i32}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<t{i:i32}>>
// CHECK:             %[[VAL_21:.*]] = fir.load %[[VAL_17]] : !fir.ref<!fir.type<t{i:i32}>>
// CHECK:             fir.store %[[VAL_21]] to %[[VAL_20]] : !fir.ref<!fir.type<t{i:i32}>>
// CHECK:           }

// Actual assignment and copy-out

// CHECK:           fir.do_loop
// CHECK:           fir.do_loop
// CHECK:           fir.freemem %[[VAL_9]] : !fir.heap<!fir.array<100x!fir.type<t{i:i32}>>>

  return
}


// Test that lower bounds from fir.box array are applied in the array value copy pass.
func @pointer_lower_bounds(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, %arg1: !fir.ref<!fir.array<100xf32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %c99 = arith.constant 99 : index
  %c100 = arith.constant 100 : index
  %c504 = arith.constant 504 : index
  %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> 
  %1:3 = fir.box_dims %0, %c0 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
  %2 = fir.shift %1#0 : (index) -> !fir.shift<1>
  %3 = fir.slice %c5, %c504, %c5 : (index, index, index) -> !fir.slice<1>
  %4 = fir.array_load %0(%2) [%3] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>, !fir.slice<1>) -> !fir.array<?xf32> 
  %5 = fir.shape %c100 : (index) -> !fir.shape<1>
  %6 = fir.array_load %arg1(%5) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %7 = fir.do_loop %arg2 = %c0 to %c99 step %c1 unordered iter_args(%arg3 = %4) -> (!fir.array<?xf32>) {
    %8 = fir.array_fetch %6, %arg2 : (!fir.array<100xf32>, index) -> f32
    %9 = fir.array_update %arg3, %8, %arg2 : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
    fir.result %9 : !fir.array<?xf32>
  }
  fir.array_merge_store %4, %7 to %0[%3] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.slice<1>
  return

// Test that a shape_shift generation and related pointer addressing.

// CHECK-LABEL: func @pointer_lower_bounds(
// CHECK-SAME:                             %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>,
// CHECK-SAME:                             %[[VAL_1:.*]]: !fir.ref<!fir.array<100xf32>>) {
// CHECK:         %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:         %[[VAL_8:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
// CHECK:         %[[VAL_9:.*]]:3 = fir.box_dims %[[VAL_8]], %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
// CHECK:         %[[VAL_12:.*]] = arith.constant 0 : index
// CHECK:         %[[VAL_13:.*]]:3 = fir.box_dims %[[VAL_8]], %[[VAL_12]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
// CHECK:         %[[VAL_14:.*]] = fir.shape_shift %[[VAL_9]]#0, %[[VAL_13]]#1 : (index, index) -> !fir.shapeshift<1>
// CHECK:         fir.do_loop %[[VAL_20:.*]] = {{.*}} {
// CHECK:           %[[VAL_21:.*]] = arith.addi %[[VAL_20]], %[[VAL_9]]#0 : index
// CHECK:           fir.array_coor %[[VAL_8]](%[[VAL_14]]) %[[VAL_21]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
// CHECK:         fir.do_loop
// CHECK:         fir.freemem

}

