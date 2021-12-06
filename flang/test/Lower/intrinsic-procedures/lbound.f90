! RUN: bbc -emit-fir %s -o - | FileCheck %s


! CHECK-LABEL: func @_QPlbound_test(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xf32>>,
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i64>,
! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<i64>) {
subroutine lbound_test(a, dim, res)
  real, dimension(:, :) :: a
  integer(8):: dim, res
! CHECK:         %[[VAL_3:.*]] = arith.constant 1 : i64
! CHECK:         fir.store %[[VAL_3]] to %[[VAL_2]] : !fir.ref<i64>
  res = lbound(a, dim, 8)
end subroutine

! CHECK-LABEL: func @_QPlbound_test_2(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xf32>>,
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i64>,
! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<i64>) {
subroutine lbound_test_2(a, dim, res)
  real, dimension(:, 2:) :: a
  integer(8):: dim, res
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.array<2xi64>
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
! CHECK:         %[[VAL_6:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
! CHECK:         %[[VAL_8:.*]] = fir.load %[[VAL_1]] : !fir.ref<i64>
! CHECK:         %[[VAL_9:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_10:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_9]] : (!fir.ref<!fir.array<2xi64>>, index) -> !fir.ref<i64>
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_5]] : (index) -> i64
! CHECK:         fir.store %[[VAL_11]] to %[[VAL_10]] : !fir.ref<i64>
! CHECK:         %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_13:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_12]] : (!fir.ref<!fir.array<2xi64>>, index) -> !fir.ref<i64>
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_7]] : (index) -> i64
! CHECK:         fir.store %[[VAL_14]] to %[[VAL_13]] : !fir.ref<i64>
! CHECK:         %[[VAL_15:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_8]] : (!fir.ref<!fir.array<2xi64>>, i64) -> !fir.ref<i64>
! CHECK:         %[[VAL_16:.*]] = fir.load %[[VAL_15]] : !fir.ref<i64>
! CHECK:         fir.store %[[VAL_16]] to %[[VAL_2]] : !fir.ref<i64>
  res = lbound(a, dim, 8)
end subroutine

! CHECK-LABEL: func @_QPlbound_test_3(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<9x?xf32>>,
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i64>,
! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<i64>) {
subroutine lbound_test_3(a, dim, res)
  real, dimension(2:10, 3:*) :: a
  integer(8):: dim, res
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.array<2xi64>
! CHECK:         %[[VAL_4:.*]] = arith.constant 2 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 3 : index
! CHECK:         %[[VAL_6:.*]] = fir.load %[[VAL_1]] : !fir.ref<i64>
! CHECK:         %[[VAL_7:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_8:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_7]] : (!fir.ref<!fir.array<2xi64>>, index) -> !fir.ref<i64>
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_4]] : (index) -> i64
! CHECK:         fir.store %[[VAL_9]] to %[[VAL_8]] : !fir.ref<i64>
! CHECK:         %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_11:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_10]] : (!fir.ref<!fir.array<2xi64>>, index) -> !fir.ref<i64>
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_5]] : (index) -> i64
! CHECK:         fir.store %[[VAL_12]] to %[[VAL_11]] : !fir.ref<i64>
! CHECK:         %[[VAL_13:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_6]] : (!fir.ref<!fir.array<2xi64>>, i64) -> !fir.ref<i64>
! CHECK:         %[[VAL_14:.*]] = fir.load %[[VAL_13]] : !fir.ref<i64>
! CHECK:         fir.store %[[VAL_14]] to %[[VAL_2]] : !fir.ref<i64>
  res = lbound(a, dim, 8)
end subroutine
