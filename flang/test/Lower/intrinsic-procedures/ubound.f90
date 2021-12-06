! RUN: bbc -emit-fir %s -o - | FileCheck %s


! CHECK-LABEL: func @_QPubound_test(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xf32>>,
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i64>,
! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<i64>) {
subroutine ubound_test(a, dim, res)
  real, dimension(:, :) :: a
  integer(8):: dim, res
! CHECK:         %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<i64>
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
! CHECK:         %[[VAL_5:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_6:.*]] = arith.subi %[[VAL_4]], %[[VAL_5]] : index
! CHECK:         %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_6]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]]#1 : (index) -> i64
! CHECK:         %[[VAL_9:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_10:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_11:.*]] = arith.subi %[[VAL_9]], %[[VAL_10]] : i64
! CHECK:         %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_8]] : i64
! CHECK:         fir.store %[[VAL_12]] to %[[VAL_2]] : !fir.ref<i64>
  res = ubound(a, dim, 8)
end subroutine

! CHECK-LABEL: func @_QPubound_test_2(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xf32>>,
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i64>,
! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<i64>) {
subroutine ubound_test_2(a, dim, res)
  real, dimension(2:, 3:) :: a
  integer(8):: dim, res
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.array<2xi64>
! CHECK:         %[[VAL_4:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
! CHECK:         %[[VAL_6:.*]] = arith.constant 3 : i64
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
! CHECK:         %[[VAL_8:.*]] = fir.load %[[VAL_1]] : !fir.ref<i64>
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
! CHECK:         %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_11:.*]] = arith.subi %[[VAL_9]], %[[VAL_10]] : index
! CHECK:         %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_11]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_13:.*]] = fir.convert %[[VAL_12]]#1 : (index) -> i64
! CHECK:         %[[VAL_14:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_15:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_14]] : (!fir.ref<!fir.array<2xi64>>, index) -> !fir.ref<i64>
! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_5]] : (index) -> i64
! CHECK:         fir.store %[[VAL_16]] to %[[VAL_15]] : !fir.ref<i64>
! CHECK:         %[[VAL_17:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_18:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_17]] : (!fir.ref<!fir.array<2xi64>>, index) -> !fir.ref<i64>
! CHECK:         %[[VAL_19:.*]] = fir.convert %[[VAL_7]] : (index) -> i64
! CHECK:         fir.store %[[VAL_19]] to %[[VAL_18]] : !fir.ref<i64>
! CHECK:         %[[VAL_20:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_8]] : (!fir.ref<!fir.array<2xi64>>, i64) -> !fir.ref<i64>
! CHECK:         %[[VAL_21:.*]] = fir.load %[[VAL_20]] : !fir.ref<i64>
! CHECK:         %[[VAL_22:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_23:.*]] = arith.subi %[[VAL_21]], %[[VAL_22]] : i64
! CHECK:         %[[VAL_24:.*]] = arith.addi %[[VAL_23]], %[[VAL_13]] : i64
! CHECK:         fir.store %[[VAL_24]] to %[[VAL_2]] : !fir.ref<i64>
  res = ubound(a, dim, 8)
end subroutine

! CHECK-LABEL: func @_QPubound_test_3(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<10x20x?xf32>>,
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i64>,
! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<i64>) {
subroutine ubound_test_3(a, dim, res)
  real, dimension(10, 20, *) :: a
  integer(8):: dim, res
! CHECK:         %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_4:.*]] = arith.constant 20 : index
! CHECK:         %[[VAL_5:.*]] = fir.undefined index
! CHECK:         %[[VAL_6:.*]] = fir.shape %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] : (index, index, index) -> !fir.shape<3>
! CHECK:         %[[VAL_7:.*]] = fir.embox %[[VAL_0]](%[[VAL_6]]) : (!fir.ref<!fir.array<10x20x?xf32>>, !fir.shape<3>) -> !fir.box<!fir.array<10x20x?xf32>>
! CHECK:         %[[VAL_8:.*]] = fir.load %[[VAL_1]] : !fir.ref<i64>
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
! CHECK:         %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_11:.*]] = arith.subi %[[VAL_9]], %[[VAL_10]] : index
! CHECK:         %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_7]], %[[VAL_11]] : (!fir.box<!fir.array<10x20x?xf32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_13:.*]] = fir.convert %[[VAL_12]]#1 : (index) -> i64
! CHECK:         %[[VAL_14:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_15:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_16:.*]] = arith.subi %[[VAL_14]], %[[VAL_15]] : i64
! CHECK:         %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_13]] : i64
! CHECK:         fir.store %[[VAL_17]] to %[[VAL_2]] : !fir.ref<i64>
! CHECK:         return
  res = ubound(a, dim, 8)
end subroutine
