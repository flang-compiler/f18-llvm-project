! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPsize_test() {
subroutine size_test()
  real, dimension(1:10, -10:10) :: a
  integer :: dim = 1
  integer :: iSize
! CHECK:         %[[VAL_0:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_1:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_2:.*]] = arith.constant -10 : index
! CHECK:         %[[VAL_3:.*]] = arith.constant 21 : index
! CHECK:         %[[VAL_4:.*]] = fir.alloca !fir.array<10x21xf32> {bindc_name = "a", uniq_name = "_QFsize_testEa"}
! CHECK:         %[[VAL_5:.*]] = fir.address_of(@_QFsize_testEdim) : !fir.ref<i32>
! CHECK:         %[[VAL_6:.*]] = fir.alloca i32 {bindc_name = "isize", uniq_name = "_QFsize_testEisize"}
! CHECK:         %[[VAL_7:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i64) -> index
! CHECK:         %[[VAL_9:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
! CHECK:         %[[VAL_11:.*]] = arith.constant 5 : i64
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
! CHECK:         %[[VAL_13:.*]] = arith.constant -1 : i64
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
! CHECK:         %[[VAL_15:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i64) -> index
! CHECK:         %[[VAL_17:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i64) -> index
! CHECK:         %[[VAL_19:.*]] = fir.shape_shift %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:         %[[VAL_20:.*]] = fir.slice %[[VAL_8]], %[[VAL_12]], %[[VAL_10]], %[[VAL_14]], %[[VAL_18]], %[[VAL_16]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:         %[[VAL_21:.*]] = fir.embox %[[VAL_4]](%[[VAL_19]]) {{\[}}%[[VAL_20]]] : (!fir.ref<!fir.array<10x21xf32>>, !fir.shapeshift<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xf32>>
! CHECK:         %[[VAL_22:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
! CHECK:         %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i32) -> index
! CHECK:         %[[VAL_24:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_25:.*]] = arith.subi %[[VAL_23]], %[[VAL_24]] : index
! CHECK:         %[[VAL_26:.*]]:3 = fir.box_dims %[[VAL_21]], %[[VAL_25]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_27:.*]] = fir.convert %[[VAL_26]]#1 : (index) -> i64
! CHECK:         %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i64) -> i32
! CHECK:         fir.store %[[VAL_28]] to %[[VAL_6]] : !fir.ref<i32>
  iSize = size(a(2:5, -1:1), dim, 8)
end subroutine size_test
