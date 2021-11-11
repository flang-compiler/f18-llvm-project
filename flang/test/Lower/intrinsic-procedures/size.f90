! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPsize_test() {
! CHECK:         %[[VAL_0:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_1:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_2:.*]] = arith.constant -10 : index
! CHECK:         %[[VAL_3:.*]] = arith.constant 21 : index
! CHECK:         %[[VAL_4:.*]] = fir.alloca !fir.array<10x21xf32> {bindc_name = "a", uniq_name = "_QFsize_testEa"}
! CHECK:         %[[VAL_5:.*]] = fir.address_of(@_QFsize_testEdim) : !fir.ref<i32>
! CHECK:         %[[VAL_6:.*]] = fir.alloca i32 {bindc_name = "isize", uniq_name = "_QFsize_testEisize"}
! CHECK:         %[[VAL_7:.*]] = arith.constant 4 : i64
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i64) -> index
! CHECK:         %[[VAL_9:.*]] = arith.constant 3 : i64
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
! CHECK:         %[[VAL_11:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
! CHECK:         %[[VAL_13:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
! CHECK:         %[[VAL_15:.*]] = arith.constant 5 : i64
! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i64) -> index
! CHECK:         %[[VAL_17:.*]] = arith.constant -1 : i64
! CHECK:         %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i64) -> index
! CHECK:         %[[VAL_19:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i64) -> index
! CHECK:         %[[VAL_21:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i64) -> index
! CHECK:         %[[VAL_23:.*]] = fir.shape_shift %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:         %[[VAL_24:.*]] = fir.slice %[[VAL_12]], %[[VAL_16]], %[[VAL_14]], %[[VAL_18]], %[[VAL_22]], %[[VAL_20]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:         %[[VAL_25:.*]] = fir.array_load %[[VAL_4]](%[[VAL_23]]) {{\[}}%[[VAL_24]]] : (!fir.ref<!fir.array<10x21xf32>>, !fir.shapeshift<2>, !fir.slice<2>) -> !fir.array<10x21xf32>
! CHECK:         %[[VAL_26:.*]] = fir.allocmem !fir.array<4x3xf32>
! CHECK:         %[[VAL_27:.*]] = fir.shape %[[VAL_8]], %[[VAL_10]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_28:.*]] = fir.array_load %[[VAL_26]](%[[VAL_27]]) : (!fir.heap<!fir.array<4x3xf32>>, !fir.shape<2>) -> !fir.array<4x3xf32>
! CHECK:         %[[VAL_29:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_30:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_31:.*]] = arith.subi %[[VAL_8]], %[[VAL_29]] : index
! CHECK:         %[[VAL_32:.*]] = arith.subi %[[VAL_10]], %[[VAL_29]] : index
! CHECK:         %[[VAL_33:.*]] = fir.do_loop %[[VAL_34:.*]] = %[[VAL_30]] to %[[VAL_32]] step %[[VAL_29]] unordered iter_args(%[[VAL_35:.*]] = %[[VAL_28]]) -> (!fir.array<4x3xf32>) {
! CHECK:           %[[VAL_36:.*]] = fir.do_loop %[[VAL_37:.*]] = %[[VAL_30]] to %[[VAL_31]] step %[[VAL_29]] unordered iter_args(%[[VAL_38:.*]] = %[[VAL_35]]) -> (!fir.array<4x3xf32>) {
! CHECK:             %[[VAL_39:.*]] = fir.array_fetch %[[VAL_25]], %[[VAL_37]], %[[VAL_34]] : (!fir.array<10x21xf32>, index, index) -> f32
! CHECK:             %[[VAL_40:.*]] = fir.array_update %[[VAL_38]], %[[VAL_39]], %[[VAL_37]], %[[VAL_34]] : (!fir.array<4x3xf32>, f32, index, index) -> !fir.array<4x3xf32>
! CHECK:             fir.result %[[VAL_40]] : !fir.array<4x3xf32>
! CHECK:           }
! CHECK:           fir.result %[[VAL_41:.*]] : !fir.array<4x3xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_28]], %[[VAL_42:.*]] to %[[VAL_26]] : !fir.array<4x3xf32>, !fir.array<4x3xf32>, !fir.heap<!fir.array<4x3xf32>>
! CHECK:         %[[VAL_43:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
! CHECK:         %[[VAL_44:.*]] = fir.shape %[[VAL_8]], %[[VAL_10]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_45:.*]] = fir.embox %[[VAL_26]](%[[VAL_44]]) : (!fir.heap<!fir.array<4x3xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<4x3xf32>>
! CHECK:         %[[VAL_46:.*]] = fir.convert %[[VAL_43]] : (i32) -> index
! CHECK:         %[[VAL_47:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_48:.*]] = arith.subi %[[VAL_46]], %[[VAL_47]] : index
! CHECK:         %[[VAL_49:.*]]:3 = fir.box_dims %[[VAL_45]], %[[VAL_48]] : (!fir.box<!fir.array<4x3xf32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_50:.*]] = fir.convert %[[VAL_49]]#1 : (index) -> i64
! CHECK:         %[[VAL_51:.*]] = fir.convert %[[VAL_50]] : (i64) -> i32
! CHECK:         fir.store %[[VAL_51]] to %[[VAL_6]] : !fir.ref<i32>
! CHECK:         fir.freemem %[[VAL_26]] : !fir.heap<!fir.array<4x3xf32>>
! CHECK:         return
! CHECK:       }

subroutine size_test()
  real, dimension(1:10, -10:10) :: a
  integer :: dim = 1
  integer :: iSize
  iSize = size(a(2:5, -1:1), dim, 8)
end subroutine size_test
