! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPsize_test() {
subroutine size_test()
  real, dimension(1:10, -10:10) :: a
  integer :: dim = 1
  integer :: iSize
! CHECK:         %[[VAL_c1:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_c10:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_cneg10:.*]] = arith.constant -10 : index
! CHECK:         %[[VAL_c21:.*]] = arith.constant 21 : index
! CHECK:         %[[VAL_0:.*]] = fir.alloca !fir.array<10x21xf32> {bindc_name = "a", uniq_name = "_QFsize_testEa"}
! CHECK:         %[[VAL_1:.*]] = fir.address_of(@_QFsize_testEdim) : !fir.ref<i32>
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "isize", uniq_name = "_QFsize_testEisize"}
! CHECK:         %[[VAL_c2_i64:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_c2_i64]] : (i64) -> index
! CHECK:         %[[VAL_c1_i64:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_c1_i64]] : (i64) -> index
! CHECK:         %[[VAL_c5_i64:.*]] = arith.constant 5 : i64
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_c5_i64]] : (i64) -> index
! CHECK:         %[[VAL_neg1_i64:.*]] = arith.constant -1 : i64
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_neg1_i64]] : (i64) -> index
! CHECK:         %[[VAL_c1_i64_0:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_c1_i64_0]] : (i64) -> index
! CHECK:         %[[VAL_c1_i64_1:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_c1_i64_1]] : (i64) -> index
! CHECK:         %[[VAL_9:.*]] = fir.shape_shift %[[VAL_c1]], %[[VAL_c10]], %[[VAL_cneg10]], %[[VAL_c21]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:         %[[VAL_10:.*]] = fir.slice %[[VAL_3]], %[[VAL_5]], %[[VAL_4]], %[[VAL_6]], %[[VAL_8]], %[[VAL_7]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:         %[[VAL_11:.*]] = fir.embox %[[VAL_0]](%[[VAL_9]]) [%[[VAL_10]]] : (!fir.ref<!fir.array<10x21xf32>>, !fir.shapeshift<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xf32>>
! CHECK:         %[[VAL_12:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:         %[[VAL_13:.*]] = fir.address_of({{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:         %[[VAL_c38_i32:.*]] = arith.constant 38 : i32
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_11]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_16:.*]] = fir.call @_FortranASizeDim(%[[VAL_14]], %[[VAL_12]], %[[VAL_15]], %[[VAL_c38_i32]]) : (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK:         %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i64) -> i32
! CHECK:         fir.store %[[VAL_17]] to %[[VAL_2]] : !fir.ref<i32>
  iSize = size(a(2:5, -1:1), dim, 8)
end subroutine size_test
