! Test derived type assignment lowering inside array expression
! RUN: bbc %s -o - | FileCheck %s

module array_derived_assign
  type simple_copy
    integer :: i
    character(10) :: c(20)
    real, pointer :: p(:)
  end type
  type deep_copy
    integer :: i
    real, allocatable :: a(:)
  end type
contains

! Simple copies are implemented inline.
! CHECK-LABEL: func @_QMarray_derived_assignPtest_simple_copy(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>,
! CHECK-SAME: %[[VAL_1:.*]]: !fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>) {
subroutine test_simple_copy(t1, t2)
  ! CHECK-DAG:     %[[VAL_2:.*]] = arith.constant 10 : index
  ! CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 0 : index
  ! CHECK-DAG:     %[[VAL_4:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_5:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
  ! CHECK:         br ^bb1(%[[VAL_3]], %[[VAL_2]] : index, index)
  ! CHECK:       ^bb1(%[[VAL_6:.*]]: index, %[[VAL_7:.*]]: index):
  ! CHECK:         %[[VAL_8:.*]] = arith.cmpi sgt, %[[VAL_7]], %[[VAL_3]] : index
  ! CHECK:         cond_br %[[VAL_8]], ^bb2, ^bb3
  ! CHECK:       ^bb2:
  ! CHECK:         %[[VAL_9:.*]] = arith.addi %[[VAL_6]], %[[VAL_4]] : index
  ! CHECK:         %[[VAL_10:.*]] = fir.array_coor %[[VAL_1]](%[[VAL_5]]) %[[VAL_9]] : (!fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
  ! CHECK:         %[[VAL_11:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_5]]) %[[VAL_9]] : (!fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
  ! CHECK:         %[[VAL_12:.*]] = fir.load %[[VAL_10]] : !fir.ref<!fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
  ! CHECK:         fir.store %[[VAL_12]] to %[[VAL_11]] : !fir.ref<!fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
  ! CHECK:         %[[VAL_13:.*]] = arith.subi %[[VAL_7]], %[[VAL_4]] : index
  ! CHECK:         br ^bb1(%[[VAL_9]], %[[VAL_13]] : index, index)
  type(simple_copy) :: t1(10), t2(10)
  t1 = t2
  ! CHECK:         return
  ! CHECK:       }
end subroutine

! Types require more complex assignments are passed to the runtime
! CHECK-LABEL: func @_QMarray_derived_assignPtest_deep_copy(
! CHECK-SAME:                                               %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>,
! CHECK-SAME:                                               %[[VAL_1:.*]]: !fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>) {
subroutine test_deep_copy(t1, t2)
  ! CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 10 : index
  ! CHECK-DAG:     %[[VAL_4:.*]] = arith.constant 0 : index
  ! CHECK-DAG:     %[[VAL_5:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_6:.*]] = fir.alloca !fir.box<!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  ! CHECK:         %[[VAL_7:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
  ! CHECK:         br ^bb1(%[[VAL_4]], %[[VAL_3]] : index, index)
  ! CHECK:       ^bb1(%[[VAL_8:.*]]: index, %[[VAL_9:.*]]: index):
  ! CHECK:         %[[VAL_10:.*]] = arith.cmpi sgt, %[[VAL_9]], %[[VAL_4]] : index
  ! CHECK:         cond_br %[[VAL_10]], ^bb2, ^bb3
  ! CHECK:       ^bb2:
  ! CHECK:         %[[VAL_11:.*]] = arith.addi %[[VAL_8]], %[[VAL_5]] : index
  ! CHECK:         %[[VAL_12:.*]] = fir.array_coor %[[VAL_1]](%[[VAL_7]]) %[[VAL_11]] : (!fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  ! CHECK:         %[[VAL_13:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_7]]) %[[VAL_11]] : (!fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  ! CHECK:         %[[VAL_14:.*]] = fir.embox %[[VAL_13]] : (!fir.ref<!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.box<!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  ! CHECK:         %[[VAL_15:.*]] = fir.embox %[[VAL_12]] : (!fir.ref<!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.box<!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  ! CHECK:         fir.store %[[VAL_14]] to %[[VAL_6]] : !fir.ref<!fir.box<!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>
  ! CHECK:         %[[VAL_16:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,
  ! CHECK:         %[[VAL_17:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.box<!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK:         %[[VAL_18:.*]] = fir.convert %[[VAL_15]] : (!fir.box<!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.box<none>
  ! CHECK:         %[[VAL_19:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_20:.*]] = fir.call @_FortranAAssign(%[[VAL_17]], %[[VAL_18]], %[[VAL_19]], %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  ! CHECK:         %[[VAL_21:.*]] = arith.subi %[[VAL_9]], %[[VAL_5]] : index
  ! CHECK:         br ^bb1(%[[VAL_11]], %[[VAL_21]] : index, index)
  type(deep_copy) :: t1(10), t2(10)
  t1 = t2
  ! CHECK:         return
  ! CHECK:       }
end subroutine
  
end module
