! Test lower of elemental user defined assignments
! RUN: bbc -emit-fir %s -o - | FileCheck %s

module defined_assignments
  type t
    integer :: i
  end type
  interface assignment(=)
    elemental subroutine assign_t(a,b)
      import t
      type(t),intent(out) :: a
      type(t),intent(in) :: b
    end
  end interface
  interface assignment(=)
    elemental subroutine assign_logical_to_real(a,b)
      real, intent(out) :: a
      logical, intent(in) :: b
    end
  end interface
  interface assignment(=)
    elemental subroutine assign_real_to_logical(a,b)
      logical, intent(out) :: a
      real, intent(in) :: b
    end
  end interface
end module

! CHECK-LABEL: func @_QPtest_derived(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>>) {
subroutine test_derived(x)
  use defined_assignments
  type(t) :: x(100)
  x = x(100:1:-1)
! CHECK:         %[[VAL_1:.*]] = constant 100 : index
! CHECK:         %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_3:.*]] = fir.array_load %[[VAL_0]](%[[VAL_2]]) : (!fir.ref<!fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>>, !fir.shape<1>) -> !fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>
! CHECK:         %[[VAL_4:.*]] = constant 100 : i64
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
! CHECK:         %[[VAL_6:.*]] = constant 100 : i64
! CHECK:         %[[VAL_7:.*]] = constant 1 : i64
! CHECK:         %[[VAL_8:.*]] = constant -1 : i64
! CHECK:         %[[VAL_9:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_10:.*]] = fir.slice %[[VAL_6]], %[[VAL_7]], %[[VAL_8]] : (i64, i64, i64) -> !fir.slice<1>
! CHECK:         %[[VAL_11:.*]] = fir.array_load %[[VAL_0]](%[[VAL_9]]) {{\[}}%[[VAL_10]]] : (!fir.ref<!fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>
! CHECK:         %[[VAL_12:.*]] = constant 1 : index
! CHECK:         %[[VAL_13:.*]] = constant 0 : index
! CHECK:         %[[VAL_14:.*]] = subi %[[VAL_5]], %[[VAL_12]] : index
! CHECK:         %[[VAL_15:.*]] = fir.do_loop %[[VAL_16:.*]] = %[[VAL_13]] to %[[VAL_14]] step %[[VAL_12]] unordered iter_args(%[[VAL_17:.*]] = %[[VAL_3]]) -> (!fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>) {
! CHECK:           %[[VAL_18:.*]] = fir.array_fetch %[[VAL_11]], %[[VAL_16]] : (!fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>, index) -> !fir.ref<!fir.type<_QMdefined_assignmentsTt{i:i32}>>
! CHECK:           %[[VAL_19:.*]]:2 = fir.array_modify %[[VAL_17]], %[[VAL_16]] : (!fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>, index) -> (!fir.ref<!fir.type<_QMdefined_assignmentsTt{i:i32}>>, !fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>)
! CHECK:           fir.call @_QPassign_t(%[[VAL_19]]#0, %[[VAL_18]]) : (!fir.ref<!fir.type<_QMdefined_assignmentsTt{i:i32}>>, !fir.ref<!fir.type<_QMdefined_assignmentsTt{i:i32}>>) -> ()
! CHECK:           fir.result %[[VAL_19]]#1 : !fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_3]], %[[VAL_20:.*]] to %[[VAL_0]] : !fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>, !fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>, !fir.ref<!fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>>
! CHECK:         return
! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QPtest_intrinsic(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.ref<!fir.array<100xf32>>) {
subroutine test_intrinsic(x)
  use defined_assignments
  real :: x(100)
  x = x(100:1:-1) .lt. 0.
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.logical<4>
! CHECK:         %[[VAL_2:.*]] = constant 100 : index
! CHECK:         %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_4:.*]] = fir.array_load %[[VAL_0]](%[[VAL_3]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
! CHECK:         %[[VAL_5:.*]] = constant 100 : i64
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
! CHECK:         %[[VAL_7:.*]] = constant 100 : i64
! CHECK:         %[[VAL_8:.*]] = constant 1 : i64
! CHECK:         %[[VAL_9:.*]] = constant -1 : i64
! CHECK:         %[[VAL_10:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_11:.*]] = fir.slice %[[VAL_7]], %[[VAL_8]], %[[VAL_9]] : (i64, i64, i64) -> !fir.slice<1>
! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_0]](%[[VAL_10]]) {{\[}}%[[VAL_11]]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<100xf32>
! CHECK:         %[[VAL_13:.*]] = constant 0.000000e+00 : f32
! CHECK:         %[[VAL_14:.*]] = constant 1 : index
! CHECK:         %[[VAL_15:.*]] = constant 0 : index
! CHECK:         %[[VAL_16:.*]] = subi %[[VAL_6]], %[[VAL_14]] : index
! CHECK:         %[[VAL_17:.*]] = fir.do_loop %[[VAL_18:.*]] = %[[VAL_15]] to %[[VAL_16]] step %[[VAL_14]] unordered iter_args(%[[VAL_19:.*]] = %[[VAL_4]]) -> (!fir.array<100xf32>) {
! CHECK:           %[[VAL_20:.*]] = fir.array_fetch %[[VAL_12]], %[[VAL_18]] : (!fir.array<100xf32>, index) -> f32
! CHECK:           %[[VAL_21:.*]] = cmpf olt, %[[VAL_20]], %[[VAL_13]] : f32
! CHECK:           %[[VAL_22:.*]]:2 = fir.array_modify %[[VAL_19]], %[[VAL_18]] : (!fir.array<100xf32>, index) -> (!fir.ref<f32>, !fir.array<100xf32>)
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_21]] : (i1) -> !fir.logical<4>
! CHECK:           fir.store %[[VAL_23]] to %[[VAL_1]] : !fir.ref<!fir.logical<4>>
! CHECK:           fir.call @_QPassign_logical_to_real(%[[VAL_22]]#0, %[[VAL_1]]) : (!fir.ref<f32>, !fir.ref<!fir.logical<4>>) -> ()
! CHECK:           fir.result %[[VAL_22]]#1 : !fir.array<100xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_4]], %[[VAL_24:.*]] to %[[VAL_0]] : !fir.array<100xf32>, !fir.array<100xf32>, !fir.ref<!fir.array<100xf32>>
! CHECK:         return
! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QPtest_intrinsic_2(
! CHECK-SAME:                            %[[VAL_0:.*]]: !fir.ref<!fir.array<100x!fir.logical<4>>>,
subroutine test_intrinsic_2(x, y)
  use defined_assignments
  logical :: x(100)
  real :: y(100)
  x = y
! CHECK-SAME:                            %[[VAL_1:.*]]: !fir.ref<!fir.array<100xf32>>) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca f32
! CHECK:         %[[VAL_3:.*]] = constant 100 : index
! CHECK:         %[[VAL_4:.*]] = constant 100 : index
! CHECK:         %[[VAL_5:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_6:.*]] = fir.array_load %[[VAL_0]](%[[VAL_5]]) : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<100x!fir.logical<4>>
! CHECK:         %[[VAL_7:.*]] = constant 100 : i64
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i64) -> index
! CHECK:         %[[VAL_9:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_10:.*]] = fir.array_load %[[VAL_1]](%[[VAL_9]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
! CHECK:         %[[VAL_11:.*]] = constant 1 : index
! CHECK:         %[[VAL_12:.*]] = constant 0 : index
! CHECK:         %[[VAL_13:.*]] = subi %[[VAL_8]], %[[VAL_11]] : index
! CHECK:         %[[VAL_14:.*]] = fir.do_loop %[[VAL_15:.*]] = %[[VAL_12]] to %[[VAL_13]] step %[[VAL_11]] unordered iter_args(%[[VAL_16:.*]] = %[[VAL_6]]) -> (!fir.array<100x!fir.logical<4>>) {
! CHECK:           %[[VAL_17:.*]] = fir.array_fetch %[[VAL_10]], %[[VAL_15]] : (!fir.array<100xf32>, index) -> f32
! CHECK:           %[[VAL_18:.*]]:2 = fir.array_modify %[[VAL_16]], %[[VAL_15]] : (!fir.array<100x!fir.logical<4>>, index) -> (!fir.ref<!fir.logical<4>>, !fir.array<100x!fir.logical<4>>)
! CHECK:           fir.store %[[VAL_17]] to %[[VAL_2]] : !fir.ref<f32>
! CHECK:           fir.call @_QPassign_real_to_logical(%[[VAL_18]]#0, %[[VAL_2]]) : (!fir.ref<!fir.logical<4>>, !fir.ref<f32>) -> ()
! CHECK:           fir.result %[[VAL_18]]#1 : !fir.array<100x!fir.logical<4>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_6]], %[[VAL_19:.*]] to %[[VAL_0]] : !fir.array<100x!fir.logical<4>>, !fir.array<100x!fir.logical<4>>, !fir.ref<!fir.array<100x!fir.logical<4>>>
! CHECK:         return
! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QPfrom_char(
! CHECK-SAME:                     %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>,
! CHECK-SAME:                     %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>) {
subroutine from_char(i, c)
  interface assignment(=)
    elemental subroutine sfrom_char(a,b)
      integer, intent(out) :: a
      character(*),intent(in) :: b
    end subroutine
  end interface
  integer :: i(:)
  character(*) :: c(:)
  i = c
! CHECK:         %[[VAL_2:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_3:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.array<?x!fir.char<1,?>>
! CHECK:         %[[VAL_4:.*]] = constant 0 : index
! CHECK:         %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_6:.*]] = constant 1 : index
! CHECK:         %[[VAL_7:.*]] = constant 0 : index
! CHECK:         %[[VAL_8:.*]] = subi %[[VAL_5]]#1, %[[VAL_6]] : index
! CHECK:         %[[VAL_9:.*]] = fir.do_loop %[[VAL_10:.*]] = %[[VAL_7]] to %[[VAL_8]] step %[[VAL_6]] unordered iter_args(%[[VAL_11:.*]] = %[[VAL_2]]) -> (!fir.array<?xi32>) {
! CHECK:           %[[VAL_12:.*]] = fir.array_fetch %[[VAL_3]], %[[VAL_10]] : (!fir.array<?x!fir.char<1,?>>, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_13:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_14:.*]]:2 = fir.array_modify %[[VAL_11]], %[[VAL_10]] : (!fir.array<?xi32>, index) -> (!fir.ref<i32>, !fir.array<?xi32>)
! CHECK:           %[[VAL_15:.*]] = fir.emboxchar %[[VAL_12]], %[[VAL_13]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:           fir.call @_QPsfrom_char(%[[VAL_14]]#0, %[[VAL_15]]) : (!fir.ref<i32>, !fir.boxchar<1>) -> ()
! CHECK:           fir.result %[[VAL_14]]#1 : !fir.array<?xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_2]], %[[VAL_16:.*]] to %[[VAL_0]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:         return
! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QPto_char(
! CHECK-SAME:                   %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>,
! CHECK-SAME:                   %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>) {
subroutine to_char(i, c)
  interface assignment(=)
    elemental subroutine sto_char(a,b)
      character(*), intent(out) :: a
      integer,intent(in) :: b
    end subroutine
  end interface
  integer :: i(:)
  character(*) :: c(:)
  c = i
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32
! CHECK:         %[[VAL_3:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.array<?x!fir.char<1,?>>
! CHECK:         %[[VAL_4:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_5:.*]] = constant 0 : index
! CHECK:         %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_5]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_7:.*]] = constant 1 : index
! CHECK:         %[[VAL_8:.*]] = constant 0 : index
! CHECK:         %[[VAL_9:.*]] = subi %[[VAL_6]]#1, %[[VAL_7]] : index
! CHECK:         %[[VAL_10:.*]] = fir.do_loop %[[VAL_11:.*]] = %[[VAL_8]] to %[[VAL_9]] step %[[VAL_7]] unordered iter_args(%[[VAL_12:.*]] = %[[VAL_3]]) -> (!fir.array<?x!fir.char<1,?>>) {
! CHECK:           %[[VAL_13:.*]] = fir.array_fetch %[[VAL_4]], %[[VAL_11]] : (!fir.array<?xi32>, index) -> i32
! CHECK:           %[[VAL_14:.*]]:2 = fir.array_modify %[[VAL_12]], %[[VAL_11]] : (!fir.array<?x!fir.char<1,?>>, index) -> (!fir.ref<!fir.char<1,?>>, !fir.array<?x!fir.char<1,?>>)
! CHECK:           %[[VAL_15:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_16:.*]] = fir.emboxchar %[[VAL_14]]#0, %[[VAL_15]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:           fir.store %[[VAL_13]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           fir.call @_QPsto_char(%[[VAL_16]], %[[VAL_2]]) : (!fir.boxchar<1>, !fir.ref<i32>) -> ()
! CHECK:           fir.result %[[VAL_14]]#1 : !fir.array<?x!fir.char<1,?>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_3]], %[[VAL_17:.*]] to %[[VAL_1]] : !fir.array<?x!fir.char<1,?>>, !fir.array<?x!fir.char<1,?>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:         return
! CHECK:       }
end subroutine
