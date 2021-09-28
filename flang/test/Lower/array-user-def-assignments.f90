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

! -----------------------------------------------------------------------------
!     Test user defined assignments inside FORALL and WHERE
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPtest_in_forall_1(
! CHECK-SAME:                            %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>>,
! CHECK-SAME:                            %[[VAL_1:.*]]: !fir.ref<!fir.array<10xf32>>) {
subroutine test_in_forall_1(x, y)
  use defined_assignments
  logical :: x(10)
  real :: y(10)
  forall (i=1:10) x(i) = y(i)
! CHECK:         %[[VAL_2:.*]] = fir.alloca f32
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_4:.*]] = constant 10 : index
! CHECK:         %[[VAL_5:.*]] = constant 10 : index
! CHECK:         %[[VAL_6:.*]] = constant 1 : i32
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:         %[[VAL_8:.*]] = constant 10 : i32
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
! CHECK:         %[[VAL_10:.*]] = constant 1 : index
! CHECK:         %[[VAL_11:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_0]](%[[VAL_11]]) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
! CHECK:         %[[VAL_13:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_14:.*]] = fir.array_load %[[VAL_1]](%[[VAL_13]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
! CHECK:         %[[VAL_15:.*]] = fir.do_loop %[[VAL_16:.*]] = %[[VAL_7]] to %[[VAL_9]] step %[[VAL_10]] unordered iter_args(%[[VAL_17:.*]] = %[[VAL_12]]) -> (!fir.array<10x!fir.logical<4>>) {
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_16]] : (index) -> i32
! CHECK:           fir.store %[[VAL_18]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_19:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i32) -> i64
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i64) -> index
! CHECK:           %[[VAL_22:.*]] = fir.array_fetch %[[VAL_14]], %[[VAL_21]] {Fortran.offsets} : (!fir.array<10xf32>, index) -> f32
! CHECK:           %[[VAL_23:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i32) -> i64
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i64) -> index
! CHECK:           %[[VAL_26:.*]]:2 = fir.array_modify %[[VAL_17]], %[[VAL_25]] {Fortran.offsets} : (!fir.array<10x!fir.logical<4>>, index) -> (!fir.ref<!fir.logical<4>>, !fir.array<10x!fir.logical<4>>)
! CHECK:           fir.store %[[VAL_22]] to %[[VAL_2]] : !fir.ref<f32>
! CHECK:           fir.call @_QPassign_real_to_logical(%[[VAL_26]]#0, %[[VAL_2]]) : (!fir.ref<!fir.logical<4>>, !fir.ref<f32>) -> ()
! CHECK:           fir.result %[[VAL_26]]#1 : !fir.array<10x!fir.logical<4>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_12]], %[[VAL_27:.*]] to %[[VAL_0]] : !fir.array<10x!fir.logical<4>>, !fir.array<10x!fir.logical<4>>, !fir.ref<!fir.array<10x!fir.logical<4>>>
! CHECK:         return
! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QPtest_in_forall_2(
! CHECK-SAME:                            %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>>,
! CHECK-SAME:                            %[[VAL_1:.*]]: !fir.ref<!fir.array<10xf32>>) {
subroutine test_in_forall_2(x, y)
  use defined_assignments
  logical :: x(10)
  real :: y(10)
  forall (i=1:10) y(i) = y(i).lt.0.
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.logical<4>
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_4:.*]] = constant 10 : index
! CHECK:         %[[VAL_5:.*]] = constant 1 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
! CHECK:         %[[VAL_7:.*]] = constant 10 : i32
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
! CHECK:         %[[VAL_9:.*]] = constant 1 : index
! CHECK:         %[[VAL_10:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_11:.*]] = fir.array_load %[[VAL_1]](%[[VAL_10]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
! CHECK:         %[[VAL_12:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_13:.*]] = fir.array_load %[[VAL_1]](%[[VAL_12]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
! CHECK:         %[[VAL_14:.*]] = fir.do_loop %[[VAL_15:.*]] = %[[VAL_6]] to %[[VAL_8]] step %[[VAL_9]] unordered iter_args(%[[VAL_16:.*]] = %[[VAL_11]]) -> (!fir.array<10xf32>) {
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_15]] : (index) -> i32
! CHECK:           fir.store %[[VAL_17]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> i64
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i64) -> index
! CHECK:           %[[VAL_21:.*]] = fir.array_fetch %[[VAL_13]], %[[VAL_20]] {Fortran.offsets} : (!fir.array<10xf32>, index) -> f32
! CHECK:           %[[VAL_22:.*]] = constant 0.000000e+00 : f32
! CHECK:           %[[VAL_23:.*]] = cmpf olt, %[[VAL_21]], %[[VAL_22]] : f32
! CHECK:           %[[VAL_24:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i32) -> i64
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i64) -> index
! CHECK:           %[[VAL_27:.*]]:2 = fir.array_modify %[[VAL_16]], %[[VAL_26]] {Fortran.offsets} : (!fir.array<10xf32>, index) -> (!fir.ref<f32>, !fir.array<10xf32>)
! CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_23]] : (i1) -> !fir.logical<4>
! CHECK:           fir.store %[[VAL_28]] to %[[VAL_2]] : !fir.ref<!fir.logical<4>>
! CHECK:           fir.call @_QPassign_logical_to_real(%[[VAL_27]]#0, %[[VAL_2]]) : (!fir.ref<f32>, !fir.ref<!fir.logical<4>>) -> ()
! CHECK:           fir.result %[[VAL_27]]#1 : !fir.array<10xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_11]], %[[VAL_29:.*]] to %[[VAL_1]] : !fir.array<10xf32>, !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>
! CHECK:         return
! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QPtest_intrinsic_where_1(
! CHECK-SAME:                                  %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>>,
! CHECK-SAME:                                  %[[VAL_1:.*]]: !fir.ref<!fir.array<10xf32>>,
! CHECK-SAME:                                  %[[VAL_2:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>>) {
subroutine test_intrinsic_where_1(x, y, l)
  use defined_assignments
  logical :: x(10), l(10)
  real :: y(10)
  where(l) x = y
! CHECK:         %[[VAL_3:.*]] = fir.alloca f32
! CHECK:         %[[VAL_4:.*]] = constant 10 : index
! CHECK:         %[[VAL_5:.*]] = constant 10 : index
! CHECK:         %[[VAL_6:.*]] = constant 10 : index
! CHECK:         %[[VAL_7:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_8:.*]] = fir.array_load %[[VAL_0]](%[[VAL_7]]) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
! CHECK:         %[[VAL_9:.*]] = constant 10 : i64
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
! CHECK:         %[[VAL_11:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_1]](%[[VAL_11]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
! CHECK:         %[[VAL_13:.*]] = constant 10 : i64
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
! CHECK:         %[[VAL_15:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_16:.*]] = fir.array_load %[[VAL_2]](%[[VAL_15]]) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
! CHECK:         %[[VAL_17:.*]] = fir.allocmem !fir.array<10x!fir.logical<4>>
! CHECK:         %[[VAL_18:.*]] = fir.shape %[[VAL_14]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_19:.*]] = fir.array_load %[[VAL_17]](%[[VAL_18]]) : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
! CHECK:         %[[VAL_20:.*]] = constant 1 : index
! CHECK:         %[[VAL_21:.*]] = constant 0 : index
! CHECK:         %[[VAL_22:.*]] = subi %[[VAL_14]], %[[VAL_20]] : index
! CHECK:         %[[VAL_23:.*]] = fir.do_loop %[[VAL_24:.*]] = %[[VAL_21]] to %[[VAL_22]] step %[[VAL_20]] unordered iter_args(%[[VAL_25:.*]] = %[[VAL_19]]) -> (!fir.array<10x!fir.logical<4>>) {
! CHECK:           %[[VAL_26:.*]] = fir.array_fetch %[[VAL_16]], %[[VAL_24]] : (!fir.array<10x!fir.logical<4>>, index) -> !fir.logical<4>
! CHECK:           %[[VAL_27:.*]] = fir.array_update %[[VAL_25]], %[[VAL_26]], %[[VAL_24]] : (!fir.array<10x!fir.logical<4>>, !fir.logical<4>, index) -> !fir.array<10x!fir.logical<4>>
! CHECK:           fir.result %[[VAL_27]] : !fir.array<10x!fir.logical<4>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_19]], %[[VAL_28:.*]] to %[[VAL_17]] : !fir.array<10x!fir.logical<4>>, !fir.array<10x!fir.logical<4>>, !fir.heap<!fir.array<10x!fir.logical<4>>>
! CHECK:         %[[VAL_29:.*]] = fir.shape %[[VAL_14]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_30:.*]] = constant 1 : index
! CHECK:         %[[VAL_31:.*]] = constant 0 : index
! CHECK:         %[[VAL_32:.*]] = subi %[[VAL_10]], %[[VAL_30]] : index
! CHECK:         %[[VAL_33:.*]] = fir.do_loop %[[VAL_34:.*]] = %[[VAL_31]] to %[[VAL_32]] step %[[VAL_30]] unordered iter_args(%[[VAL_35:.*]] = %[[VAL_8]]) -> (!fir.array<10x!fir.logical<4>>) {
! CHECK:           %[[VAL_36:.*]] = constant 1 : index
! CHECK:           %[[VAL_37:.*]] = addi %[[VAL_34]], %[[VAL_36]] : index
! CHECK:           %[[VAL_38:.*]] = fir.array_coor %[[VAL_17]](%[[VAL_29]]) %[[VAL_37]] : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_39:.*]] = fir.load %[[VAL_38]] : !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (!fir.logical<4>) -> i1
! CHECK:           %[[VAL_41:.*]] = fir.if %[[VAL_40]] -> (!fir.array<10x!fir.logical<4>>) {
! CHECK:             %[[VAL_42:.*]] = fir.array_fetch %[[VAL_12]], %[[VAL_34]] : (!fir.array<10xf32>, index) -> f32
! CHECK:             %[[VAL_43:.*]]:2 = fir.array_modify %[[VAL_35]], %[[VAL_34]] : (!fir.array<10x!fir.logical<4>>, index) -> (!fir.ref<!fir.logical<4>>, !fir.array<10x!fir.logical<4>>)
! CHECK:             fir.store %[[VAL_42]] to %[[VAL_3]] : !fir.ref<f32>
! CHECK:             fir.call @_QPassign_real_to_logical(%[[VAL_43]]#0, %[[VAL_3]]) : (!fir.ref<!fir.logical<4>>, !fir.ref<f32>) -> ()
! CHECK:             fir.result %[[VAL_43]]#1 : !fir.array<10x!fir.logical<4>>
! CHECK:           } else {
! CHECK:             fir.result %[[VAL_35]] : !fir.array<10x!fir.logical<4>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_44:.*]] : !fir.array<10x!fir.logical<4>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_8]], %[[VAL_45:.*]] to %[[VAL_0]] : !fir.array<10x!fir.logical<4>>, !fir.array<10x!fir.logical<4>>, !fir.ref<!fir.array<10x!fir.logical<4>>>
! CHECK:         fir.freemem %[[VAL_17]] : !fir.heap<!fir.array<10x!fir.logical<4>>>
! CHECK:         return
! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QPtest_intrinsic_where_2(
! CHECK-SAME:                                  %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>>,
! CHECK-SAME:                                  %[[VAL_1:.*]]: !fir.ref<!fir.array<10xf32>>,
! CHECK-SAME:                                  %[[VAL_2:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>>) {
subroutine test_intrinsic_where_2(x, y, l)
  use defined_assignments
  logical :: x(10), l(10)
  real :: y(10)
  where(l) y = y.lt.0.
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.logical<4>
! CHECK:         %[[VAL_4:.*]] = constant 10 : index
! CHECK:         %[[VAL_5:.*]] = constant 10 : index
! CHECK:         %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_7:.*]] = fir.array_load %[[VAL_1]](%[[VAL_6]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
! CHECK:         %[[VAL_8:.*]] = constant 10 : i64
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
! CHECK:         %[[VAL_10:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_11:.*]] = fir.array_load %[[VAL_1]](%[[VAL_10]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
! CHECK:         %[[VAL_12:.*]] = constant 0.000000e+00 : f32
! CHECK:         %[[VAL_13:.*]] = constant 10 : i64
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
! CHECK:         %[[VAL_15:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_16:.*]] = fir.array_load %[[VAL_2]](%[[VAL_15]]) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
! CHECK:         %[[VAL_17:.*]] = fir.allocmem !fir.array<10x!fir.logical<4>>
! CHECK:         %[[VAL_18:.*]] = fir.shape %[[VAL_14]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_19:.*]] = fir.array_load %[[VAL_17]](%[[VAL_18]]) : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
! CHECK:         %[[VAL_20:.*]] = constant 1 : index
! CHECK:         %[[VAL_21:.*]] = constant 0 : index
! CHECK:         %[[VAL_22:.*]] = subi %[[VAL_14]], %[[VAL_20]] : index
! CHECK:         %[[VAL_23:.*]] = fir.do_loop %[[VAL_24:.*]] = %[[VAL_21]] to %[[VAL_22]] step %[[VAL_20]] unordered iter_args(%[[VAL_25:.*]] = %[[VAL_19]]) -> (!fir.array<10x!fir.logical<4>>) {
! CHECK:           %[[VAL_26:.*]] = fir.array_fetch %[[VAL_16]], %[[VAL_24]] : (!fir.array<10x!fir.logical<4>>, index) -> !fir.logical<4>
! CHECK:           %[[VAL_27:.*]] = fir.array_update %[[VAL_25]], %[[VAL_26]], %[[VAL_24]] : (!fir.array<10x!fir.logical<4>>, !fir.logical<4>, index) -> !fir.array<10x!fir.logical<4>>
! CHECK:           fir.result %[[VAL_27]] : !fir.array<10x!fir.logical<4>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_19]], %[[VAL_28:.*]] to %[[VAL_17]] : !fir.array<10x!fir.logical<4>>, !fir.array<10x!fir.logical<4>>, !fir.heap<!fir.array<10x!fir.logical<4>>>
! CHECK:         %[[VAL_29:.*]] = fir.shape %[[VAL_14]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_30:.*]] = constant 1 : index
! CHECK:         %[[VAL_31:.*]] = constant 0 : index
! CHECK:         %[[VAL_32:.*]] = subi %[[VAL_9]], %[[VAL_30]] : index
! CHECK:         %[[VAL_33:.*]] = fir.do_loop %[[VAL_34:.*]] = %[[VAL_31]] to %[[VAL_32]] step %[[VAL_30]] unordered iter_args(%[[VAL_35:.*]] = %[[VAL_7]]) -> (!fir.array<10xf32>) {
! CHECK:           %[[VAL_36:.*]] = constant 1 : index
! CHECK:           %[[VAL_37:.*]] = addi %[[VAL_34]], %[[VAL_36]] : index
! CHECK:           %[[VAL_38:.*]] = fir.array_coor %[[VAL_17]](%[[VAL_29]]) %[[VAL_37]] : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_39:.*]] = fir.load %[[VAL_38]] : !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (!fir.logical<4>) -> i1
! CHECK:           %[[VAL_41:.*]] = fir.if %[[VAL_40]] -> (!fir.array<10xf32>) {
! CHECK:             %[[VAL_42:.*]] = fir.array_fetch %[[VAL_11]], %[[VAL_34]] : (!fir.array<10xf32>, index) -> f32
! CHECK:             %[[VAL_43:.*]] = cmpf olt, %[[VAL_42]], %[[VAL_12]] : f32
! CHECK:             %[[VAL_44:.*]]:2 = fir.array_modify %[[VAL_35]], %[[VAL_34]] : (!fir.array<10xf32>, index) -> (!fir.ref<f32>, !fir.array<10xf32>)
! CHECK:             %[[VAL_45:.*]] = fir.convert %[[VAL_43]] : (i1) -> !fir.logical<4>
! CHECK:             fir.store %[[VAL_45]] to %[[VAL_3]] : !fir.ref<!fir.logical<4>>
! CHECK:             fir.call @_QPassign_logical_to_real(%[[VAL_44]]#0, %[[VAL_3]]) : (!fir.ref<f32>, !fir.ref<!fir.logical<4>>) -> ()
! CHECK:             fir.result %[[VAL_44]]#1 : !fir.array<10xf32>
! CHECK:           } else {
! CHECK:             fir.result %[[VAL_35]] : !fir.array<10xf32>
! CHECK:           }
! CHECK:           fir.result %[[VAL_46:.*]] : !fir.array<10xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_7]], %[[VAL_47:.*]] to %[[VAL_1]] : !fir.array<10xf32>, !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>
! CHECK:         fir.freemem %[[VAL_17]] : !fir.heap<!fir.array<10x!fir.logical<4>>>
! CHECK:         return
! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QPtest_scalar_func_but_not_elemental(
! CHECK-SAME:                                              %[[VAL_0:.*]]: !fir.ref<!fir.array<100x!fir.logical<4>>>,
! CHECK-SAME:                                              %[[VAL_1:.*]]: !fir.ref<!fir.array<100xi32>>) {
subroutine test_scalar_func_but_not_elemental(x, y)
  interface assignment(=)
    ! scalar, but not elemental
    elemental subroutine assign_integer_to_logical(a,b)
      logical, intent(out) :: a
      integer, intent(in) :: b
    end
  end interface
  logical :: x(100)
  integer :: y(100)
  ! Scalar assignment in forall should be treated just like elemental
  ! functions.
  forall(i=1:10) x(i) = y(i)
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_4:.*]] = constant 100 : index
! CHECK:         %[[VAL_5:.*]] = constant 100 : index
! CHECK:         %[[VAL_6:.*]] = constant 1 : i32
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:         %[[VAL_8:.*]] = constant 10 : i32
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
! CHECK:         %[[VAL_10:.*]] = constant 1 : index
! CHECK:         %[[VAL_11:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_0]](%[[VAL_11]]) : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<100x!fir.logical<4>>
! CHECK:         %[[VAL_13:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_14:.*]] = fir.array_load %[[VAL_1]](%[[VAL_13]]) : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.array<100xi32>
! CHECK:         %[[VAL_15:.*]] = fir.do_loop %[[VAL_16:.*]] = %[[VAL_7]] to %[[VAL_9]] step %[[VAL_10]] unordered iter_args(%[[VAL_17:.*]] = %[[VAL_12]]) -> (!fir.array<100x!fir.logical<4>>) {
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_16]] : (index) -> i32
! CHECK:           fir.store %[[VAL_18]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_19:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i32) -> i64
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i64) -> index
! CHECK:           %[[VAL_22:.*]] = fir.array_fetch %[[VAL_14]], %[[VAL_21]] {Fortran.offsets} : (!fir.array<100xi32>, index) -> i32
! CHECK:           %[[VAL_23:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i32) -> i64
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i64) -> index
! CHECK:           %[[VAL_26:.*]]:2 = fir.array_modify %[[VAL_17]], %[[VAL_25]] {Fortran.offsets} : (!fir.array<100x!fir.logical<4>>, index) -> (!fir.ref<!fir.logical<4>>, !fir.array<100x!fir.logical<4>>)
! CHECK:           fir.store %[[VAL_22]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           fir.call @_QPassign_integer_to_logical(%[[VAL_26]]#0, %[[VAL_2]]) : (!fir.ref<!fir.logical<4>>, !fir.ref<i32>) -> ()
! CHECK:           fir.result %[[VAL_26]]#1 : !fir.array<100x!fir.logical<4>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_12]], %[[VAL_27:.*]] to %[[VAL_0]] : !fir.array<100x!fir.logical<4>>, !fir.array<100x!fir.logical<4>>, !fir.ref<!fir.array<100x!fir.logical<4>>>
! CHECK:         return
! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QPtest_in_forall_with_cleanup(
! CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>>,
! CHECK-SAME:                                       %[[VAL_1:.*]]: !fir.ref<!fir.array<10xf32>>) {
subroutine test_in_forall_with_cleanup(x, y)
  use defined_assignments
  interface
    pure function returns_alloc(i)
      integer, intent(in) :: i
      real, allocatable :: returns_alloc
    end function
  end interface
  logical :: x(10)
  real :: y(10)
  forall (i=1:10) x(i) = returns_alloc(i)
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = ".result"}
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_4:.*]] = constant 10 : index
! CHECK:         %[[VAL_5:.*]] = constant 1 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
! CHECK:         %[[VAL_7:.*]] = constant 10 : i32
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
! CHECK:         %[[VAL_9:.*]] = constant 1 : index
! CHECK:         %[[VAL_10:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_11:.*]] = fir.array_load %[[VAL_0]](%[[VAL_10]]) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
! CHECK:         %[[VAL_12:.*]] = fir.do_loop %[[VAL_13:.*]] = %[[VAL_6]] to %[[VAL_8]] step %[[VAL_9]] unordered iter_args(%[[VAL_14:.*]] = %[[VAL_11]]) -> (!fir.array<10x!fir.logical<4>>) {
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_13]] : (index) -> i32
! CHECK:           fir.store %[[VAL_15]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_16:.*]] = fir.call @_QPreturns_alloc(%[[VAL_3]]) : (!fir.ref<i32>) -> !fir.box<!fir.heap<f32>>
! CHECK:           fir.save_result %[[VAL_16]] to %[[VAL_2]] : !fir.box<!fir.heap<f32>>, !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:           %[[VAL_17:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:           %[[VAL_18:.*]] = fir.box_addr %[[VAL_17]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
! CHECK:           %[[VAL_19:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i32) -> i64
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i64) -> index
! CHECK:           %[[VAL_22:.*]]:2 = fir.array_modify %[[VAL_14]], %[[VAL_21]] {Fortran.offsets} : (!fir.array<10x!fir.logical<4>>, index) -> (!fir.ref<!fir.logical<4>>, !fir.array<10x!fir.logical<4>>)
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_18]] : (!fir.heap<f32>) -> !fir.ref<f32>
! CHECK:           fir.call @_QPassign_real_to_logical(%[[VAL_22]]#0, %[[VAL_23]]) : (!fir.ref<!fir.logical<4>>, !fir.ref<f32>) -> ()
! CHECK:           %[[VAL_24:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:           %[[VAL_25:.*]] = fir.box_addr %[[VAL_24]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (!fir.heap<f32>) -> i64
! CHECK:           %[[VAL_27:.*]] = constant 0 : i64
! CHECK:           %[[VAL_28:.*]] = cmpi ne, %[[VAL_26]], %[[VAL_27]] : i64
! CHECK:           fir.if %[[VAL_28]] {
! CHECK:             fir.freemem %[[VAL_25]] : !fir.heap<f32>
! CHECK:           }
! CHECK:           fir.result %[[VAL_22]]#1 : !fir.array<10x!fir.logical<4>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_11]], %[[VAL_29:.*]] to %[[VAL_0]] : !fir.array<10x!fir.logical<4>>, !fir.array<10x!fir.logical<4>>, !fir.ref<!fir.array<10x!fir.logical<4>>>
! CHECK:         return
! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QPtest_forall_array(
! CHECK-SAME:                             %[[VAL_0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>,
! CHECK-SAME:                             %[[VAL_1:.*]]: !fir.box<!fir.array<?x?xf32>>) {
subroutine test_forall_array(x, y)
  use defined_assignments
  logical :: x(:, :)
  real :: y(:, :)
  forall (i=1:10) x(i, :) = y(i, :)
! CHECK:         %[[VAL_2:.*]] = fir.alloca f32
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_4:.*]] = constant 1 : i32
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
! CHECK:         %[[VAL_6:.*]] = constant 10 : i32
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:         %[[VAL_8:.*]] = constant 1 : index
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.array<?x?x!fir.logical<4>>
! CHECK:         %[[VAL_10:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.array<?x?xf32>
! CHECK:         %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_8]] unordered iter_args(%[[VAL_13:.*]] = %[[VAL_9]]) -> (!fir.array<?x?x!fir.logical<4>>) {
! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (index) -> i32
! CHECK:           fir.store %[[VAL_14]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_15:.*]] = constant 1 : index
! CHECK:           %[[VAL_16:.*]] = constant 1 : index
! CHECK:           %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_16]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_18:.*]] = addi %[[VAL_15]], %[[VAL_17]]#1 : index
! CHECK:           %[[VAL_19:.*]] = subi %[[VAL_18]], %[[VAL_15]] : index
! CHECK:           %[[VAL_20:.*]] = constant 1 : i64
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i64) -> index
! CHECK:           %[[VAL_22:.*]] = constant 0 : index
! CHECK:           %[[VAL_23:.*]] = subi %[[VAL_19]], %[[VAL_15]] : index
! CHECK:           %[[VAL_24:.*]] = addi %[[VAL_23]], %[[VAL_21]] : index
! CHECK:           %[[VAL_25:.*]] = divi_signed %[[VAL_24]], %[[VAL_21]] : index
! CHECK:           %[[VAL_26:.*]] = cmpi sgt, %[[VAL_25]], %[[VAL_22]] : index
! CHECK:           %[[VAL_27:.*]] = select %[[VAL_26]], %[[VAL_25]], %[[VAL_22]] : index
! CHECK:           %[[VAL_28:.*]] = constant 1 : index
! CHECK:           %[[VAL_29:.*]] = constant 0 : index
! CHECK:           %[[VAL_30:.*]] = subi %[[VAL_27]], %[[VAL_28]] : index
! CHECK:           %[[VAL_31:.*]] = fir.do_loop %[[VAL_32:.*]] = %[[VAL_29]] to %[[VAL_30]] step %[[VAL_28]] unordered iter_args(%[[VAL_33:.*]] = %[[VAL_13]]) -> (!fir.array<?x?x!fir.logical<4>>) {
! CHECK:             %[[VAL_34:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:             %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (i32) -> i64
! CHECK:             %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (i64) -> index
! CHECK:             %[[VAL_37:.*]] = constant 1 : index
! CHECK:             %[[VAL_38:.*]] = constant 1 : i64
! CHECK:             %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (i64) -> index
! CHECK:             %[[VAL_40:.*]] = muli %[[VAL_32]], %[[VAL_39]] : index
! CHECK:             %[[VAL_41:.*]] = addi %[[VAL_37]], %[[VAL_40]] : index
! CHECK:             %[[VAL_42:.*]] = fir.array_fetch %[[VAL_10]], %[[VAL_36]], %[[VAL_41]] {Fortran.offsets} : (!fir.array<?x?xf32>, index, index) -> f32
! CHECK:             %[[VAL_43:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:             %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (i32) -> i64
! CHECK:             %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (i64) -> index
! CHECK:             %[[VAL_46:.*]] = constant 1 : index
! CHECK:             %[[VAL_47:.*]] = constant 1 : i64
! CHECK:             %[[VAL_48:.*]] = fir.convert %[[VAL_47]] : (i64) -> index
! CHECK:             %[[VAL_49:.*]] = muli %[[VAL_32]], %[[VAL_48]] : index
! CHECK:             %[[VAL_50:.*]] = addi %[[VAL_46]], %[[VAL_49]] : index
! CHECK:             %[[VAL_51:.*]]:2 = fir.array_modify %[[VAL_13]], %[[VAL_45]], %[[VAL_50]] {Fortran.offsets} : (!fir.array<?x?x!fir.logical<4>>, index, index) -> (!fir.ref<!fir.logical<4>>, !fir.array<?x?x!fir.logical<4>>)
! CHECK:             fir.store %[[VAL_42]] to %[[VAL_2]] : !fir.ref<f32>
! CHECK:             fir.call @_QPassign_real_to_logical(%[[VAL_51]]#0, %[[VAL_2]]) : (!fir.ref<!fir.logical<4>>, !fir.ref<f32>) -> ()
! CHECK:             fir.result %[[VAL_51]]#1 : !fir.array<?x?x!fir.logical<4>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_52:.*]] : !fir.array<?x?x!fir.logical<4>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_9]], %[[VAL_53:.*]] to %[[VAL_0]] : !fir.array<?x?x!fir.logical<4>>, !fir.array<?x?x!fir.logical<4>>, !fir.box<!fir.array<?x?x!fir.logical<4>>>
! CHECK:         return
! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QPfrom_char_forall(
! CHECK-SAME:                            %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>,
! CHECK-SAME:                            %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>) {
subroutine from_char_forall(i, c)
  interface assignment(=)
    elemental subroutine sfrom_char(a,b)
      integer, intent(out) :: a
      character(*),intent(in) :: b
    end subroutine
  end interface
  integer :: i(:)
  character(*) :: c(:)
  forall (j=1:10) i(j) = c(j)
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
! CHECK:         %[[VAL_3:.*]] = constant 1 : i32
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i32) -> index
! CHECK:         %[[VAL_5:.*]] = constant 10 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
! CHECK:         %[[VAL_7:.*]] = constant 1 : index
! CHECK:         %[[VAL_8:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.array<?x!fir.char<1,?>>
! CHECK:         %[[VAL_10:.*]] = fir.do_loop %[[VAL_11:.*]] = %[[VAL_4]] to %[[VAL_6]] step %[[VAL_7]] unordered iter_args(%[[VAL_12:.*]] = %[[VAL_8]]) -> (!fir.array<?xi32>) {
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_11]] : (index) -> i32
! CHECK:           fir.store %[[VAL_13]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_14:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i32) -> i64
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i64) -> index
! CHECK:           %[[VAL_17:.*]] = fir.array_fetch %[[VAL_9]], %[[VAL_16]] {Fortran.offsets} : (!fir.array<?x!fir.char<1,?>>, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_18:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_19:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i32) -> i64
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i64) -> index
! CHECK:           %[[VAL_22:.*]]:2 = fir.array_modify %[[VAL_12]], %[[VAL_21]] {Fortran.offsets} : (!fir.array<?xi32>, index) -> (!fir.ref<i32>, !fir.array<?xi32>)
! CHECK:           %[[VAL_23:.*]] = fir.emboxchar %[[VAL_17]], %[[VAL_18]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:           fir.call @_QPsfrom_char(%[[VAL_22]]#0, %[[VAL_23]]) : (!fir.ref<i32>, !fir.boxchar<1>) -> ()
! CHECK:           fir.result %[[VAL_22]]#1 : !fir.array<?xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_8]], %[[VAL_24:.*]] to %[[VAL_0]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:         return
! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QPto_char_forall(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>,
! CHECK-SAME:                          %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>) {
subroutine to_char_forall(i, c)
  interface assignment(=)
    elemental subroutine sto_char(a,b)
      character(*), intent(out) :: a
      integer,intent(in) :: b
    end subroutine
  end interface
  integer :: i(:)
  character(*) :: c(:)
  forall (j=1:10) c(j) = i(j)
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
! CHECK:         %[[VAL_4:.*]] = constant 1 : i32
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
! CHECK:         %[[VAL_6:.*]] = constant 10 : i32
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:         %[[VAL_8:.*]] = constant 1 : index
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.array<?x!fir.char<1,?>>
! CHECK:         %[[VAL_10:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_8]] unordered iter_args(%[[VAL_13:.*]] = %[[VAL_9]]) -> (!fir.array<?x!fir.char<1,?>>) {
! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (index) -> i32
! CHECK:           fir.store %[[VAL_14]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> i64
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i64) -> index
! CHECK:           %[[VAL_18:.*]] = fir.array_fetch %[[VAL_10]], %[[VAL_17]] {Fortran.offsets} : (!fir.array<?xi32>, index) -> i32
! CHECK:           %[[VAL_19:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i32) -> i64
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i64) -> index
! CHECK:           %[[VAL_22:.*]]:2 = fir.array_modify %[[VAL_13]], %[[VAL_21]] {Fortran.offsets} : (!fir.array<?x!fir.char<1,?>>, index) -> (!fir.ref<!fir.char<1,?>>, !fir.array<?x!fir.char<1,?>>)
! CHECK:           %[[VAL_23:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_24:.*]] = fir.emboxchar %[[VAL_22]]#0, %[[VAL_23]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:           fir.store %[[VAL_18]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           fir.call @_QPsto_char(%[[VAL_24]], %[[VAL_2]]) : (!fir.boxchar<1>, !fir.ref<i32>) -> ()
! CHECK:           fir.result %[[VAL_22]]#1 : !fir.array<?x!fir.char<1,?>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_9]], %[[VAL_25:.*]] to %[[VAL_1]] : !fir.array<?x!fir.char<1,?>>, !fir.array<?x!fir.char<1,?>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:         return
! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QPfrom_char_forall_array(
! CHECK-SAME:                                  %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>>,
! CHECK-SAME:                                  %[[VAL_1:.*]]: !fir.box<!fir.array<?x?x!fir.char<1,?>>>) {
subroutine from_char_forall_array(i, c)
  interface assignment(=)
    elemental subroutine sfrom_char(a,b)
      integer, intent(out) :: a
      character(*),intent(in) :: b
    end subroutine
  end interface
  integer :: i(:, :)
  character(*) :: c(:, :)
  forall (j=1:10) i(j, :) = c(j, :)
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
! CHECK:         %[[VAL_3:.*]] = constant 1 : i32
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i32) -> index
! CHECK:         %[[VAL_5:.*]] = constant 10 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
! CHECK:         %[[VAL_7:.*]] = constant 1 : index
! CHECK:         %[[VAL_8:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.array<?x?xi32>
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.array<?x?x!fir.char<1,?>>
! CHECK:         %[[VAL_10:.*]] = fir.do_loop %[[VAL_11:.*]] = %[[VAL_4]] to %[[VAL_6]] step %[[VAL_7]] unordered iter_args(%[[VAL_12:.*]] = %[[VAL_8]]) -> (!fir.array<?x?xi32>) {
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_11]] : (index) -> i32
! CHECK:           fir.store %[[VAL_13]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_14:.*]] = constant 1 : index
! CHECK:           %[[VAL_15:.*]] = constant 1 : index
! CHECK:           %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_15]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_17:.*]] = addi %[[VAL_14]], %[[VAL_16]]#1 : index
! CHECK:           %[[VAL_18:.*]] = subi %[[VAL_17]], %[[VAL_14]] : index
! CHECK:           %[[VAL_19:.*]] = constant 1 : i64
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i64) -> index
! CHECK:           %[[VAL_21:.*]] = constant 0 : index
! CHECK:           %[[VAL_22:.*]] = subi %[[VAL_18]], %[[VAL_14]] : index
! CHECK:           %[[VAL_23:.*]] = addi %[[VAL_22]], %[[VAL_20]] : index
! CHECK:           %[[VAL_24:.*]] = divi_signed %[[VAL_23]], %[[VAL_20]] : index
! CHECK:           %[[VAL_25:.*]] = cmpi sgt, %[[VAL_24]], %[[VAL_21]] : index
! CHECK:           %[[VAL_26:.*]] = select %[[VAL_25]], %[[VAL_24]], %[[VAL_21]] : index
! CHECK:           %[[VAL_27:.*]] = constant 1 : index
! CHECK:           %[[VAL_28:.*]] = constant 0 : index
! CHECK:           %[[VAL_29:.*]] = subi %[[VAL_26]], %[[VAL_27]] : index
! CHECK:           %[[VAL_30:.*]] = fir.do_loop %[[VAL_31:.*]] = %[[VAL_28]] to %[[VAL_29]] step %[[VAL_27]] unordered iter_args(%[[VAL_32:.*]] = %[[VAL_12]]) -> (!fir.array<?x?xi32>) {
! CHECK:             %[[VAL_33:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:             %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i32) -> i64
! CHECK:             %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (i64) -> index
! CHECK:             %[[VAL_36:.*]] = constant 1 : index
! CHECK:             %[[VAL_37:.*]] = constant 1 : i64
! CHECK:             %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (i64) -> index
! CHECK:             %[[VAL_39:.*]] = muli %[[VAL_31]], %[[VAL_38]] : index
! CHECK:             %[[VAL_40:.*]] = addi %[[VAL_36]], %[[VAL_39]] : index
! CHECK:             %[[VAL_41:.*]] = fir.array_fetch %[[VAL_9]], %[[VAL_35]], %[[VAL_40]] {Fortran.offsets} : (!fir.array<?x?x!fir.char<1,?>>, index, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:             %[[VAL_42:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> index
! CHECK:             %[[VAL_43:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:             %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (i32) -> i64
! CHECK:             %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (i64) -> index
! CHECK:             %[[VAL_46:.*]] = constant 1 : index
! CHECK:             %[[VAL_47:.*]] = constant 1 : i64
! CHECK:             %[[VAL_48:.*]] = fir.convert %[[VAL_47]] : (i64) -> index
! CHECK:             %[[VAL_49:.*]] = muli %[[VAL_31]], %[[VAL_48]] : index
! CHECK:             %[[VAL_50:.*]] = addi %[[VAL_46]], %[[VAL_49]] : index
! CHECK:             %[[VAL_51:.*]]:2 = fir.array_modify %[[VAL_12]], %[[VAL_45]], %[[VAL_50]] {Fortran.offsets} : (!fir.array<?x?xi32>, index, index) -> (!fir.ref<i32>, !fir.array<?x?xi32>)
! CHECK:             %[[VAL_52:.*]] = fir.emboxchar %[[VAL_41]], %[[VAL_42]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:             fir.call @_QPsfrom_char(%[[VAL_51]]#0, %[[VAL_52]]) : (!fir.ref<i32>, !fir.boxchar<1>) -> ()
! CHECK:             fir.result %[[VAL_51]]#1 : !fir.array<?x?xi32>
! CHECK:           }
! CHECK:           fir.result %[[VAL_53:.*]] : !fir.array<?x?xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_8]], %[[VAL_54:.*]] to %[[VAL_0]] : !fir.array<?x?xi32>, !fir.array<?x?xi32>, !fir.box<!fir.array<?x?xi32>>
! CHECK:         return
! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QPto_char_forall_array(
! CHECK-SAME:                                %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>>,
! CHECK-SAME:                                %[[VAL_1:.*]]: !fir.box<!fir.array<?x?x!fir.char<1,?>>>) {
subroutine to_char_forall_array(i, c)
  interface assignment(=)
    elemental subroutine sto_char(a,b)
      character(*), intent(out) :: a
      integer,intent(in) :: b
    end subroutine
  end interface
  integer :: i(:, :)
  character(*) :: c(:, :)
  forall (j=1:10) c(j, :) = i(j, :)
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
! CHECK:         %[[VAL_4:.*]] = constant 1 : i32
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
! CHECK:         %[[VAL_6:.*]] = constant 10 : i32
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:         %[[VAL_8:.*]] = constant 1 : index
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.array<?x?x!fir.char<1,?>>
! CHECK:         %[[VAL_10:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.array<?x?xi32>
! CHECK:         %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_8]] unordered iter_args(%[[VAL_13:.*]] = %[[VAL_9]]) -> (!fir.array<?x?x!fir.char<1,?>>) {
! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (index) -> i32
! CHECK:           fir.store %[[VAL_14]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_15:.*]] = constant 1 : index
! CHECK:           %[[VAL_16:.*]] = constant 1 : index
! CHECK:           %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_16]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_18:.*]] = addi %[[VAL_15]], %[[VAL_17]]#1 : index
! CHECK:           %[[VAL_19:.*]] = subi %[[VAL_18]], %[[VAL_15]] : index
! CHECK:           %[[VAL_20:.*]] = constant 1 : i64
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i64) -> index
! CHECK:           %[[VAL_22:.*]] = constant 0 : index
! CHECK:           %[[VAL_23:.*]] = subi %[[VAL_19]], %[[VAL_15]] : index
! CHECK:           %[[VAL_24:.*]] = addi %[[VAL_23]], %[[VAL_21]] : index
! CHECK:           %[[VAL_25:.*]] = divi_signed %[[VAL_24]], %[[VAL_21]] : index
! CHECK:           %[[VAL_26:.*]] = cmpi sgt, %[[VAL_25]], %[[VAL_22]] : index
! CHECK:           %[[VAL_27:.*]] = select %[[VAL_26]], %[[VAL_25]], %[[VAL_22]] : index
! CHECK:           %[[VAL_28:.*]] = constant 1 : index
! CHECK:           %[[VAL_29:.*]] = constant 0 : index
! CHECK:           %[[VAL_30:.*]] = subi %[[VAL_27]], %[[VAL_28]] : index
! CHECK:           %[[VAL_31:.*]] = fir.do_loop %[[VAL_32:.*]] = %[[VAL_29]] to %[[VAL_30]] step %[[VAL_28]] unordered iter_args(%[[VAL_33:.*]] = %[[VAL_13]]) -> (!fir.array<?x?x!fir.char<1,?>>) {
! CHECK:             %[[VAL_34:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:             %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (i32) -> i64
! CHECK:             %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (i64) -> index
! CHECK:             %[[VAL_37:.*]] = constant 1 : index
! CHECK:             %[[VAL_38:.*]] = constant 1 : i64
! CHECK:             %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (i64) -> index
! CHECK:             %[[VAL_40:.*]] = muli %[[VAL_32]], %[[VAL_39]] : index
! CHECK:             %[[VAL_41:.*]] = addi %[[VAL_37]], %[[VAL_40]] : index
! CHECK:             %[[VAL_42:.*]] = fir.array_fetch %[[VAL_10]], %[[VAL_36]], %[[VAL_41]] {Fortran.offsets} : (!fir.array<?x?xi32>, index, index) -> i32
! CHECK:             %[[VAL_43:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:             %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (i32) -> i64
! CHECK:             %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (i64) -> index
! CHECK:             %[[VAL_46:.*]] = constant 1 : index
! CHECK:             %[[VAL_47:.*]] = constant 1 : i64
! CHECK:             %[[VAL_48:.*]] = fir.convert %[[VAL_47]] : (i64) -> index
! CHECK:             %[[VAL_49:.*]] = muli %[[VAL_32]], %[[VAL_48]] : index
! CHECK:             %[[VAL_50:.*]] = addi %[[VAL_46]], %[[VAL_49]] : index
! CHECK:             %[[VAL_51:.*]]:2 = fir.array_modify %[[VAL_13]], %[[VAL_45]], %[[VAL_50]] {Fortran.offsets} : (!fir.array<?x?x!fir.char<1,?>>, index, index) -> (!fir.ref<!fir.char<1,?>>, !fir.array<?x?x!fir.char<1,?>>)
! CHECK:             %[[VAL_52:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> index
! CHECK:             %[[VAL_53:.*]] = fir.emboxchar %[[VAL_51]]#0, %[[VAL_52]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:             fir.store %[[VAL_42]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:             fir.call @_QPsto_char(%[[VAL_53]], %[[VAL_2]]) : (!fir.boxchar<1>, !fir.ref<i32>) -> ()
! CHECK:             fir.result %[[VAL_51]]#1 : !fir.array<?x?x!fir.char<1,?>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_54:.*]] : !fir.array<?x?x!fir.char<1,?>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_9]], %[[VAL_55:.*]] to %[[VAL_1]] : !fir.array<?x?x!fir.char<1,?>>, !fir.array<?x?x!fir.char<1,?>>, !fir.box<!fir.array<?x?x!fir.char<1,?>>>
! CHECK:         return
! CHECK:       }
end subroutine

! TODO: test array user defined assignment inside FORALL.
!subroutine test_todo(x, y)
!  interface assignment(=)
!    ! User assignment is not elemental, it takes array arguments.
!    pure subroutine assign_array(a,b)
!      logical, intent(out) :: a(:)
!      integer, intent(in) :: b(:)
!    end
!  end interface
!  logical :: x(10, 10)
!  integer :: y(10, 10)
!  forall(i=1:10) x(i, :) = y(i, :)
!end subroutine
