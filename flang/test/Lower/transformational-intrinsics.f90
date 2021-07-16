! Test how transformational intrinsic function references are lowered

! RUN: bbc -emit-fir %s -o - | FileCheck %s

! The exact intrinsic being tested does not really matter, what is
! tested here is that transformational intrinsics are lowered correctly
! regardless of the context they appear into.



module test2
interface
  subroutine takes_array_desc(l)
    logical(1) :: l(:)
  end subroutine
end interface

contains

! CHECK-LABEL: func @_QMtest2Pin_io(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<1>>>) {
subroutine in_io(x)
  logical(1) :: x(:, :)
  ! CHECK: %[[res_desc:.]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>
  ! CHECK-DAG: %[[res_arg:.*]] = fir.convert %[[res_desc]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[x_arg:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?x!fir.logical<1>>>) -> !fir.box<none>
  ! CHECK: fir.call @_Fortran{{.*}}AllDim(%[[res_arg]], %[[x_arg]], {{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
  ! CHECK: %[[res_desc_load:.*]] = fir.load %[[res_desc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>>
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[res_desc_load]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[res_addr:.*]] = fir.box_addr %[[res_desc_load]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>) -> !fir.heap<!fir.array<?x!fir.logical<1>>>
  ! CHECK-DAG: %[[res_shape:.*]] = fir.shape_shift %[[dims]]#0, %[[dims]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[io_embox:.*]] = fir.embox %[[res_addr]](%[[res_shape]]) : (!fir.heap<!fir.array<?x!fir.logical<1>>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?x!fir.logical<1>>>
  ! CHECK: %[[io_embox_cast:.*]] = fir.convert %[[io_embox]] : (!fir.box<!fir.array<?x!fir.logical<1>>>) -> !fir.box<none>
  ! CHECK: fir.call @_Fortran{{.*}}ioOutputDescriptor({{.*}}, %[[io_embox_cast]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
  print *, all(x, 1)
  ! CHECK: fir.freemem %[[res_addr]] : !fir.heap<!fir.array<?x!fir.logical<1>>>
end subroutine

! CHECK-LABEL: func @_QMtest2Pin_call(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<1>>>) {
subroutine in_call(x)
  implicit none
  logical(1) :: x(:, :)
  ! CHECK: %[[res_desc:.]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>
  ! CHECK-DAG: %[[res_arg:.*]] = fir.convert %[[res_desc]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[x_arg:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?x!fir.logical<1>>>) -> !fir.box<none>
  ! CHECK: fir.call @_Fortran{{.*}}AllDim(%[[res_arg]], %[[x_arg]], {{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
  ! CHECK: %[[res_desc_load:.*]] = fir.load %[[res_desc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>>
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[res_desc_load]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[res_addr:.*]] = fir.box_addr %[[res_desc_load]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>) -> !fir.heap<!fir.array<?x!fir.logical<1>>>
  ! CHECK-DAG: %[[res_shape:.*]] = fir.shape_shift %[[dims]]#0, %[[dims]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[call_embox:.*]] = fir.embox %[[res_addr]](%[[res_shape]]) : (!fir.heap<!fir.array<?x!fir.logical<1>>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?x!fir.logical<1>>>
  ! CHECK: fir.call @_QPtakes_array_desc(%[[call_embox]]) : (!fir.box<!fir.array<?x!fir.logical<1>>>) -> ()
  call takes_array_desc(all(x, 1))
  ! CHECK: fir.freemem %[[res_addr]] : !fir.heap<!fir.array<?x!fir.logical<1>>>
end subroutine

! CHECK-LABEL: func @_QMtest2Pin_implicit_call(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<1>>>) {
subroutine in_implicit_call(x)
  logical(1) :: x(:, :)
  ! CHECK: %[[res_desc:.]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>
  ! CHECK: fir.call @_Fortran{{.*}}AllDim
  ! CHECK: %[[res_desc_load:.*]] = fir.load %[[res_desc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>>
  ! CHECK: %[[res_addr:.*]] = fir.box_addr %[[res_desc_load]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>) -> !fir.heap<!fir.array<?x!fir.logical<1>>>
  ! CHECK: %[[res_addr_cast:.*]] = fir.convert %[[res_addr]] : (!fir.heap<!fir.array<?x!fir.logical<1>>>) -> !fir.ref<!fir.array<?x!fir.logical<1>>>
  ! CHECK: fir.call @_QPtakes_implicit_array(%[[res_addr_cast]]) : (!fir.ref<!fir.array<?x!fir.logical<1>>>) -> ()
  call takes_implicit_array(all(x, 1))
  ! CHECK: fir.freemem %[[res_addr]] : !fir.heap<!fir.array<?x!fir.logical<1>>>
end subroutine

! CHECK-LABEL: func @_QMtest2Pin_assignment(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<1>>>,
! CHECK-SAME: %[[arg1:.*]]: !fir.box<!fir.array<?x!fir.logical<1>>>)
subroutine in_assignment(x, y)
  logical(1) :: x(:, :), y(:)
  ! CHECK: %[[res_desc:.]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>
  ! CHECK: %[[y_load:.*]] = fir.array_load %[[arg1]] : (!fir.box<!fir.array<?x!fir.logical<1>>>) -> !fir.array<?x!fir.logical<1>>
  ! CHECK: fir.call @_Fortran{{.*}}AllDim

  ! CHECK: %[[res_desc_load:.*]] = fir.load %[[res_desc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>>
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[res_desc_load]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[res_addr:.*]] = fir.box_addr %[[res_desc_load]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>) -> !fir.heap<!fir.array<?x!fir.logical<1>>>
  ! CHECK-DAG: %[[res_shape:.*]] = fir.shape_shift %[[dims]]#0, %[[dims]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[res_load:.*]] = fir.array_load %[[res_addr]](%[[res_shape]]) : (!fir.heap<!fir.array<?x!fir.logical<1>>>, !fir.shapeshift<1>) -> !fir.array<?x!fir.logical<1>>

  ! CHECK: %[[assign:.*]] = fir.do_loop %[[idx:.*]] = %{{.*}} to {{.*}} {
    ! CHECK: %[[res_elt:.*]] = fir.array_fetch %[[res_load]], %[[idx]] : (!fir.array<?x!fir.logical<1>>, index) -> !fir.logical<1>
    ! CHECK: fir.array_update %{{.*}} %[[res_elt]], %[[idx]] : (!fir.array<?x!fir.logical<1>>, !fir.logical<1>, index) -> !fir.array<?x!fir.logical<1>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[y_load]], %[[assign]] to %[[arg1]] : !fir.array<?x!fir.logical<1>>, !fir.array<?x!fir.logical<1>>, !fir.box<!fir.array<?x!fir.logical<1>>>
  y = all(x, 1)
  ! CHECK: fir.freemem %[[res_addr]] : !fir.heap<!fir.array<?x!fir.logical<1>>>
end subroutine

! CHECK-LABEL: func @_QMtest2Pin_elem_expr(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<1>>>,
! CHECK-SAME: %[[arg1:.*]]: !fir.box<!fir.array<?x!fir.logical<1>>>,
! CHECK-SAME: %[[arg2:.*]]: !fir.box<!fir.array<?x!fir.logical<1>>>)
subroutine in_elem_expr(x, y, z)
  logical(1) :: x(:, :), y(:), z(:)
  ! CHECK: %[[res_desc:.]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>
  ! CHECK-DAG: %[[y_load:.*]] = fir.array_load %[[arg1]] : (!fir.box<!fir.array<?x!fir.logical<1>>>) -> !fir.array<?x!fir.logical<1>>
  ! CHECK-DAG: %[[z_load:.*]] = fir.array_load %[[arg2]] : (!fir.box<!fir.array<?x!fir.logical<1>>>) -> !fir.array<?x!fir.logical<1>>
  ! CHECK: fir.call @_Fortran{{.*}}AllDim

  ! CHECK: %[[res_desc_load:.*]] = fir.load %[[res_desc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>>
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[res_desc_load]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[res_addr:.*]] = fir.box_addr %[[res_desc_load]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>) -> !fir.heap<!fir.array<?x!fir.logical<1>>>
  ! CHECK-DAG: %[[res_shape:.*]] = fir.shape_shift %[[dims]]#0, %[[dims]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[res_load:.*]] = fir.array_load %[[res_addr]](%[[res_shape]]) : (!fir.heap<!fir.array<?x!fir.logical<1>>>, !fir.shapeshift<1>) -> !fir.array<?x!fir.logical<1>>

  ! CHECK: %[[elem_expr:.*]] = fir.do_loop %[[idx:.*]] = %{{.*}} to {{.*}} {
    ! CHECK-DAG: %[[y_elt:.*]] = fir.array_fetch %[[y_load]], %[[idx]] : (!fir.array<?x!fir.logical<1>>, index) -> !fir.logical<1>
    ! CHECK-DAG: %[[res_elt:.*]] = fir.array_fetch %[[res_load]], %[[idx]] : (!fir.array<?x!fir.logical<1>>, index) -> !fir.logical<1>
    ! CHECK-DAG: %[[y_elt_i1:.*]] = fir.convert %[[y_elt]] : (!fir.logical<1>) -> i1
    ! CHECK-DAG: %[[res_elt_i1:.*]] = fir.convert %[[res_elt]] : (!fir.logical<1>) -> i1
    ! CHECK: %[[neqv_i1:.*]] = cmpi ne, %[[y_elt_i1]], %[[res_elt_i1]] : i1
    ! CHECK: %[[neqv:.*]] = fir.convert %[[neqv_i1]] : (i1) -> !fir.logical<1>
    ! CHECK: fir.array_update %{{.*}} %[[neqv]], %[[idx]] : (!fir.array<?x!fir.logical<1>>, !fir.logical<1>, index) -> !fir.array<?x!fir.logical<1>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[z_load]], %[[elem_expr]] to %[[arg2]] : !fir.array<?x!fir.logical<1>>, !fir.array<?x!fir.logical<1>>, !fir.box<!fir.array<?x!fir.logical<1>>>
  z = y .neqv. all(x, 1)
  ! CHECK: fir.freemem %[[res_addr]] : !fir.heap<!fir.array<?x!fir.logical<1>>>
end subroutine

! UNPACK
! CHECK-LABEL: unpack_test
subroutine unpack_test()
  integer, dimension(3) :: vector
  integer, dimension (3,3) :: field

  logical, dimension(3,3) :: mask
  integer, dimension(3,3) :: result
  result = unpack(vector, mask, field)
  ! CHECK-DAG: %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>> {uniq_name = ""}
  ! CHECK-DAG: %[[a1:.*]] = fir.alloca i32
  ! CHECK-DAG: %[[a2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>> {uniq_name = ""}
  ! CHECK-DAG: %[[a3:.*]] = fir.alloca !fir.array<3x3xi32> {bindc_name = "field", uniq_name = "_QMtest2Funpack_testEfield"}
  ! CHECK-DAG: %[[a4:.*]] = fir.alloca !fir.array<3x3x!fir.logical<4>> {bindc_name = "mask", uniq_name = "_QMtest2Funpack_testEmask"}
  ! CHECK-DAG: %[[a5:.*]] = fir.alloca !fir.array<3x3xi32> {bindc_name = "result", uniq_name = "_QMtest2Funpack_testEresult"}
  ! CHECK-DAG: %[[a6:.*]] = fir.alloca !fir.array<3xi32> {bindc_name = "vector", uniq_name = "_QMtest2Funpack_testEvector"}
  ! CHECK-DAG: %[[a7:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK-DAG: %[[a8:.*]] = fir.array_load %[[a5]](%[[a7]]) : (!fir.ref<!fir.array<3x3xi32>>, !fir.shape<2>) -> !fir.array<3x3xi32>
  ! CHECK-DAG: %[[a9:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
  ! CHECK-DAG: %[[a10:.*]] = fir.embox %[[a6]](%[[a9]]) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi32>>
  ! CHECK-DAG: %[[a11:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK-DAG: %[[a12:.*]] = fir.embox %[[a4]](%[[a11]]) : (!fir.ref<!fir.array<3x3x!fir.logical<4>>>, !fir.shape<2>) -> !fir.box<!fir.array<3x3x!fir.logical<4>>>
  ! CHECK-DAG: %[[a13:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK-DAG: %[[a14:.*]] = fir.embox %[[a3]](%[[a13]]) : (!fir.ref<!fir.array<3x3xi32>>, !fir.shape<2>) -> !fir.box<!fir.array<3x3xi32>>
  ! CHECK-DAG: %[[a15:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xi32>>
  ! CHECK-DAG: %[[a16:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK-DAG: %[[a17:.*]] = fir.embox %[[a15]](%[[a16]]) : (!fir.heap<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xi32>>>
  ! CHECK-DAG: fir.store %[[a17]] to %[[a2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
  ! CHECK-DAG: %[[a19:.*]] = fir.convert %[[a2]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[a20:.*]] = fir.convert %[[a10]] : (!fir.box<!fir.array<3xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[a21:.*]] = fir.convert %[[a12]] : (!fir.box<!fir.array<3x3x!fir.logical<4>>>) -> !fir.box<none>
  ! CHECK-DAG: %[[a22:.*]] = fir.convert %[[a14]] : (!fir.box<!fir.array<3x3xi32>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAUnpack(%[[a19]], %[[a20]], %[[a21]], %[[a22]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  ! CHECK: %[[a22:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
  ! CHECK-DAG: %[[a25:.*]] = fir.box_addr %[[a22]] : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>) -> !fir.heap<!fir.array<?x?xi32>>
  ! CHECK: fir.freemem %[[a25]] : !fir.heap<!fir.array<?x?xi32>>
  ! CHECK-DAG: %[[a36:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK-DAG: %[[a38:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
  ! CHECK-DAG: %[[a39:.*]] = fir.embox %[[a6]](%[[a38]]) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi32>>
  ! CHECK-DAG: %[[a40:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK-DAG: %[[a41:.*]] = fir.embox %[[a4]](%40) : (!fir.ref<!fir.array<3x3x!fir.logical<4>>>, !fir.shape<2>) -> !fir.box<!fir.array<3x3x!fir.logical<4>>>
  ! CHECK-DAG: %[[a42:.*]] = fir.embox %[[a1]] : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG: %[[a43:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xi32>>
  ! CHECK-DAG: %[[a44:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK-DAG: %[[a45:.*]] = fir.embox %[[a43]](%[[a44]]) : (!fir.heap<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xi32>>>
  ! CHECK-DAG: fir.store %[[a45]] to %[[a0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
  ! CHECK-DAG: %[[a47:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[a48:.*]] = fir.convert %[[a39]] : (!fir.box<!fir.array<3xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[a49:.*]] = fir.convert %[[a41]] : (!fir.box<!fir.array<3x3x!fir.logical<4>>>) -> !fir.box<none>
  ! CHECK-DAG: %[[a50:.*]] = fir.convert %[[a42]] : (!fir.box<i32>) -> !fir.box<none>
  result = unpack(vector, mask, 343)
  ! CHECK: fir.call @_FortranAUnpack(%[[a47]], %[[a48]], %[[a49]], %[[a50]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  ! CHECK-DAG: %[[a53:.*]] = fir.load %[[a0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
  ! CHECK-DAG: %[[a56:.*]] = fir.box_addr %[[a53]] : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>) -> !fir.heap<!fir.array<?x?xi32>>
  ! CHECK: fir.freemem %[[a56]] : !fir.heap<!fir.array<?x?xi32>>
end subroutine unpack_test

end module
