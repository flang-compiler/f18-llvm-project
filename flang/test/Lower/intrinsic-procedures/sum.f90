! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPsum_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) -> i32 {
integer function sum_test(a)
  integer :: a(:)
! CHECK-DAG:  %[[c0:.*]] = arith.constant 0 : index
! CHECK-DAG:  %[[a1:.*]] = fir.absent !fir.box<i1>
! CHECK-DAG: %[[a3:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK-DAG:  %[[a5:.*]] = fir.convert %[[c0]] : (index) -> i32
! CHECK-DAG:  %[[a6:.*]] = fir.convert %[[a1]] : (!fir.box<i1>) -> !fir.box<none>
  sum_test = sum(a)
! CHECK:  %{{.*}} = fir.call @_FortranASumInteger4(%[[a3]], %{{.*}}, %{{.*}}, %[[a5]], %[[a6]]) : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
end function

! CHECK-LABEL: func @_QPsum_test2(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) {
subroutine sum_test2(a,r)
  integer :: a(:,:)
  integer :: r(:)
! CHECK-DAG:  %[[c2_i32:.*]] = arith.constant 2 : i32
! CHECK-DAG:  %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK-DAG:  %[[a1:.*]] = fir.absent !fir.box<i1>
! CHECK-DAG:  %[[a6:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK-DAG:  %[[a7:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
! CHECK-DAG:  %[[a9:.*]] = fir.convert %[[a1]] : (!fir.box<i1>) -> !fir.box<none>
  r = sum(a,dim=2)
! CHECK:  %{{.*}} = fir.call @_FortranASumDim(%[[a6]], %[[a7]], %[[c2_i32]], %{{.*}}, %{{.*}}, %[[a9]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>) -> none
! CHECK-DAG: %[[a11:.*]] = fir.load %[[a0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK-DAG:  %[[a13:.*]] = fir.box_addr %[[a11]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK-DAG:  fir.freemem %[[a13]] : !fir.heap<!fir.array<?xi32>>
end subroutine

! CHECK-LABEL: func @_QPsum_test3(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x!fir.complex<4>>>{{.*}}) -> !fir.complex<4> {
complex function sum_test3(a)
  complex :: a(:)
! CHECK-DAG:  %[[c0:.*]] = arith.constant 0 : index
! CHECK-DAG:  %[[a0:.*]] = fir.alloca !fir.complex<4>
! CHECK-DAG:  %[[a3:.*]] = fir.absent !fir.box<i1>
! CHECK-DAG: %[[a5:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.complex<4>>) -> !fir.ref<complex<f32>>
! CHECK-DAG:  %[[a6:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x!fir.complex<4>>>) -> !fir.box<none>
! CHECK-DAG:  %[[a8:.*]] = fir.convert %[[c0]] : (index) -> i32
! CHECK-DAG:  %[[a9:.*]] = fir.convert %[[a3]] : (!fir.box<i1>) -> !fir.box<none>
  sum_test3 = sum(a)
! CHECK:  %{{.*}} = fir.call @_FortranACppSumComplex4(%[[a5]], %[[a6]], %{{.*}}, %{{.*}}, %[[a8]], %[[a9]]) : (!fir.ref<complex<f32>>, !fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> none
end function

! CHECK-LABEL: func @_QPsum_test4(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x!fir.complex<10>>>{{.*}}) -> !fir.complex<10> {
complex(10) function sum_test4(x)
  complex(10):: x(:)
! CHECK-DAG:  %[[c0:.*]] = arith.constant 0 : index
! CHECK-DAG:  %[[a0:.*]] = fir.alloca !fir.complex<10>
  sum_test4 = sum(x)
! CHECK-DAG: %[[a2:.*]] = fir.absent !fir.box<i1>
! CHECK-DAG: %[[a4:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.complex<10>>) -> !fir.ref<complex<f80>>
! CHECK-DAG: %[[a5:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x!fir.complex<10>>>) -> !fir.box<none>
! CHECK-DAG:  %[[a7:.*]] = fir.convert %[[c0]] : (index) -> i32
! CHECK-DAG:  %[[a8:.*]] = fir.convert %[[a2]] : (!fir.box<i1>) -> !fir.box<none>
! CHECK: fir.call @_FortranACppSumComplex10(%[[a4]], %[[a5]], %{{.*}}, %{{.*}}, %[[a7]], %8) : (!fir.ref<complex<f80>>, !fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> ()
end
