! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPreshape_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:[^:]+]]: !fir.box<!fir.array<?x?x?xi32>>{{.*}}, %[[arg2:[^:]+]]: !fir.box<!fir.array<?x?x?xi32>>{{.*}}, %[[arg3:.*]]: !fir.ref<!fir.array<2xi32>>{{.*}}, %[[arg4:.*]]: !fir.ref<!fir.array<2xi32>>{{.*}}) {
subroutine reshape_test(x, source, pd, sh, ord)
  integer :: x(:,:)
  integer :: source(:,:,:)
  integer :: pd(:,:,:)
  integer :: sh(2)
  integer :: ord(2)
! CHECK-DAG:  %[[c2:.*]] = arith.constant 2 : index
! CHECK-DAG:  %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>>
! CHECK-DAG:  %[[a1:.*]] = fir.shape %[[c2]] : (index) -> !fir.shape<1>
! CHECK-DAG:  %[[a2:.*]] = fir.embox %[[arg3]](%{{.*}}) : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xi32>>
! CHECK-DAG:  %[[a3:.*]] = fir.embox %[[arg4]](%{{.*}}) : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xi32>>
! CHECK-DAG:  %[[a8:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK-DAG:  %[[a9:.*]] = fir.convert %[[arg1]] : (!fir.box<!fir.array<?x?x?xi32>>) -> !fir.box<none>
! CHECK-DAG:  %[[a10:.*]] = fir.convert %[[a2]] : (!fir.box<!fir.array<2xi32>>) -> !fir.box<none>
! CHECK-DAG:  %[[a11:.*]] = fir.convert %[[arg2]] : (!fir.box<!fir.array<?x?x?xi32>>) -> !fir.box<none>
! CHECK-DAG:  %[[a12:.*]] = fir.convert %[[a3]] : (!fir.box<!fir.array<2xi32>>) -> !fir.box<none>
  x = reshape(source, sh, pd, ord)
! CHECK:  %{{.*}} = fir.call @_FortranAReshape(%[[a8]], %[[a9]], %[[a10]], %[[a11]], %[[a12]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK-DAG:  %[[a15:.*]] = fir.load %[[a0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
! CHECK-DAG:  %[[a18:.*]] = fir.box_addr %[[a15]] : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>) -> !fir.heap<!fir.array<?x?xi32>>
! CHECK-DAG:  fir.freemem %[[a18]] : !fir.heap<!fir.array<?x?xi32>>
end subroutine

