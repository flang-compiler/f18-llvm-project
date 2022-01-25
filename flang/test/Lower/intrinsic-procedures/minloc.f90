! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPminloc_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[arg1:.*]]: !fir.box<!fir.array<?xi32>>
subroutine minloc_test(arr,res)
  integer :: arr(:)
  integer :: res(:)
! CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
! CHECK-DAG: %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK-DAG: %[[a1:.*]] = fir.absent !fir.box<i1>
! CHECK-DAG: %[[a6:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK-DAG: %[[a7:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK-DAG: %[[a8:.*]] = fir.convert %[[c4]] : (index) -> i32
! CHECK-DAG: %[[a10:.*]] = fir.convert %[[a1]] : (!fir.box<i1>) -> !fir.box<none>
  res = minloc(arr)
! CHECK: %{{.*}} = fir.call @_FortranAMinloc(%[[a6]], %[[a7]], %[[a8]], %{{.*}}, %{{.*}}, %[[a10]], %false) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
! CHECK-DAG: %[[a12:.*]] = fir.load %[[a0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK-DAG: %[[a14:.*]] = fir.box_addr %[[a12]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK-DAG: fir.freemem %[[a14]] : !fir.heap<!fir.array<?xi32>>
end subroutine

! CHECK-LABEL: func @_QPminloc_test2(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[arg1:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[arg2:.*]]: !fir.ref<i32>
subroutine minloc_test2(arr,res,d)
  integer :: arr(:)
  integer :: res(:)
  integer :: d
! CHECK-DAG:  %[[c4:.*]] = arith.constant 4 : index
! CHECK-DAG:  %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<i32>>
! CHECK-DAG:  %[[a1:.*]] = fir.load %arg2 : !fir.ref<i32>
! CHECK-DAG:  %[[a2:.*]] = fir.absent !fir.box<i1>
! CHECK-DAG:  %[[a6:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK-DAG:  %[[a7:.*]] = fir.convert %arg0 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK-DAG:  %[[a8:.*]] = fir.convert %[[c4]] : (index) -> i32
! CHECK-DAG:  %[[a10:.*]] = fir.convert %[[a2]] : (!fir.box<i1>) -> !fir.box<none>
  res = minloc(arr, dim=d)
! CHECK:  %{{.*}} = fir.call @_FortranAMinlocDim(%[[a6]], %[[a7]], %[[a8]], %[[a1]], %{{.*}}, %{{.*}}, %[[a10]], %false) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
! CHECK:  %[[a12:.*]] = fir.load %0 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:  %[[a13:.*]] = fir.box_addr %[[a12]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:  fir.freemem %[[a13]] : !fir.heap<i32>
end subroutine
