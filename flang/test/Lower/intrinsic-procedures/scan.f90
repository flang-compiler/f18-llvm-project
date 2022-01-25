! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPscan_test(
! CHECK-SAME: %[[s:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[ss:[^:]+]]: !fir.boxchar<1>{{.*}}) -> i32
integer function scan_test(s1, s2)
  character(*) :: s1, s2
! CHECK: %[[tmpBox:.*]] = fir.alloca !fir.box<!fir.heap<i32>>
! CHECK-DAG: %[[c:.*]]:2 = fir.unboxchar %[[s]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-DAG: %[[cBox:.*]] = fir.embox %[[c]]#0 typeparams %[[c]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-DAG: %[[cBoxNone:.*]] = fir.convert %[[cBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-DAG: %[[c2:.*]]:2 = fir.unboxchar %[[ss]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-DAG: %[[cBox2:.*]] = fir.embox %[[c2]]#0 typeparams %[[c2]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-DAG: %[[cBoxNone2:.*]] = fir.convert %[[cBox2]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-DAG: %[[backOptBox:.*]] = fir.absent !fir.box<i1>
! CHECK-DAG: %[[backBox:.*]] = fir.convert %[[backOptBox]] : (!fir.box<i1>) -> !fir.box<none>
! CHECK-DAG: %[[kindConstant:.*]] = arith.constant 4 : i32
! CHECK-DAG: %[[resBox:.*]] = fir.convert %[[tmpBox:.*]] : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK: fir.call @{{.*}}Scan(%[[resBox]], %[[cBoxNone]], %[[cBoxNone2]], %[[backBox]], %[[kindConstant]], {{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
  scan_test = scan(s1, s2, kind=4)
! CHECK-DAG: %[[tmpAddr:.*]] = fir.box_addr
! CHECK: fir.freemem %[[tmpAddr]] : !fir.heap<i32>
end function scan_test

! CHECK-LABEL: func @_QPscan_test2(
! CHECK-SAME: %[[s:[^:]+]]: !fir.boxchar<1>{{.*}},
! CHECK-SAME: %[[ss:[^:]+]]: !fir.boxchar<1>{{.*}}) -> i32
integer function scan_test2(s1, s2)
  character(*) :: s1, s2
  ! CHECK: %[[st:[^:]*]]:2 = fir.unboxchar %[[s]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[sst:[^:]*]]:2 = fir.unboxchar %[[ss]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[a1:.*]] = fir.convert %[[st]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: %[[a2:.*]] = fir.convert %[[st]]#1 : (index) -> i64
  ! CHECK: %[[a3:.*]] = fir.convert %[[sst]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: %[[a4:.*]] = fir.convert %[[sst]]#1 : (index) -> i64
  ! CHECK: = fir.call @_FortranAScan1(%[[a1]], %[[a2]], %[[a3]], %[[a4]], %{{.*}}) : (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
  scan_test2 = scan(s1, s2, .true.)
end function scan_test2

