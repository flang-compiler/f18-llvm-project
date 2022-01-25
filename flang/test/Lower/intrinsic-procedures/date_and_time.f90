! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPdate_and_time_test(
! CHECK-SAME: %[[date:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[time:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[zone:.*]]: !fir.boxchar<1>{{.*}}, %[[values:.*]]: !fir.box<!fir.array<?xi64>>{{.*}}) {
subroutine date_and_time_test(date, time, zone, values)
  character(*) :: date, time, zone
  integer(8) :: values(:)
  ! CHECK: %[[dateUnbox:.*]]:2 = fir.unboxchar %[[date]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[timeUnbox:.*]]:2 = fir.unboxchar %[[time]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[zoneUnbox:.*]]:2 = fir.unboxchar %[[zone]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[dateBuffer:.*]] = fir.convert %[[dateUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: %[[dateLen:.*]] = fir.convert %[[dateUnbox]]#1 : (index) -> i64
  ! CHECK: %[[timeBuffer:.*]] = fir.convert %[[timeUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: %[[timeLen:.*]] = fir.convert %[[timeUnbox]]#1 : (index) -> i64
  ! CHECK: %[[zoneBuffer:.*]] = fir.convert %[[zoneUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: %[[zoneLen:.*]] = fir.convert %[[zoneUnbox]]#1 : (index) -> i64
  ! CHECK: %[[valuesCast:.*]] = fir.convert %[[values]] : (!fir.box<!fir.array<?xi64>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranADateAndTime(%[[dateBuffer]], %[[dateLen]], %[[timeBuffer]], %[[timeLen]], %[[zoneBuffer]], %[[zoneLen]], %{{.*}}, %{{.*}}, %[[valuesCast]]) : (!fir.ref<i8>, i64, !fir.ref<i8>, i64, !fir.ref<i8>, i64, !fir.ref<i8>, i32, !fir.box<none>) -> none
  call date_and_time(date, time, zone, values)
end subroutine

! CHECK-LABEL: func @_QPdate_and_time_test2(
! CHECK-SAME: %[[date:.*]]: !fir.boxchar<1>{{.*}})
subroutine date_and_time_test2(date)
  character(*) :: date
  ! CHECK: %[[dateUnbox:.*]]:2 = fir.unboxchar %[[date]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[values:.*]] = fir.absent !fir.box<none> 
  ! CHECK: %[[dateBuffer:.*]] = fir.convert %[[dateUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: %[[dateLen:.*]] = fir.convert %[[dateUnbox]]#1 : (index) -> i64
  ! CHECK: %[[timeBuffer:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.ref<i8>
  ! CHECK: %[[timeLen:.*]] = fir.convert %c0{{.*}} : (index) -> i64
  ! CHECK: %[[zoneBuffer:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.ref<i8>
  ! CHECK: %[[zoneLen:.*]] = fir.convert %c0{{.*}} : (index) -> i64
  ! CHECK: fir.call @_FortranADateAndTime(%[[dateBuffer]], %[[dateLen]], %[[timeBuffer]], %[[timeLen]], %[[zoneBuffer]], %[[zoneLen]], %{{.*}}, %{{.*}}, %[[values]]) : (!fir.ref<i8>, i64, !fir.ref<i8>, i64, !fir.ref<i8>, i64, !fir.ref<i8>, i32, !fir.box<none>) -> none
  call date_and_time(date)
end subroutine
