! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: system_clock_test
subroutine system_clock_test()
  integer(4) :: c
  integer(8) :: m
  real :: r
  ! CHECK-DAG: %[[c:.*]] = fir.alloca i32 {bindc_name = "c"
  ! CHECK-DAG: %[[m:.*]] = fir.alloca i64 {bindc_name = "m"
  ! CHECK-DAG: %[[r:.*]] = fir.alloca f32 {bindc_name = "r"
  ! CHECK: %[[Count:.*]] = fir.call @_FortranASystemClockCount() : () -> i64
  ! CHECK: %[[Count1:.*]] = fir.convert %[[Count]] : (i64) -> i32
  ! CHECK: fir.store %[[Count1]] to %[[c]] : !fir.ref<i32>
  ! CHECK: %[[Rate:.*]] = fir.call @_FortranASystemClockCountRate() : () -> i64
  ! CHECK: %[[Rate1:.*]] = fir.convert %[[Rate]] : (i64) -> f32
  ! CHECK: fir.store %[[Rate1]] to %[[r]] : !fir.ref<f32>
  ! CHECK: %[[Max:.*]] = fir.call @_FortranASystemClockCountMax() : () -> i64
  ! CHECK: fir.store %[[Max]] to %[[m]] : !fir.ref<i64>
  call system_clock(c, r, m)
! print*, c, r, m
  ! CHECK-NOT: fir.call
  ! CHECK: %[[Rate:.*]] = fir.call @_FortranASystemClockCountRate() : () -> i64
  ! CHECK: fir.store %[[Rate]] to %[[m]] : !fir.ref<i64>
  call system_clock(count_rate=m)
  ! CHECK-NOT: fir.call
! print*, m
end subroutine
