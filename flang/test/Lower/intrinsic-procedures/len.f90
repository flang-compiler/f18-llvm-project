! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: len_test
subroutine len_test(i, c)
  integer :: i
  character(*) :: c
  ! CHECK: %[[c:.*]]:2 = fir.unboxchar %arg1
  ! CHECK: %[[xx:.*]] = fir.convert %[[c]]#1 : (index) -> i32
  ! CHECK: fir.store %[[xx]] to %arg0
  i = len(c)
end subroutine

! CHECK-LABEL: len_test_array
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"}, %[[arg1:.*]]: !fir.boxchar<1> {fir.bindc_name = "c"}
subroutine len_test_array(i, c)
  integer :: i
  character(*) :: c(100)
  ! CHECK: %[[c:.*]]:2 = fir.unboxchar %[[arg1]]
  ! CHECK: %[[xx:.*]] = fir.convert %[[c]]#1 : (index) -> i32
  ! CHECK: fir.store %[[xx]] to %[[arg0]]
  i = len(c)
end subroutine
