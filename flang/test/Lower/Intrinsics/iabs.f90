! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: iabs_test
subroutine iabs_test(a, b)
  integer :: a, b
  ! CHECK: arith.shrsi
  ! CHECK: arith.xori
  ! CHECK: arith.subi
  b = iabs(a)
end subroutine

!Check if the return type (RT) has default kind.
! CHECK-LABEL: iabs_test
subroutine iabs_testRT(a, b)
  integer(KIND=4) :: a
  integer(KIND=16) :: b
  ! CHECK: arith.shrsi
  ! CHECK: arith.xori
  ! CHECK: %[[RT:.*]] =  arith.subi
  ! CHECK: fir.convert %[[RT]] : (i32)
  b = iabs(a)
end subroutine

