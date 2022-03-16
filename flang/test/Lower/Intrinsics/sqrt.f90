! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: sqrt_testr
subroutine sqrt_testr(a, b)
  real :: a, b
  ! CHECK: fir.call {{.*}}sqrt
  b = sqrt(a)
end subroutine

