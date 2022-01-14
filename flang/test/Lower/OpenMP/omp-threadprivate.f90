! This test checks lowering of OpenMP threadprivate Directive.
! XFAIL: *
! RUN: %bbc -fopenmp -emit-fir %s -o  - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

module mod1
  integer :: z
  !$omp threadprivate(z)
end

program main
  integer, save :: x, y

  !$omp threadprivate(x, y)
end
