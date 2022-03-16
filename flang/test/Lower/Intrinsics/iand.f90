! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: iand_test
subroutine iand_test(a, b)
  integer :: a, b
  print *, iand(a, b)
  ! CHECK: %{{[0-9]+}} = arith.andi %{{[0-9]+}}, %{{[0-9]+}} : i{{(8|16|32|64|128)}}
end subroutine iand_test

