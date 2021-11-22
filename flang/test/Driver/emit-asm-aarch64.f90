! TODO Add `-target <arch>` once that's available

! RUN: %flang_fc1 -S %s -o - | FileCheck %s

! REQUIRES: aarch64-registered-target

! CHECK-LABEL: _QQmain:
! CHECK-NEXT: .Lfunc_begin0:
! CHECK: ret

end program
