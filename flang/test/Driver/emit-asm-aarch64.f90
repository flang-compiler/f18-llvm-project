! RUN: %flang_fc1 -S -triple aarch64-unknown-linux-gnu %s -o - | FileCheck %s

! REQUIRES: aarch64-registered-target

! CHECK-LABEL: _QQmain:
! CHECK-NEXT: .Lfunc_begin0:
! CHECK: ret

end program
