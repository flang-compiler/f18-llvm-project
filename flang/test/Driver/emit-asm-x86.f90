! RUN: %flang_fc1 -S -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s

! REQUIRES: x86-registered-target

! CHECK-LABEL: _QQmain:
! CHECK-NEXT: .Lfunc_begin0:
! CHECK: ret

end program
