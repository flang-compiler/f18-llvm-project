! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck %s

! CHECK: ; ModuleID = 'FIRModule'
! CHECK: define void @_QQmain()
! CHECK-NEXT:  ret void
! CHECK-NEXT: }

end program
