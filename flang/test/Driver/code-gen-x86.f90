! TODO Add `-target <arch>` once that's available

! RUN: rm -f %t.o
! RUN: %flang_fc1 -emit-obj %s -o %t.o
! RUN: llvm-objdump --disassemble-all %t.o | FileCheck %s
! RUN: rm -f %t.o
! RUN: %flang -c %s -o %t.o
! RUN: llvm-objdump --disassemble-all %t.o | FileCheck %s

! REQUIRES: x86-registered-target

! CHECK-LABEL: <_QQmain>:
! CHECK-NEXT:  	retq

end program
