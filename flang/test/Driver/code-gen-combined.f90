! Verify that:
!   * `flang -emit-llvm file.f90` followed by `flang file.ll`
! (and various combinations of this) generates identical output as when using
!   * `flang -c file.f90`.
! Linking is skipped. In both cases it would be a seperate invocation with
! identical input (verified by this test).

!--------------
! Object files
!--------------
! RUN: rm -f %t_default.o %t.bc %t.ll %t_combined.o
! RUN: %flang -c %s -o %t_default.o
! RUN: %flang -c -emit-llvm %s -o %t.bc
! RUN: %flang -S -emit-llvm %s -o %t.ll

! RUN: %flang -c %t.bc -o %t_combined.o
! RUN: diff %t_default.o %t_combined.o

! RUN: %flang -c %t.ll -o %t_combined.o
! RUN: diff %t_default.o %t_combined.o

!----------------
! Assembly files
!----------------
! RUN: rm -f %t_default.s %t.bc %t.ll %t_combined.s
! RUN: %flang -S %s -o %t_default.s
! RUN: %flang -c -emit-llvm %s -o %t.bc
! RUN: %flang -S -emit-llvm %s -o %t.ll

! RUN: %flang -S %t.bc -o %t_combined.s
! RUN: diff %t_default.s %t_combined.s

! RUN: %flang -S %t.ll -o %t_combined.s
! RUN: diff %t_default.s %t_combined.s

!----------------
! Test input
!----------------
end program
