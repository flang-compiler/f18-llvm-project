! Verify that the Fortran runtime libraries are present in the linker
! invocation. These libraries are added on top of other standard runtime
! libraries that the Clang driver will include.

! NOTE: The additional linker flags tested here are currently specified in
! clang/lib/Driver/Toolchains/Gnu.cpp. This makes the current implementation GNU
! (Linux) specific. The following line will make sure that this test is skipped
! on Windows. Ideally we should find a more robust way of testing this.
! REQUIRES: shell

! RUN: %flang -### %S/Inputs/hello.f90 2>&1 | FileCheck %s

! CHECK: -lFortran_main
! CHECK: -lFortranRuntime
! CHECK: -lFortranDecimal
! CHECK: -lm
