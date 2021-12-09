! Test that -print-target-triple prints correct triple. This test was adapted
! from clang/test/Driver/print-target-triple.c.

! RUN: %flang -print-target-triple 2>&1 \
! RUN:     --target=aarch64-linux-gnu \
! RUN:   | FileCheck --check-prefix=AARCH64 %s
! AARCH64: aarch64-unknown-linux-gnu

! RUN: %flang -print-target-triple 2>&1 \
! RUN:     --target=x86_64-linux-gnu \
! RUN:   | FileCheck --check-prefix=X86_64 %s
! X86_64: x86_64-unknown-linux-gnu
