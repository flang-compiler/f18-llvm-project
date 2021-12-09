!  This test was adapted from clang/test/Driver/target.c


! RUN: %flang --target=unknown-unknown-unknown -c %s \
! RUN:   -o %t.o -### 2>&1 | FileCheck %s

! Ensure we get a crazy triple here as we asked for one.
! CHECK: Target: unknown-unknown-unknown
