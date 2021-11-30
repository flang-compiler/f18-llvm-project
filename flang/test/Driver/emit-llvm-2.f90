! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck %s

! Extracted from: https://github.com/flang-compiler/f18-llvm-project/issues/1280

! CHECK: ; ModuleID = 'FIRModule'

MODULE test_module
    REAL(8_4), ALLOCATABLE, DIMENSION(:) :: mu
    CONTAINS
    SUBROUTINE test_subr ( istat )

        INTEGER(4_4), INTENT(INOUT) :: istat
        IF ( istat > 0 ) RETURN
        w = zero

    END SUBROUTINE test_subr
END MODULE test_module
