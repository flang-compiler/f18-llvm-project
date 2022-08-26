! RUN: %python %S/test_symbols.py %s %flang_fc1 -fopenmp
! XFAIL: *
!Checking for name collision of a critical construct with other elements of a program

!DEF: /check_symbols MainProgram
program check_symbols
    !DEF: /check_symbols/i ObjectEntity INTEGER(4)
    integer i, bar 
    !REF: /check_symbols/i
    foo: do i = 1, 10 
    end do foo
    
    !$omp critical (foo)
    !$omp end critical (foo)

    !$omp critical (bar)
    !$omp end critical (bar)
end program check_symbols
