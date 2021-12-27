! This test checks the type of "depend_vec" for lowering OpenMP Ordered
! Directive with Depend Clause. OpenMP runtime requires i64 type.

! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s --check-prefix=FIRDialect

subroutine ordered(N)
        integer(kind=1) :: i1, i1_lb, i1_ub, i1_s
        integer(kind=2) :: i2, i2_lb, i2_ub, i2_s
        integer(kind=4) :: i4, i4_lb, i4_ub, i4_s
        integer(kind=8) :: i8, i8_lb, i8_ub, i8_s
        integer(kind=16) :: i16, i16_lb, i16_ub, i16_s
        real, dimension(N) :: B, C
        real, external :: foo

!FIRDialect:  omp.wsloop ([[TMP1:%.*]]) : i32
!FIRDialect:    [[TMP2:%.*]] = fir.convert [[TMP1]] : (i32) -> i64
!FIRDialect:    omp.ordered depend_type("dependsource") depend_vec([[TMP2]] : i64) {num_loops_val = 1 : i64}
!$OMP DO ORDERED(1)
        do i1 = i1_lb, i1_ub, i1_s
!$OMP ORDERED DEPEND(SOURCE)
          C(i) = foo(B(i))
        end do
!$OMP END DO

!FIRDialect:  omp.wsloop ([[TMP21:%.*]]) : i32
!FIRDialect:    [[TMP22:%.*]] = fir.convert [[TMP21]] : (i32) -> i64
!FIRDialect:    [[TMP23:%.*]] = arith.constant 1 : i32
!FIRDialect:    [[TMP24:%.*]] = fir.convert [[TMP23]] : (i32) -> i64
!FIRDialect:    [[TMP25:%.*]] = arith.subi [[TMP22]], [[TMP24]] : i64
!FIRDialect:    omp.ordered depend_type("dependsink") depend_vec([[TMP25]] : i64) {num_loops_val = 1 : i64}
!$OMP DO ORDERED(1)
        do i2 = i2_lb, i2_ub, i2_s
!$OMP ORDERED DEPEND(SINK: i2 - 1)
          C(i) = foo(B(i2 - 1))
        end do
!$OMP END DO

!FIRDialect:  omp.wsloop ([[TMP41:%.*]]) : i32
!FIRDialect:    [[TMP42:%.*]] = fir.convert [[TMP41]] : (i32) -> i64
!FIRDialect:    [[TMP43:%.*]] = arith.constant 1 : i32
!FIRDialect:    [[TMP44:%.*]] = fir.convert [[TMP43]] : (i32) -> i64
!FIRDialect:    [[TMP45:%.*]] = arith.subi [[TMP42]], [[TMP44]] : i64
!FIRDialect:    omp.ordered depend_type("dependsink") depend_vec([[TMP45]] : i64) {num_loops_val = 1 : i64}
!$OMP DO ORDERED(1)
        do i4 = i4_lb, i4_ub, i4_s
!$OMP ORDERED DEPEND(SINK: i4 - 1)
          C(i) = foo(B(i))
        end do
!$OMP END DO

!FIRDialect:  omp.wsloop ([[TMP81:%.*]]) : i64
!FIRDialect:    omp.ordered depend_type("dependsource") depend_vec([[TMP81]] : i64) {num_loops_val = 1 : i64}
!$OMP DO ORDERED(1)
        do i8 = i8_lb, i8_ub, i8_s
!$OMP ORDERED DEPEND(SOURCE)
          C(i) = foo(B(i))
        end do
!$OMP END DO

!FIRDialect:  omp.wsloop ([[TMP161:%.*]]) : i64
!FIRDialect:    omp.ordered depend_type("dependsource") depend_vec([[TMP161]] : i64) {num_loops_val = 1 : i64}
!$OMP DO ORDERED(1)
        do i16 = i16_lb, i16_ub, i16_s
!$OMP ORDERED DEPEND(SOURCE)
          C(i) = foo(B(i))
        end do
!$OMP END DO

end
