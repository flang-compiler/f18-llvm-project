! This test checks lowering of OpenMP parallel Directive with
! `PRIVATE` clause present.

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

!FIRDialect: func @_QPprivate_clause(%[[ARG1:.*]]: !fir.ref<i32>, %[[ARG2:.*]]: !fir.ref<!fir.array<10xi32>>, %[[ARG3:.*]]: !fir.boxchar<1>, %[[ARG4:.*]]: !fir.boxchar<1>) {
!FIRDialect-DAG: %[[ALPHA:.*]] = fir.alloca i32 {{{.*}}, uniq_name = "{{.*}}Ealpha"}
!FIRDialect-DAG: %[[ALPHA_ARRAY:.*]] = fir.alloca !fir.array<10xi32> {{{.*}}, uniq_name = "{{.*}}Ealpha_array"}
!FIRDialect-DAG: %[[BETA:.*]] = fir.alloca !fir.char<1,5> {{{.*}}, uniq_name = "{{.*}}Ebeta"}
!FIRDialect-DAG: %[[BETA_ARRAY:.*]] = fir.alloca !fir.array<10x!fir.char<1,5>> {{{.*}}, uniq_name = "{{.*}}Ebeta_array"}

!FIRDialect-DAG:  omp.parallel {
!FIRDialect-DAG: %[[ALPHA_PRIVATE:.*]] = fir.alloca i32 {{{.*}}, uniq_name = "{{.*}}Ealpha"}
!FIRDialect-DAG: %[[ALPHA_ARRAY_PRIVATE:.*]] = fir.alloca !fir.array<10xi32> {{{.*}}, uniq_name = "{{.*}}Ealpha_array"}
!FIRDialect-DAG: %[[BETA_PRIVATE:.*]] = fir.alloca !fir.char<1,5> {{{.*}}, uniq_name = "{{.*}}Ebeta"}
!FIRDialect-DAG: %[[BETA_ARRAY_PRIVATE:.*]] = fir.alloca !fir.array<10x!fir.char<1,5>> {{{.*}}, uniq_name = "{{.*}}Ebeta_array"}
!FIRDialect-DAG: %[[ARG1_PRIVATE:.*]] = fir.alloca i32 {{{.*}}, uniq_name = "{{.*}}Earg1"}
!FIRDialect-DAG: %[[ARG2_ARRAY_PRIVATE:.*]] = fir.alloca !fir.array<10xi32> {{{.*}}, uniq_name = "{{.*}}Earg2"}
!FIRDialect-DAG: %[[ARG3_PRIVATE:.*]] = fir.alloca !fir.char<1,5> {{{.*}}, uniq_name = "{{.*}}Earg3"}
!FIRDialect-DAG: %[[ARG4_ARRAY_PRIVATE:.*]] = fir.alloca !fir.array<10x!fir.char<1,5>> {{{.*}}, uniq_name = "{{.*}}Earg4"}
!FIRDialect:    omp.terminator
!FIRDialect:  }

subroutine private_clause(arg1, arg2, arg3, arg4)

        integer :: arg1, arg2(10)
        integer :: alpha, alpha_array(10)
        character(5) :: arg3, arg4(10)
        character(5) :: beta, beta_array(10)

!$OMP PARALLEL PRIVATE(alpha, alpha_array, beta, beta_array, arg1, arg2, arg3, arg4)
        alpha = 1
        alpha_array = 4
        beta = "hi"
        beta_array = "hi"
        arg1 = 2
        arg2 = 3
        arg3 = "world"
        arg4 = "world"
!$OMP END PARALLEL

end subroutine
