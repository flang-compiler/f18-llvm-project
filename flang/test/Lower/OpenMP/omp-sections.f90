! This test checks the lowering of OpenMP sections construct with several clauses present

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN: FileCheck %s --check-prefix=FIRDialect

!FIRDialect: func @_QQmain() {
!FIRDialect: %[[COUNT:.*]] = fir.alloca i32 {bindc_name = "count", pinned, uniq_name = "_QFEcount"}
!FIRDialect: %[[DOUBLE_COUNT:.*]] = fir.alloca i32 {bindc_name = "double_count", pinned, uniq_name = "_QFEdouble_count"}
!FIRDialect: %[[ETA:.*]] = fir.alloca f32 {bindc_name = "eta", uniq_name = "_QFEeta"}
!FIRDialect: %[[ALLOCATOR:.*]] = arith.constant 1 : i32
!FIRDialect: omp.sections allocate(%[[ALLOCATOR]] : i32 -> %[[ETA]] : !fir.ref<f32>) {
!FIRDialect: omp.section {
!FIRDialect: {{.*}} = arith.constant 5 : i32
!FIRDialect: fir.store {{.*}} to {{.*}} : !fir.ref<i32>
!FIRDialect: {{.*}} = fir.load %[[COUNT]] : !fir.ref<i32>
!FIRDialect: {{.*}} = fir.load %[[DOUBLE_COUNT]] : !fir.ref<i32>
!FIRDialect: {{.*}} = arith.muli {{.*}}, {{.*}} : i32
!FIRDialect: {{.*}} = fir.convert {{.*}} : (i32) -> f32
!FIRDialect: fir.store {{.*}} to %[[ETA]] : !fir.ref<f32>
!FIRDialect: omp.terminator
!FIRDialect: }
!FIRDialect: omp.section {
!FIRDialect: {{.*}} = fir.load {{.*}} : !fir.ref<i32>
!FIRDialect: {{.*}} = arith.constant 1 : i32
!FIRDialect: {{.*}} = arith.addi {{.*}} : i32
!FIRDialect: fir.store {{.*}} to %[[DOUBLE_COUNT]] : !fir.ref<i32>
!FIRDialect: omp.terminator
!FIRDialect: }
!FIRDialect: omp.section {
!FIRDialect: {{.*}} = fir.load {{.*}} : !fir.ref<f32>
!FIRDialect: {{.*}} = arith.constant 7.000000e+00 : f32
!FIRDialect: {{.*}} = arith.subf {{.*}} : f32
!FIRDialect: fir.store {{.*}} to {{.*}} : !fir.ref<f32>
!FIRDialect: {{.*}} = fir.load %[[COUNT]] : !fir.ref<i32>
!FIRDialect: {{.*}} = fir.convert {{.*}} : (i32) -> f32
!FIRDialect: {{.*}} = fir.load %[[ETA]] : !fir.ref<f32>
!FIRDialect: {{.*}} = arith.mulf {{.*}}, {{.*}} : f32
!FIRDialect: {{.*}} = fir.convert {{.*}} : (f32) -> i32
!FIRDialect: fir.store {{.*}} to %[[COUNT]] : !fir.ref<i32>
!FIRDialect: {{.*}} = fir.load %[[COUNT]] : !fir.ref<i32>
!FIRDialect: {{.*}} = fir.convert {{.*}} : (i32) -> f32
!FIRDialect: {{.*}} = fir.load %[[ETA]] : !fir.ref<f32>
!FIRDialect: {{.*}} = arith.subf {{.*}}, {{.*}} : f32
!FIRDialect: {{.*}} = fir.convert {{.*}} : (f32) -> i32
!FIRDialect: fir.store {{.*}} to %[[DOUBLE_COUNT]] : !fir.ref<i32>
!FIRDialect: omp.terminator
!FIRDialect: }
!FIRDialect: omp.terminator
!FIRDialect: }
!FIRDialect: omp.sections nowait {
!FIRDialect: omp.terminator
!FIRDialect: }
!FIRDialect: return
!FIRDialect: }

program sample
    use omp_lib
    integer :: count = 0, double_count = 1 
    !$omp sections private (count, double_count) allocate(omp_default_mem_alloc: eta)  
        !$omp section
            count = 1 + 4
            eta = count * double_count
        !$omp section
            double_count = double_count + 1
        !$omp section
            eta = eta - 7
            count = count * eta
            double_count = count - eta
    !$omp end sections

    !$omp sections
    !$omp end sections nowait
end program sample

!FIRDialect: func @_QPfirstprivate(%[[ARG:.*]]: !fir.ref<f32> {fir.bindc_name = "alpha"}) {
!FIRDialect: %[[ALPHA:.*]] = fir.alloca f32 {bindc_name = "alpha", pinned, uniq_name = "_QFfirstprivateEalpha"}
!FIRDialect: %[[ALPHA_STORE:.*]] = fir.load %[[ARG]] : !fir.ref<f32>
!FIRDialect: fir.store %[[ALPHA_STORE]] to %[[ALPHA]] : !fir.ref<f32>
!FIRDialect: omp.sections {
!FIRDialect: omp.section  {
!FIRDialect: omp.terminator
!FIRDialect: }
!FIRDialect: omp.terminator
!FIRDialect: }
!FIRDialect: omp.sections {
!FIRDialect: omp.section  {
!FIRDialect: %[[PRIVATE_VAR:.*]] = fir.load %[[ARG]] : !fir.ref<f32>
!FIRDialect: %[[CONSTANT:.*]] = arith.constant 5.000000e+00 : f32
!FIRDialect: %[[PRIVATE_VAR_2:.*]] = arith.mulf %[[PRIVATE_VAR]], %[[CONSTANT]] : f32
!FIRDialect: fir.store %[[PRIVATE_VAR_2]] to %[[ARG]] : !fir.ref<f32>
!FIRDialect: omp.terminator
!FIRDialect: }
!FIRDialect: omp.terminator
!FIRDialect: }
!FIRDialect: return
!FIRDialect: }

subroutine firstprivate(alpha)
    real :: alpha 
    !$omp sections firstprivate(alpha)
    !$omp end sections

    !$omp sections
        alpha = alpha * 5
    !$omp end sections
end subroutine
