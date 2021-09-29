! This test checks lowering of OpenMP `FIRSTPRIVATE` clause for arrays.

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

!FIRDialect: func @_QPfirstprivate_arrays(%[[ARG1:.*]]: !fir.boxchar<1>, %[[ARG2:.*]]: !fir.ref<!fir.array<10xi32>>) {
!FIRDialect-DAG: %[[ARG1_UNBOX:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
!FIRDialect-DAG: %[[FIVE:.*]] = arith.constant 5 : index
!FIRDialect-DAG: omp.parallel {
!FIRDialect-DAG:   %[[ARG1_PVT:.*]] = fir.alloca !fir.array<5x!fir.char<1>> {bindc_name = "arg1", pinned, uniq_name = "_QFfirstprivate_arraysEarg1"}
!FIRDialect-DAG    %[[SHAPE1:.*]] = fir.shape %c5 : (index) -> !fir.shape<1>
!FIRDialect-DAG    %[[ARG1_INIT_VAL:.*]] = fir.array_load %[[ARG1_PVT]](%[[SHAPE1]]) : (!fir.ref<!fir.array<5x!fir.char<1>>>, !fir.shape<1>) -> !fir.array<5x!fir.char<1>>
!FIRDialect-DAG    %[[ARG1_PVT_UPDATES:.*]] = fir.do_loop %arg2 = %c0 to %7 step %c1 unordered iter_args(%arg3 = %4) -> (!fir.array<5x!fir.char<1>>) {
!FIRDialect-DAG    {{.*}}fir.array_fetch{{.*}}
!FIRDialect-DAG    {{.*}}fir.array_update{{.*}}
!FIRDialect-DAG      fir.result{{.*}}
!FIRDialect-DAG    }
!FIRDialect-DAG    fir.array_merge_store %[[ARG1_INIT_VAL]], %[[ARG1_PVT_UPDATES]] to %[[ARG1_PVT]] : !fir.array<5x!fir.char<1>>, !fir.array<5x!fir.char<1>>, !fir.ref<!fir.array<5x!fir.char<1>>>

!FIRDialect: %[[ARG2_PVT:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "arg2", pinned, uniq_name = "{{.*}}Earg2"}
!FIRDialect-DAG: %[[SHAPE2:.*]] = fir.shape %c10 : (index) -> !fir.shape<1>
!FIRDialect-DAG: %[[ARG2_INIT_VAL:.*]] = fir.array_load %[[ARG2_PVT:.*]](%[[SHAPE2]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
!FIRDialect-DAG: %[[ARG2_PVT_UPDATES:.*]] = fir.do_loop {{.*}} unordered
!FIRDialect-DAG: {{.*}}fir.array_fetch{{.*}}
!FIRDialect-DAG: {{.*}}fir.array_update{{.*}}
!FIRDialect-DAG: fir.result{{.*}}
!FIRDialect-DAG: }
!FIRDialect-DAG: fir.array_merge_store %[[ARG2_INIT_VAL]], %[[ARG2_PVT_UPDATES]] to %[[ARG2_PVT]] : !fir.array<10xi32>, !fir.array<10xi32>, !fir.ref<!fir.array<10xi32>>
!FIRDialect-DAG: omp.terminator
!FIRDialect-DAG: }

subroutine firstprivate_arrays(arg1, arg2)
        character :: arg1(5)
        integer :: arg2(10)

!$OMP PARALLEL FIRSTPRIVATE(arg1, arg2)
        print *, arg1, arg2
!$OMP END PARALLEL

end subroutine
