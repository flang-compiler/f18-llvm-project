! This test checks lowering of OpenMP parallel Directive with
! `FIRSTPRIVATE` clause present.

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

!FIRDialect: func @_QPfirstprivate_clause(%[[ARG1:.*]]: !fir.ref<i32>, %[[ARG2:.*]]: !fir.ref<!fir.array<10xi32>>, %[[ARG3:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>>) {
!FIRDialect-DAG:  omp.parallel
!FIRDialect-DAG: %[[ARG1_PVT:.*]] = fir.alloca i32 {{{.*}}uniq_name = "{{.*}}Earg1"}
!FIRDialect-DAG: %[[ARG1_PVT_LOAD:.*]] = fir.load %[[ARG1]] : !fir.ref<i32>
!FIRDialect-DAG: fir.store %[[ARG1_PVT_LOAD]] to %[[ARG1_PVT]] : !fir.ref<i32>
!FIRDialect-DAG: %[[ARG2_PRIVATE:.*]] = fir.alloca !fir.array<10xi32> {{{.*}}uniq_name = "{{.*}}Earg2"}
!FIRDialect-DAG: %[[SHAPE:.*]] = fir.shape %c10 : (index) -> !fir.shape<1>
!FIRDialect-DAG: %[[ARG2_INIT_VAL:.*]] = fir.array_load %[[ARG2_PRIVATE:.*]](%[[SHAPE]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
!FIRDialect-DAG: %[[FIRST_PRIVATE_UPDATES:.*]] = fir.do_loop
!FIRDialect-DAG: {{.*}}fir.array_fetch{{.*}}
!FIRDialect-DAG: {{.*}}fir.array_update{{.*}}
!FIRDialect-DAG: fir.result{{.*}}
!FIRDialect-DAG: }
!FIRDialect-DAG: fir.array_merge_store %[[ARG2_INIT_VAL]], %[[FIRST_PRIVATE_UPDATES]] to %[[ARG2_PRIVATE]] : !fir.array<10xi32>, !fir.array<10xi32>, !fir.ref<!fir.array<10xi32>>
!FIRDialect-DAG: %[[ARG3_BOX:.*]] = fir.load %[[ARG3]] : !fir.ref<!fir.box<!fir.heap<i32>>>
!FIRDialect-DAG: %[[ARG3_ADDR:.*]] = fir.box_addr %[[ARG3_BOX]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
!FIRDialect-DAG: %[[ARG3_PVT:.*]] = fir.alloca i32 {{{.*}}uniq_name = "{{.*}}Earg3"}
!FIRDialect-DAG: %[[ARG3_PVT_LOAD:.*]] = fir.load %[[ARG3_ADDR]] : !fir.heap<i32>
!FIRDialect-DAG: fir.store %[[ARG3_PVT_LOAD]] to %[[ARG3_PVT]] : !fir.ref<i32>
!FIRDialect-DAG: omp.terminator
!FIRDialect-DAG: }

subroutine firstprivate_clause(arg1, arg2, arg3)

        integer :: arg1, arg2(10)
        integer, allocatable :: arg3

!$OMP PARALLEL FIRSTPRIVATE(arg1, arg2, arg3)
        print*, arg1, arg2, arg3
!$OMP END PARALLEL

end subroutine
