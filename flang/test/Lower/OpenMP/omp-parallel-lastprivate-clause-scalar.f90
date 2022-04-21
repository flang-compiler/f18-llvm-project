! This test checks lowering of `FIRSTPRIVATE` clause for scalar types.

! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s --check-prefix=FIRDialect

!FIRDialect: func @_QPlastprivate_character(%[[ARG1:.*]]: !fir.boxchar<1>{{.*}}) {
!FIRDialect-DAG: %[[ARG1_UNBOX:.*]]:2 = fir.unboxchar
!FIRDialect: omp.parallel {
!FIRDialect-DAG: %[[ARG1_PVT:.*]] = fir.alloca !fir.char<1,5> {bindc_name = "arg1", 
! Check that we are accessing the clone inside the loop
!FIRDialect-DAG: %[[ARG1_PVT_REF:.*]] = fir.convert %[[ARG1_PVT]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>

! Check we are copying back the last iterated value back to the clone before exiting
!FIRDialect-DAG: %[[LOCAL_VAR:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!FIRDialect-DAG: omp.wsloop (%[[INDX_WS:.*]]) : {{.*}} { 
!FIRDialect-DAG: fir.store %[[INDX_WS]] to %[[LOCAL_VAR]] : !fir.ref<i32>
!FIRDialect-DAG: %[[ADDR:.*]] = fir.address_of(@_QQcl.63) : !fir.ref<!fir.char<1>>

! Testing string copy
!FIRDialect-DAG: %[[CVT:.*]] = fir.convert %[[ARG1_UNBOX]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
!FIRDialect-DAG: %[[CVT1:.*]] = fir.convert %[[ARG1_PVT]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
!FIRDialect-DAG: fir.call @llvm.memmove.p0i8.p0i8.i64(%[[CVT]], %[[CVT1]]{{.*}})

!FIRDialect: %[[THIRTY_TWO:.*]] = arith.constant 32 : i8
!FIRDialect-DAG: %[[UNDEF:.*]] = fir.undefined !fir.char<1>
!FIRDialect-DAG: %[[INSERT:.*]] = fir.insert_value %[[UNDEF]], %[[THIRTY_TWO]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
!FIRDialect-DAG: %[[ONE_3:.*]] = arith.constant 1 : index

!FIRDialect: fir.do_loop %[[INDX_WS]] = {{.*}} {
!FIRDialect-DAG: %[[CVT_2:.*]] = fir.convert %[[ARG1_UNBOX]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
!FIRDialect-DAG: %[[COORD:.*]] = fir.coordinate_of %[[CVT_2]], %[[INDX_WS]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
!FIRDialect-DAG: fir.store %[[INSERT]] to %[[COORD]] : !fir.ref<!fir.char<1>>
!FIRDialect-DAG: }


subroutine lastprivate_character(arg1)
        character(5) :: arg1

!$OMP PARALLEL 
!$OMP DO LASTPRIVATE(arg1)
do n = 1, 5
        arg1(n:n) = 'c'
        print *, arg1
end do
!$OMP END DO
!$OMP END PARALLEL

end subroutine

!FIRDialect: func @_QPlastprivate_int(%[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "arg1"}) {
!FIRDialect-DAG: omp.parallel  {
!FIRDialect-DAG: %[[CLONE:.*]] = fir.alloca i32 {bindc_name = "arg1"
!FIRDialect: omp.yield
!FIRDialect: %[[CLONE_LD:.*]] = fir.load %[[CLONE]] : !fir.ref<i32>
!FIRDialect-DAG: fir.store %[[CLONE_LD]] to %[[ARG1]] : !fir.ref<i32>
!FIRDialect-DAG: omp.terminator

subroutine lastprivate_int(arg1)
        integer :: arg1
!$OMP PARALLEL 
!$OMP DO LASTPRIVATE(arg1)
do n = 1, 5
        arg1 = 2
        print *, arg1
end do
!$OMP END DO
!$OMP END PARALLEL
print *, arg1
end subroutine
