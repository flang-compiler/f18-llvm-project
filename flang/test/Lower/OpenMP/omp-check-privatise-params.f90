! This test checks whether privatisation uses the correct parameters.

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

!FIRDialect: func @_QParray(%{{.*}}: !fir.ref<!fir.array<?xi32>>{{.*}}, %[[ARG1:.*]]: !fir.ref<i32>{{.*}}) {
!FIRDialect-DAG:  %[[N:.*]] = fir.load %[[ARG1]] : !fir.ref<i32>
!FIRDialect-DAG:  %[[N_CVT1:.*]] = fir.convert %[[N]] : (i32) -> i64
!FIRDialect-DAG:  %[[N_CVT2:.*]] = fir.convert %[[N_CVT1]] : (i64) -> index
!FIRDialect-DAG:  %[[C0:.*]] = arith.constant 0 : index 
!FIRDialect-DAG:  %[[CMP:.*]] = arith.cmpi sgt, %[[N_CVT2]], %[[C0]] : index 
!FIRDialect-DAG:  %[[EXTENT:.*]] = select %[[CMP]], %[[N_CVT2]], %[[C0]] : index 
!FIRDialect-DAG:  omp.parallel {
!FIRDialect-DAG:  {{.*}} = fir.alloca !fir.array<?xi32>, %[[EXTENT]] {{{.*}}, pinned, uniq_name = "_QFarrayEx"}
!FIRDialect:    omp.terminator

subroutine array(x,n)
integer :: i
integer :: x(n)
n = 5
!$omp parallel private(x)
x = i
!$omp end parallel
print *, x
end subroutine
