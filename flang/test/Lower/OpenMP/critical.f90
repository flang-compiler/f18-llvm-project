! This test checks lowering of OpenMP Critical Directive.

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp %s -o - | \
! RUN:   tco --disable-llvm --print-ir-after=fir-to-llvm-ir 2>&1 | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   tco | FileCheck %s --check-prefix=LLVMIR


program mn
        use omp_lib
        integer :: x, y
!FIRDialect: omp.critical.declare @help hint(contended)
!LLVMDialect: omp.critical.declare @help hint(contended)
!FIRDialect: omp.critical(@help)
!LLVMIRDialect: omp.critical(@help)
!LLVMIR: call void @__kmpc_critical_with_hint({{.*}}, {{.*}}, {{.*}} @{{.*}}help.var, i32 2)
!$OMP CRITICAL(help) HINT(omp_lock_hint_contended)
        x = x + y
!FIRDialect: omp.terminator
!LLVMIRDialect: omp.terminator
!LLVMIR: call void @__kmpc_end_critical({{.*}}, {{.*}}, {{.*}} @{{.*}}help.var)
!$OMP END CRITICAL(help)

! Test that the same name can be used again
! Also test with the zero hint expression
!FIRDialect: omp.critical(@help)
!LLVMIRDialect: omp.critical(@help)
!LLVMIR: call void @__kmpc_critical_with_hint({{.*}}, {{.*}}, {{.*}} @{{.*}}help.var, i32 2)
!$OMP CRITICAL(help) HINT(omp_lock_hint_none)
        x = x - y
!FIRDialect: omp.terminator
!LLVMIRDialect: omp.terminator
!LLVMIR: call void @__kmpc_end_critical({{.*}}, {{.*}}, {{.*}} @{{.*}}help.var)
!$OMP END CRITICAL(help)

!FIRDialect: omp.critical
!LLVMIRDialect: omp.critical
!LLVMIR: call void @__kmpc_critical({{.*}}, {{.*}}, {{.*}} @{{.*}}_.var)
!$OMP CRITICAL
        y = x + y
!FIRDialect: omp.terminator
!LLVMIRDialect: omp.terminator
!LLVMIR: call void @__kmpc_end_critical({{.*}}, {{.*}}, {{.*}} @{{.*}}_.var)
!$OMP END CRITICAL
end program
