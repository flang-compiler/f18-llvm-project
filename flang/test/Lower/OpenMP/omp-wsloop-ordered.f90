! This test checks lowering of worksharing-loop construct with ordered clause.

! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   tco --disable-llvm --print-ir-after=fir-to-llvm-ir 2>&1 | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | tco | FileCheck %s --check-prefix=LLVMIR

! This checks lowering ordered clause specified without parameter
subroutine wsloop_ordered_no_para()
  integer :: a(10), i

! FIRDialect:  omp.wsloop (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) ordered(0)  {
! FIRDialect:    omp.yield
! FIRDialect:  }

! LLVMIRDialect:    omp.wsloop (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) ordered(0)  {
! LLVMIRDialect:      omp.yield
! LLVMIRDialect:    }

! LLVMIR: omp_loop.preheader:
! LLVMIR: [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB0:[0-9]+]])
! LLVMIR-NEXT: call void @__kmpc_dispatch_init_4u(%struct.ident_t* @[[GLOB0]], i32 [[TMP0]],
! LLVMIR: omp_loop.inc:
! LLVMIR: call void @__kmpc_dispatch_fini_4u(%struct.ident_t* @[[GLOB0]], i32 [[TMP0]])
! LLVMIR: omp_loop.preheader.outer.cond:
! LLVMIR: {{.*}} = call i32 @__kmpc_dispatch_next_4u(%struct.ident_t* @[[GLOB0]], i32 [[TMP0]],

  !$omp do ordered
  do i = 2, 10
    !$omp ordered
    a(i) = a(i-1) + 1
    !$omp end ordered
  end do
  !$omp end do

end
