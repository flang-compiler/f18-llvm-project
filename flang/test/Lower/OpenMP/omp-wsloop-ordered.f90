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

! This checks lowering ordered clause specified with a parameter
subroutine wsloop_ordered_with_para()
  integer :: a(10), i

! FIRDialect: func @_QPwsloop_ordered_with_para() {
! FIRDialect:    [[ARG3:%.*]] = arith.constant 1 : i64
! FIRDialect-DAG:    [[ARG1:%.*]] = fir.convert %{{.*}} : (i32) -> i64
! FIRDialect-DAG:    [[ARG2:%.*]] = fir.convert %{{.*}} : (i32) -> i64
! FIRDialect:  omp.wsloop (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) ordered(1) doacross([[ARG1]] : i64, [[ARG2]] : i64, [[ARG3]] : i64)  {
! FIRDialect:    omp.yield
! FIRDialect:  }

! LLVMIRDialect:  llvm.func @_QPwsloop_ordered_with_para() {
! LLVMIRDialect:    [[ARG3:%.*]] = llvm.mlir.constant(1 : i64) : i64
! LLVMIRDialect-DAG:    [[ARG1:%.*]] = llvm.sext %{{.*}} : i32 to i64
! LLVMIRDialect-DAG:    [[ARG2:%.*]] = llvm.sext %{{.*}} : i32 to i64
! LLVMIRDialect:    omp.wsloop (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) ordered(1) doacross([[ARG1]] : i64, [[ARG2]] : i64, [[ARG3]] : i64)  {
! LLVMIRDialect:      omp.yield
! LLVMIRDialect:    }

! LLVMIR: define void @_QPwsloop_ordered_with_para
! LLVMIR: omp_loop.preheader:
! LLVMIR:   [[ADDR:%.*]] = getelementptr inbounds [1 x %kmp_dim], [1 x %kmp_dim]* [[DIMS:%.*]], i64 0, i64 0
! LLVMIR:   [[ST1:%.*]] = getelementptr inbounds %kmp_dim, %kmp_dim* [[ADDR]], i32 0, i32 0
! LLVMIR:   store i64 2, i64* [[ST1]], align 8
! LLVMIR:   [[ST2:%.*]] = getelementptr inbounds %kmp_dim, %kmp_dim* [[ADDR]], i32 0, i32 1
! LLVMIR:   store i64 10, i64* [[ST2]], align 8
! LLVMIR:   [[ST3:%.*]] = getelementptr inbounds %kmp_dim, %kmp_dim* [[ADDR]], i32 0, i32 2
! LLVMIR:   store i64 1, i64* [[ST3]], align 8
! LLVMIR:   [[BASE:%.*]] = getelementptr inbounds [1 x %kmp_dim], [1 x %kmp_dim]* [[DIMS]], i64 0, i64 0
! LLVMIR:   [[INIT_ARG:%.*]] = bitcast %kmp_dim* [[BASE]] to i8*
! LLVMIR:   [[TMP1:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB1:[0-9]+]])
! LLVMIR:   call void @__kmpc_doacross_init(%struct.ident_t* @[[GLOB1]], i32 [[TMP1]], i32 1, i8* [[INIT_ARG]])
! LLVMIR: omp_loop.exit:
! LLVMIR:   call void @__kmpc_doacross_fini(%struct.ident_t* @[[GLOB1]], i32 [[TMP1]])

  !$omp do ordered(1)
  do i = 2, 10
    !!$omp ordered depend(sink: i-1)
    a(i) = a(i-1) + 1
    !!$omp ordered depend(source)
  end do
  !$omp end do

end
