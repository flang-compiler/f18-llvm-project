! This test checks lowering of OpenMP DO Directive(Worksharing).

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   tco --disable-llvm --print-ir-after=fir-to-llvm-ir 2>&1 | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   tco | FileCheck %s --check-prefix=LLVMIR

program wsloop_dynamic
        integer :: i
!FIRDialect: func @_QQmain()
!LLVMIRDialect: func @_QQmain()

!LLVMIR: define void @_QQmain()
!LLVMIR:call i32 @__kmpc_global_thread_num{{.*}}
!LLVMIR:  br label %omp_parallel

!$OMP PARALLEL
!FIRDialect-LABLEL:  omp.parallel {
!LLVMIRDialect-LABLEL:  omp.parallel {

!LLVMIR: omp_parallel:                                     ; preds = %0
!LLVMIR:   @__kmpc_fork_call
!$OMP DO SCHEDULE(dynamic)
!FIRDialect:     %[[WS_LB:.*]] = arith.constant 1 : i32
!FIRDialect:     %[[WS_UB:.*]] = arith.constant 9 : i32
!FIRDialect:     %[[WS_STEP:.*]] = arith.constant 1 : i32
!FIRDialect:     omp.wsloop (%[[I:.*]]) : i32 = (%[[WS_LB]]) to (%[[WS_UB]]) inclusive step (%[[WS_STEP]]) schedule(dynamic) nowait
!FIRDialect:       fir.store %[[I]] to %[[STORE:.*]] : !fir.ref<i32>

!LLVMIRDialect-DAG:  %[[WS_UB:.*]] = llvm.mlir.constant(9 : i32) : i32
!LLVMIRDialect-DAG:  %[[WS_LB_STEP:.*]] = llvm.mlir.constant(1 : i32) : i32
!LLVMIRDialect:  omp.wsloop (%[[I:.*]]) : i32 = (%[[WS_LB_STEP]]) to (%[[WS_UB]]) inclusive step (%[[WS_LB_STEP]]) schedule(dynamic) nowait

!LLVMIR:  define internal void @_QQmain..omp_par
!LLVMIR:  omp.par.entry:
!LLVMIR:    br label %omp.par.region
!LLVMIR:  omp.par.region:                                   ; preds = %omp.par.entry
!LLVMIR:    br label %omp.par.region1
!LLVMIR:  omp.par.region1:                                  ; preds = %omp.par.region
!LLVMIR:    br label %omp_loop.preheader
!LLVMIR:  omp_loop.preheader:                               ; preds = %omp.par.region1
!LLVMIR:    @__kmpc_global_thread_num
!LLVMIR:    @__kmpc_dispatch_init_4u(%struct.ident_t* @{{.*}}, i32 %omp_global_thread_num{{.*}}, i32 1073741859, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
!LLVMIR:    br label %omp_loop.preheader.outer.cond
!LLVMIR:  omp_loop.preheader.outer.cond:
!LLVMIR:    @__kmpc_dispatch_next_4u
!LLVMIR:    %{{.*}} = icmp ne i32 %{{.*}}, 0
!LLVMIR:    %{{.*}} = load i32, i32* %p.lowerbound, align 4
!LLVMIR:    %{{.*}} = sub i32 %{{.*}}, 1
!LLVMIR:    br i1 %{{.*}}, label %omp_loop.header, label %omp_loop.exit
!LLVMIR:  omp_loop.exit:                                  ; preds = %omp_loop.preheader.outer.cond
!LLVMIR:   br label %omp_loop.after
!LLVMIR:  omp_loop.header:                                  ; preds = %omp_loop.preheader.outer.cond, %omp_loop.inc
!LLVMIR:    %omp_loop.iv = phi i32 [ %lb, %omp_loop.preheader.outer.cond ], [ %omp_loop.next, %omp_loop.inc ]

do i=1, 9
print*, i
!FIRDialect:    %[[RTBEGIN:.*]] = fir.call @_FortranAioBeginExternalListOutput
!FIRDialect:    %[[LOAD:.*]] = fir.load %[[STORE]] : !fir.ref<i32>
!FIRDialect:    fir.call @_FortranAioOutputInteger32(%[[RTBEGIN]], %[[LOAD]]) : (!fir.ref<i8>, i32) -> i1
!FIRDialect:    fir.call @_FortranAioEndIoStatement(%[[RTBEGIN]]) : (!fir.ref<i8>) -> i32


!LLVMIRDialect:     llvm.call @_FortranAioBeginExternalListOutput(%{{.*}}, %{{.*}}, %{{.*}}) : (i32, !llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
!LLVMIRDialect:     llvm.call @_FortranAioOutputInteger32(%{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, i32) -> i1
!LLVMIRDialect:     llvm.call @_FortranAioEndIoStatement(%{{.*}}) : (!llvm.ptr<i8>) -> i32

!LLVMIR:   br label %omp_loop.cond
!LLVMIR: omp_loop.cond:                                    ; preds = %omp_loop.header
!LLVMIR:   %{{.*}} = load i32, i32* %{{.*}}, align 4
!LLVMIR:   %omp_loop.cmp = icmp ult i32 %{{.*}}, %{{.*}}
!LLVMIR:   br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.preheader.outer.cond
!LLVMIR: omp_loop.body:                                    ; preds = %omp_loop.cond
!LLVMIR:   %{{.*}} = mul i32 %{{.*}}, 1
!LLVMIR:   %{{.*}} = add i32 %{{.*}}, 1
!LLVMIR:   br label %omp.wsloop.region
!LLVMIR: omp.wsloop.region:                                ; preds = %omp_loop.body
!LLVMIR:   %{{.*}} = call i8* @_FortranAioBeginExternalListOutput
!LLVMIR:   %{{.*}} = call i1 @_FortranAioOutputInteger32
!LLVMIR:   %{{.*}} = call i32 @_FortranAioEndIoStatement

end do
!FIRDialect:       omp.yield
!FIRDialect:         }
!FIRDialect:       omp.terminator
!FIRDialect:     }

!LLVMIRDialect:    omp.yield
!LLVMIRDialect:      }
!LLVMIRDialect:    omp.terminator
!LLVMIRDialect:  }
!LLVMIRDialect:  llvm.return
!LLVMIRDialect: }
!$OMP END DO NOWAIT
!$OMP END PARALLEL
end
