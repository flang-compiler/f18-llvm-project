! This test checks lowering of OpenMP Ordered Directive with Depend Clause.

! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   tco --disable-llvm --print-ir-after=fir-to-llvm-ir 2>&1 | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | tco | FileCheck %s --check-prefix=LLVMIR

subroutine ordered(N, M)
        integer :: i, j, k, N, M
        real, dimension(M, N) :: A, B, C
        real, external :: foo, bar

!LLVMIR: [[ADDR28:%.*]] = alloca [2 x i64], align 8, !dbg !{{.*}}
!LLVMIR: [[ADDR26:%.*]] = alloca [2 x i64], align 8, !dbg !{{.*}}
!LLVMIR: [[ADDR24:%.*]] = alloca [2 x i64], align 8, !dbg !{{.*}}
!LLVMIR: [[ADDR4:%.*]] = alloca [1 x i64], align 8, !dbg !{{.*}}
!LLVMIR: [[ADDR:%.*]] = alloca [1 x i64], align 8, !dbg !{{.*}}

!$OMP DO ORDERED(1)
        do j = 2, N
          do i = 1, M
!FIRDialect: [[TMP1:%.*]] = fir.convert %{{.*}} : (i32) -> i64
!FIRDialect: [[TMP2:%.*]] = arith.constant 1 : i32
!FIRDialect: [[TMP3:%.*]] = fir.convert [[TMP2:%.*]] : (i32) -> i64
!FIRDialect: [[TMP4:%.*]] = arith.subi [[TMP1:%.*]], [[TMP3:%.*]] : i64
!FIRDialect: omp.ordered depend_type("dependsink") depend_vec([[TMP4]] : i64) {num_loops_val = 1 : i64}

!LLVMIRDialect: [[NTMP1:%.*]] = llvm.sext %{{.*}} : i32 to i64
!LLVMIRDialect: [[NTMP2:%.*]] = llvm.sext %{{.*}} : i32 to i64
!LLVMIRDialect: [[NTMP3:%.*]] = llvm.sub [[NTMP1:%.*]], [[NTMP2:%.*]]  : i64
!LLVMIRDialect: omp.ordered depend_type("dependsink") depend_vec([[NTMP3]] : i64) {num_loops_val = 1 : i64}

!LLVMIR: [[MTMP1:%.*]] = sext i32 %{{.*}} to i64, !dbg !{{.*}}
!LLVMIR: [[MTMP2:%.*]] = sub i64 [[MTMP1:%.*]], 1, !dbg !{{.*}}
!LLVMIR: [[MTMP3:%.*]] = getelementptr inbounds [1 x i64], [1 x i64]* [[ADDR]], i64 0, i64 0, !dbg !{{.*}}
!LLVMIR: store i64 [[NTMP2:%.*]], i64* [[MTMP3]]
!LLVMIR: [[MTMP4:%.*]] = getelementptr inbounds [1 x i64], [1 x i64]* [[ADDR]], i64 0, i64 0, !dbg !{{.*}}
!LLVMIR: [[MTMP5:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB1:[0-9]+]]), !dbg !{{.*}}
!LLVMIR: call void @__kmpc_doacross_wait(%struct.ident_t* @[[GLOB1]], i32 [[MTMP5]], i64* [[MTMP4]]), !dbg !{{.*}}
!$OMP ORDERED DEPEND(SINK: j - 1)
            B(i, j) = foo(A(i, j), B(i, j - 1))
!FIRDialect: [[TMP5:%.*]] = fir.convert %{{.*}} : (i32) -> i64
!FIRDialect: omp.ordered depend_type("dependsource") depend_vec([[TMP5:%.*]] : i64) {num_loops_val = 1 : i64}

!LLVMIRDialect: omp.ordered depend_type("dependsource") depend_vec([[NTMP2:%.*]] : i64) {num_loops_val = 1 : i64}

!LLVMIR: [[MTMP6:%.*]] = getelementptr inbounds [1 x i64], [1 x i64]* [[ADDR4]], i64 0, i64 0, !dbg !{{.*}}
!LLVMIR: store i64 [[MTMP1:%.*]], i64* [[MTMP6:%.*]]
!LLVMIR: [[MTMP7:%.*]] = getelementptr inbounds [1 x i64], [1 x i64]* [[ADDR4]], i64 0, i64 0, !dbg !{{.*}}
!LLVMIR: [[MTMP8:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB3:[0-9]+]]), !dbg !{{.*}}
!LLVMIR: call void @__kmpc_doacross_post(%struct.ident_t* @[[GLOB3]], i32 [[MTMP8]], i64* [[MTMP7]]), !dbg !{{.*}}
!$OMP ORDERED DEPEND(SOURCE)
            C(i, j) = bar(B(i, j))
          end do
        end do
!$OMP END DO

!$OMP DO ORDERED(2)
        do j = N - 1, 1
          do i = M - 1, 1
!FIRDialect: [[TMP8:%.*]] = fir.convert %{{.*}} : (i32) -> i64
!FIRDialect: [[TMP9:%.*]] = arith.constant 1 : i32
!FIRDialect: [[TMP10:%.*]] = fir.convert [[TMP9:%.*]] : (i32) -> i64
!FIRDialect: [[TMP11:%.*]] = arith.addi [[TMP8:%.*]], [[TMP10:%.*]] : i64
!FIRDialect: [[TMP12:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
!FIRDialect: [[TMP13:%.*]] = fir.convert [[TMP12:%.*]] : (i32) -> i64
!FIRDialect: [[TMP14:%.*]] = fir.convert %{{.*}} : (i32) -> i64
!FIRDialect: [[TMP15:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
!FIRDialect: [[TMP16:%.*]] = fir.convert [[TMP15:%.*]] : (i32) -> i64
!FIRDialect: [[TMP17:%.*]] = arith.constant 1 : i32
!FIRDialect: [[TMP18:%.*]] = fir.convert [[TMP17:%.*]] : (i32) -> i64
!FIRDialect: [[TMP19:%.*]] = arith.addi [[TMP16:%.*]], [[TMP18:%.*]] : i64
!FIRDialect: omp.ordered depend_type("dependsink") depend_vec([[TMP11]], [[TMP13]], [[TMP14]], [[TMP19]] : i64, i64, i64, i64) {num_loops_val = 2 : i64}

!LLVMIRDialect: [[NTMP4:%.*]] = llvm.sext %{{.*}} : i32 to i64
!LLVMIRDialect: [[NTMP5:%.*]] = llvm.sext %{{.*}} : i32 to i64
!LLVMIRDialect: [[NTMP6:%.*]] = llvm.add [[NTMP4:%.*]], [[NTMP5:%.*]]  : i64
!LLVMIRDialect: [[NTMP7:%.*]] = llvm.load %{{.*}} : !llvm.ptr<i32>
!LLVMIRDialect: [[NTMP8:%.*]] = llvm.sext [[NTMP7:%.*]] : i32 to i64
!LLVMIRDialect: [[NTMP9:%.*]] = llvm.add [[NTMP8:%.*]], [[NTMP5:%.*]]  : i64
!LLVMIRDialect: omp.ordered depend_type("dependsink") depend_vec([[NTMP6:%.*]], [[NTMP8:%.*]], [[NTMP4:%.*]], [[NTMP9:%.*]] : i64, i64, i64, i64) {num_loops_val = 2 : i64}

!LLVMIR: [[MTMP9:%.*]] = sext i32 {{.*}} to i64, !dbg !{{.*}}
!LLVMIR: [[MTMP10:%.*]] = add i64 [[MTMP9:%.*]], 1, !dbg !{{.*}}
!LLVMIR: [[MTMP11:%.*]] = load i32, i32* {{.*}}, align 4, !dbg !{{.*}}
!LLVMIR: [[MTMP12:%.*]] = sext i32 [[MTMP11:%.*]] to i64, !dbg !{{.*}}
!LLVMIR: [[MTMP13:%.*]] = add i64 [[MTMP12:%.*]], 1, !dbg !{{.*}}
!LLVMIR: [[MTMP14:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[ADDR24]], i64 0, i64 0, !dbg !{{.*}}
!LLVMIR: store i64 [[MTMP9:%.*]], i64* [[MTMP14:%.*]]
!LLVMIR: [[MTMP15:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[ADDR24]], i64 0, i64 1, !dbg !{{.*}}
!LLVMIR: store i64 [[MTMP12:%.*]], i64* [[MTMP15:%.*]]
!LLVMIR: [[MTMP16:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[ADDR24]], i64 0, i64 0, !dbg !{{.*}}
!LLVMIR: [[MTMP17:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB8:[0-9]+]]), !dbg !{{.*}}
!LLVMIR: call void @__kmpc_doacross_wait(%struct.ident_t* @[[GLOB8]], i32 [[MTMP17]], i64* [[MTMP16]]), !dbg !{{.*}}
!LLVMIR: [[MTMP18:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[ADDR26]], i64 0, i64 0, !dbg !{{.*}}
!LLVMIR: store i64 [[MTMP9:%.*]], i64* [[MTMP21:%.*]]
!LLVMIR: [[MTMP19:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[ADDR26]], i64 0, i64 1, !dbg !{{.*}}
!LLVMIR: store i64 [[MTMP13:%.*]], i64* [[MTMP22:%.*]]
!LLVMIR: [[MTMP20:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[ADDR26]], i64 0, i64 0, !dbg !{{.*}}
!LLVMIR: [[MTMP21:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB8]]), !dbg !{{.*}}
!LLVMIR: call void @__kmpc_doacross_wait(%struct.ident_t* @[[GLOB8]], i32 [[MTMP21]], i64* [[MTMP20]]), !dbg !{{.*}}
!$OMP ORDERED DEPEND(SINK: j + 1, i) DEPEND(SINK: j, i + 1)
            B(i, j) = foo(A(i, j), B(i + 1, j), B(i, j + 1))
!FIRDialect: [[TMP20:%.*]] = fir.convert %{{.*}} : (i32) -> i64
!FIRDialect: [[TMP21:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
!FIRDialect: [[TMP22:%.*]] = fir.convert [[TMP21:%.*]] : (i32) -> i64
!FIRDialect: omp.ordered depend_type("dependsource") depend_vec([[TMP20:%.*]], [[TMP22:%.*]] : i64, i64) {num_loops_val = 2 : i64}

!LLVMIRDialect: [[NTMP10:%.*]] = llvm.load %{{.*}} : !llvm.ptr<i32>
!LLVMIRDialect: [[NTMP11:%.*]] = llvm.sext [[NTMP10:%.*]] : i32 to i64
!LLVMIRDialect: omp.ordered depend_type("dependsource") depend_vec([[NTMP4:%.*]], [[NTMP11:%.*]] : i64, i64) {num_loops_val = 2 : i64}

!LLVMIR: [[MTMP22:%.*]] = load i32, i32* {{.*}}, align 4, !dbg !{{.*}}
!LLVMIR: [[MTMP23:%.*]] = sext i32 [[MTMP22:%.*]] to i64, !dbg !{{.*}}
!LLVMIR: [[MTMP24:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[ADDR28]], i64 0, i64 0, !dbg !{{.*}}
!LLVMIR: store i64 [[MTMP9:%.*]], i64* [[MTMP24:%.*]]
!LLVMIR: [[MTMP25:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[ADDR28]], i64 0, i64 1, !dbg !{{.*}}
!LLVMIR: store i64 [[MTMP27:%.*]], i64* [[MTMP25:%.*]]
!LLVMIR: [[MTMP26:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[ADDR28]], i64 0, i64 0, !dbg !{{.*}}
!LLVMIR: [[MTMP27:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB10:[0-9]+]]), !dbg !{{.*}}
!LLVMIR: call void @__kmpc_doacross_post(%struct.ident_t* @[[GLOB10]], i32 [[MTMP27]], i64* [[MTMP26]]), !dbg !{{.*}}
!$OMP ORDERED DEPEND(SOURCE)
            C(i, j) = bar(B(i, j))
          end do
        end do
!$OMP END DO

end
