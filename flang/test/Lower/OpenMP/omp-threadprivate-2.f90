! This test checks lowering of OpenMP Threadprivate Directive.
! Test for real, logical, complex, and derived type.

! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | tco | FileCheck %s --check-prefix=LLVMIR

module test
  type my_type
    integer :: t_i
    real :: t_arr(5)
  end type my_type
  real :: x
  complex :: y
  logical :: z
  type(my_type) :: t

  !$omp threadprivate(x, y, z, t)

!FIRDialect-DAG: fir.global @_QMtestEt : !fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}> {
!FIRDialect-DAG: fir.global @_QMtestEx : f32 {
!FIRDialect-DAG: fir.global @_QMtestEy : !fir.complex<4> {
!FIRDialect-DAG: fir.global @_QMtestEz : !fir.logical<4> {

!LLVMIR-DAG: %_QMtestTmy_type = type { i32, [5 x float] }
!LLVMIR-DAG: @_QMtestEt = global %_QMtestTmy_type undef
!LLVMIR-DAG: @_QMtestEx = global float undef
!LLVMIR-DAG: @_QMtestEy = global { float, float } undef
!LLVMIR-DAG: @_QMtestEz = global i32 undef

!LLVMIR-DAG: @_QMtestEt.cache = common global i8** null
!LLVMIR-DAG: @_QMtestEx.cache = common global i8** null
!LLVMIR-DAG: @_QMtestEy.cache = common global i8** null
!LLVMIR-DAG: @_QMtestEz.cache = common global i8** null

contains
  subroutine sub()
!FIRDialect-DAG:  [[ADDR0:%.*]] = fir.address_of(@_QMtestEt) : !fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>>
!FIRDialect-DAG:  [[NEWADDR0:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>> -> !fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>>
!FIRDialect-DAG:  [[ADDR1:%.*]] = fir.address_of(@_QMtestEx) : !fir.ref<f32>
!FIRDialect-DAG:  [[NEWADDR1:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<f32> -> !fir.ref<f32>
!FIRDialect-DAG:  [[ADDR2:%.*]] = fir.address_of(@_QMtestEy) : !fir.ref<!fir.complex<4>>
!FIRDialect-DAG:  [[NEWADDR2:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<!fir.complex<4>> -> !fir.ref<!fir.complex<4>>
!FIRDialect-DAG:  [[ADDR3:%.*]] = fir.address_of(@_QMtestEz) : !fir.ref<!fir.logical<4>>
!FIRDialect-DAG:  [[NEWADDR3:%.*]] = omp.threadprivate [[ADDR3]] : !fir.ref<!fir.logical<4>> -> !fir.ref<!fir.logical<4>>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR1]] : !fir.ref<f32>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR2]] : !fir.ref<!fir.complex<4>>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR3]] : !fir.ref<!fir.logical<4>>
!FIRDialect-DAG:  %{{.*}} = fir.coordinate_of [[NEWADDR0]]

! CHECK-LABEL: @_QMtestPsub()
!LLVMIR:  [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB0:[0-9]+]])
!LLVMIR-DAG:  [[CACHE0:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB0:[0-9]+]], i32 [[TMP0:%.*]], i8* bitcast (%_QMtestTmy_type* @_QMtestEt to i8*), i64 24, i8*** @_QMtestEt.cache)
!LLVMIR-DAG:  [[LOAD0:%.*]] = bitcast i8* [[CACHE0]] to %_QMtestTmy_type*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP1:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB1:[0-9]+]])
!LLVMIR-DAG:  [[CACHE1:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB1:[0-9]+]], i32 [[TMP1:%.*]], i8* bitcast (float* @_QMtestEx to i8*), i64 4, i8*** @_QMtestEx.cache)
!LLVMIR-DAG:  [[LOAD1:%.*]] = bitcast i8* [[CACHE1]] to float*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP2:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB2:[0-9]+]])
!LLVMIR-DAG:  [[CACHE2:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB2:[0-9]+]], i32 [[TMP2:%.*]], i8* bitcast ({ float, float }* @_QMtestEy to i8*), i64 8, i8*** @_QMtestEy.cache)
!LLVMIR-DAG:  [[LOAD2:%.*]] = bitcast i8* [[CACHE2]] to { float, float }*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP3:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB3:[0-9]+]])
!LLVMIR-DAG:  [[CACHE3:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB3:[0-9]+]], i32 [[TMP3:%.*]], i8* bitcast (i32* @_QMtestEz to i8*), i64 4, i8*** @_QMtestEz.cache)
!LLVMIR-DAG:  [[LOAD3:%.*]] = bitcast i8* [[CACHE3]] to i32*, !dbg !{{.*}}
!LLVMIR-DAG:  {{.*}} = load float, float* [[LOAD1]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  {{.*}} = load { float, float }, { float, float }* [[LOAD2]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  {{.*}} = load i32, i32* [[LOAD3]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  [[GEP0:%.*]] = getelementptr %_QMtestTmy_type, %_QMtestTmy_type* [[LOAD0]], i64 0, i32 0, !dbg !{{.*}}
!LLVMIR-DAG:  {{.*}} = load i32, i32* [[GEP0]], align 4, !dbg !{{.*}}
    print *, x, y, z, t%t_i

    !$omp parallel
!FIRDialect-DAG:    [[ADDR38:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>> -> !fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>>
!FIRDialect-DAG:    [[ADDR39:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<f32> -> !fir.ref<f32>
!FIRDialect-DAG:    [[ADDR40:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<!fir.complex<4>> -> !fir.ref<!fir.complex<4>>
!FIRDialect-DAG:    [[ADDR41:%.*]] = omp.threadprivate [[ADDR3]] : !fir.ref<!fir.logical<4>> -> !fir.ref<!fir.logical<4>>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR39]] : !fir.ref<f32>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR40]] : !fir.ref<!fir.complex<4>>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR41]] : !fir.ref<!fir.logical<4>>
!FIRDialect-DAG:    %{{.*}} = fir.coordinate_of [[ADDR38]]
      print *, x, y, z, t%t_i
    !$omp end parallel

!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR1]] : !fir.ref<f32>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR2]] : !fir.ref<!fir.complex<4>>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR3]] : !fir.ref<!fir.logical<4>>
!FIRDialect-DAG:  %{{.*}} = fir.coordinate_of [[NEWADDR0]]

!LLVMIR-DAG:  %{{.*}} = load float, float* [[LOAD1]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = load { float, float }, { float, float }* [[LOAD2]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = load i32, i32* [[LOAD3]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = load i32, i32* [[GEP0]], align 4, !dbg !{{.*}}
    print *, x, y, z, t%t_i

! LLVMIR-LABEL: omp.par.region{{.*}}
!LLVMIR:  [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB0:[0-9]+]])
!LLVMIR-DAG:  [[CACHE0:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB0:[0-9]+]], i32 [[TMP0:%.*]], i8* bitcast (%_QMtestTmy_type* @_QMtestEt to i8*), i64 24, i8*** @_QMtestEt.cache)
!LLVMIR-DAG:  [[LOAD0:%.*]] = bitcast i8* [[CACHE0]] to %_QMtestTmy_type*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP1:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB1:[0-9]+]])
!LLVMIR-DAG:  [[CACHE1:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB1:[0-9]+]], i32 [[TMP1:%.*]], i8* bitcast (float* @_QMtestEx to i8*), i64 4, i8*** @_QMtestEx.cache)
!LLVMIR-DAG:  [[LOAD1:%.*]] = bitcast i8* [[CACHE1]] to float*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP2:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB2:[0-9]+]])
!LLVMIR-DAG:  [[CACHE2:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB2:[0-9]+]], i32 [[TMP2:%.*]], i8* bitcast ({ float, float }* @_QMtestEy to i8*), i64 8, i8*** @_QMtestEy.cache)
!LLVMIR-DAG:  [[LOAD2:%.*]] = bitcast i8* [[CACHE2]] to { float, float }*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP3:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB3:[0-9]+]])
!LLVMIR-DAG:  [[CACHE3:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB3:[0-9]+]], i32 [[TMP3:%.*]], i8* bitcast (i32* @_QMtestEz to i8*), i64 4, i8*** @_QMtestEz.cache)
!LLVMIR-DAG:  [[LOAD3:%.*]] = bitcast i8* [[CACHE3]] to i32*, !dbg !{{.*}}
!LLVMIR-DAG:  {{.*}} = load float, float* [[LOAD1]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  {{.*}} = load { float, float }, { float, float }* [[LOAD2]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  {{.*}} = load i32, i32* [[LOAD3]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  [[GEP0:%.*]] = getelementptr %_QMtestTmy_type, %_QMtestTmy_type* [[LOAD0]], i64 0, i32 0, !dbg !{{.*}}
!LLVMIR-DAG:  {{.*}} = load i32, i32* [[GEP0]], align 4, !dbg !{{.*}}
  end
end
