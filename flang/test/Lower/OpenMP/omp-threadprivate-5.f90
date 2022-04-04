! This test checks lowering of OpenMP Threadprivate Directive.
! Test for allocatable and pointer variables.

! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | tco | FileCheck %s --check-prefix=LLVMIR

module test
  integer, pointer :: x(:), m
  real, allocatable :: y(:), n

  !$omp threadprivate(x, y, m, n)

!FIRDialect-DAG: fir.global @_QMtestEm : !fir.box<!fir.ptr<i32>> {
!FIRDialect-DAG: fir.global @_QMtestEn : !fir.box<!fir.heap<f32>> {
!FIRDialect-DAG: fir.global @_QMtestEx : !fir.box<!fir.ptr<!fir.array<?xi32>>> {
!FIRDialect-DAG: fir.global @_QMtestEy : !fir.box<!fir.heap<!fir.array<?xf32>>> {

!LLVMIR-DAG: @_QMtestEm = global { i32*, i64, i32, i8, i8, i8, i8 } { i32* null, i64 4, i32 20180515, i8 0, i8 9, i8 1, i8 0 }
!LLVMIR-DAG: @_QMtestEn = global { float*, i64, i32, i8, i8, i8, i8 } { float* null, i64 4, i32 20180515, i8 0, i8 27, i8 2, i8 0 }
!LLVMIR-DAG: @_QMtestEx = global { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } { i32* null, i64 4, i32 20180515, i8 1, i8 9, i8 1, i8 0
!LLVMIR-DAG: @_QMtestEy = global { float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } { float* null, i64 4, i32 20180515, i8 1, i8 27, i8 2, i8 0

!LLVMIR-DAG: @_QMtestEm.cache = common global i8** null
!LLVMIR-DAG: @_QMtestEn.cache = common global i8** null
!LLVMIR-DAG: @_QMtestEx.cache = common global i8** null
!LLVMIR-DAG: @_QMtestEy.cache = common global i8** null

contains
  subroutine sub()
!FIRDialect-DAG:  [[ADDR0:%.*]] = fir.address_of(@_QMtestEm) : !fir.ref<!fir.box<!fir.ptr<i32>>>
!FIRDialect-DAG:  [[NEWADDR0:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.box<!fir.ptr<i32>>> -> !fir.ref<!fir.box<!fir.ptr<i32>>>
!FIRDialect-DAG:  [[ADDR1:%.*]] = fir.address_of(@_QMtestEn) : !fir.ref<!fir.box<!fir.heap<f32>>>
!FIRDialect-DAG:  [[NEWADDR1:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<!fir.box<!fir.heap<f32>>> -> !fir.ref<!fir.box<!fir.heap<f32>>>
!FIRDialect-DAG:  [[ADDR2:%.*]] = fir.address_of(@_QMtestEx) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!FIRDialect-DAG:  [[NEWADDR2:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!FIRDialect-DAG:  [[ADDR3:%.*]] = fir.address_of(@_QMtestEy) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!FIRDialect-DAG:  [[NEWADDR3:%.*]] = omp.threadprivate [[ADDR3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR2]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR1]] : !fir.ref<!fir.box<!fir.heap<f32>>>

!LLVMIR:  [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB0:[0-9]+]])
!LLVMIR-DAG:  [[CACHE0:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB0:[0-9]+]], i32 [[TMP0:%.*]], i8* bitcast ({ i32*, i64, i32, i8, i8, i8, i8 }* @_QMtestEm to i8*), i64 24, i8*** @_QMtestEm.cache)
!LLVMIR-DAG:  [[LOAD0:%.*]] = bitcast i8* [[CACHE0]] to { i32*, i64, i32, i8, i8, i8, i8 }*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP1:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB1:[0-9]+]])
!LLVMIR-DAG:  [[CACHE1:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB1:[0-9]+]], i32 [[TMP1:%.*]], i8* bitcast ({ float*, i64, i32, i8, i8, i8, i8 }* @_QMtestEn to i8*), i64 24, i8*** @_QMtestEn.cache)
!LLVMIR-DAG:  [[LOAD1:%.*]] = bitcast i8* [[CACHE1]] to { float*, i64, i32, i8, i8, i8, i8 }*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP2:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB2:[0-9]+]])
!LLVMIR-DAG:  [[CACHE2:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB2:[0-9]+]], i32 [[TMP2:%.*]], i8* bitcast ({ i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* @_QMtestEx to i8*), i64 48, i8*** @_QMtestEx.cache)
!LLVMIR-DAG:  [[LOAD2:%.*]] = bitcast i8* [[CACHE2]] to { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP3:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB3:[0-9]+]])
!LLVMIR-DAG:  [[CACHE3:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB3:[0-9]+]], i32 [[TMP3:%.*]], i8* bitcast ({ float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* @_QMtestEy to i8*), i64 48, i8*** @_QMtestEy.cache)
!LLVMIR-DAG:  [[LOAD3:%.*]] = bitcast i8* [[CACHE3]] to { float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }*, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = getelementptr { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* [[LOAD2]], i32 0, i32 7, i64 0, i32 0, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = getelementptr { float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, { float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* [[LOAD3]], i32 0, i32 7, i64 0, i32 0, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = getelementptr { i32*, i64, i32, i8, i8, i8, i8 }, { i32*, i64, i32, i8, i8, i8, i8 }* [[LOAD0]], i32 0, i32 0, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = getelementptr { float*, i64, i32, i8, i8, i8, i8 }, { float*, i64, i32, i8, i8, i8, i8 }* [[LOAD1]], i32 0, i32 0, !dbg !{{.*}}
    print *, x, y, m, n

    !$omp parallel
!FIRDialect-DAG:    [[ADDR54:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.box<!fir.ptr<i32>>> -> !fir.ref<!fir.box<!fir.ptr<i32>>>
!FIRDialect-DAG:    [[ADDR55:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<!fir.box<!fir.heap<f32>>> -> !fir.ref<!fir.box<!fir.heap<f32>>>
!FIRDialect-DAG:    [[ADDR56:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!FIRDialect-DAG:    [[ADDR57:%.*]] = omp.threadprivate [[ADDR3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR56]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR57]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR54]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR55]] : !fir.ref<!fir.box<!fir.heap<f32>>>
      print *, x, y, m, n
    !$omp end parallel

!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR2]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR1]] : !fir.ref<!fir.box<!fir.heap<f32>>>

!LLVMIR-DAG:  %{{.*}} = getelementptr { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* [[LOAD2]], i32 0, i32 7, i64 0, i32 0, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = getelementptr { float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, { float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* [[LOAD3]], i32 0, i32 7, i64 0, i32 0, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = getelementptr { i32*, i64, i32, i8, i8, i8, i8 }, { i32*, i64, i32, i8, i8, i8, i8 }* [[LOAD0]], i32 0, i32 0, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = getelementptr { float*, i64, i32, i8, i8, i8, i8 }, { float*, i64, i32, i8, i8, i8, i8 }* [[LOAD1]], i32 0, i32 0, !dbg !{{.*}}
    print *, x, y, m, n

! LLVMIR-LABEL: omp.par.region{{.*}}
!LLVMIR:  [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB0:[0-9]+]])
!LLVMIR-DAG:  [[CACHE0:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB0:[0-9]+]], i32 [[TMP0:%.*]], i8* bitcast ({ i32*, i64, i32, i8, i8, i8, i8 }* @_QMtestEm to i8*), i64 24, i8*** @_QMtestEm.cache)
!LLVMIR-DAG:  [[LOAD0:%.*]] = bitcast i8* [[CACHE0]] to { i32*, i64, i32, i8, i8, i8, i8 }*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP1:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB1:[0-9]+]])
!LLVMIR-DAG:  [[CACHE1:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB1:[0-9]+]], i32 [[TMP1:%.*]], i8* bitcast ({ float*, i64, i32, i8, i8, i8, i8 }* @_QMtestEn to i8*), i64 24, i8*** @_QMtestEn.cache)
!LLVMIR-DAG:  [[LOAD1:%.*]] = bitcast i8* [[CACHE1]] to { float*, i64, i32, i8, i8, i8, i8 }*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP2:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB2:[0-9]+]])
!LLVMIR-DAG:  [[CACHE2:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB2:[0-9]+]], i32 [[TMP2:%.*]], i8* bitcast ({ i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* @_QMtestEx to i8*), i64 48, i8*** @_QMtestEx.cache)
!LLVMIR-DAG:  [[LOAD2:%.*]] = bitcast i8* [[CACHE2]] to { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP3:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB3:[0-9]+]])
!LLVMIR-DAG:  [[CACHE3:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB3:[0-9]+]], i32 [[TMP3:%.*]], i8* bitcast ({ float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* @_QMtestEy to i8*), i64 48, i8*** @_QMtestEy.cache)
!LLVMIR-DAG:  [[LOAD3:%.*]] = bitcast i8* [[CACHE3]] to { float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }*, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = getelementptr { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* [[LOAD2]], i32 0, i32 7, i64 0, i32 0, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = getelementptr { float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, { float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* [[LOAD3]], i32 0, i32 7, i64 0, i32 0, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = getelementptr { i32*, i64, i32, i8, i8, i8, i8 }, { i32*, i64, i32, i8, i8, i8, i8 }* [[LOAD0]], i32 0, i32 0, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = getelementptr { float*, i64, i32, i8, i8, i8, i8 }, { float*, i64, i32, i8, i8, i8, i8 }* [[LOAD1]], i32 0, i32 0, !dbg !{{.*}}
  end
end
