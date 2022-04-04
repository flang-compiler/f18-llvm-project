! This test checks lowering of OpenMP Threadprivate Directive.
! Test for non-character non-SAVEd non-initialized scalars with or without
! allocatable or pointer attribute in main program.

! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | tco | FileCheck %s --check-prefix=LLVMIR

program test
  integer :: x
  real :: y
  logical :: z
  complex :: w
  integer, pointer :: a
  real, allocatable :: b

!FIRDialect-DAG:  [[ADDR0:%.*]] = fir.address_of(@_QFEa) : !fir.ref<!fir.box<!fir.ptr<i32>>>
!FIRDialect-DAG:  [[NEWADDR0:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.box<!fir.ptr<i32>>> -> !fir.ref<!fir.box<!fir.ptr<i32>>>
!FIRDialect-DAG:  [[ADDR1:%.*]] = fir.address_of(@_QFEb) : !fir.ref<!fir.box<!fir.heap<f32>>>
!FIRDialect-DAG:  [[NEWADDR1:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<!fir.box<!fir.heap<f32>>> -> !fir.ref<!fir.box<!fir.heap<f32>>>
!FIRDialect-DAG:  [[ADDR2:%.*]] = fir.address_of(@_QFEw) : !fir.ref<!fir.complex<4>>
!FIRDialect-DAG:  [[NEWADDR2:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<!fir.complex<4>> -> !fir.ref<!fir.complex<4>>
!FIRDialect-DAG:  [[ADDR3:%.*]] = fir.address_of(@_QFEx) : !fir.ref<i32>
!FIRDialect-DAG:  [[NEWADDR3:%.*]] = omp.threadprivate [[ADDR3]] : !fir.ref<i32> -> !fir.ref<i32>
!FIRDialect-DAG:  [[ADDR4:%.*]] = fir.address_of(@_QFEy) : !fir.ref<f32>
!FIRDialect-DAG:  [[NEWADDR4:%.*]] = omp.threadprivate [[ADDR4]] : !fir.ref<f32> -> !fir.ref<f32>
!FIRDialect-DAG:  [[ADDR5:%.*]] = fir.address_of(@_QFEz) : !fir.ref<!fir.logical<4>>
!FIRDialect-DAG:  [[NEWADDR5:%.*]] = omp.threadprivate [[ADDR5]] : !fir.ref<!fir.logical<4>> -> !fir.ref<!fir.logical<4>>

!LLVMIR-DAG: @_QFEa = internal global { i32*, i64, i32, i8, i8, i8, i8 } { i32* null, i64 4, i32 20180515, i8 0, i8 9, i8 1, i8 0 }
!LLVMIR-DAG: @_QFEb = internal global { float*, i64, i32, i8, i8, i8, i8 } { float* null, i64 4, i32 20180515, i8 0, i8 27, i8 2, i8 0 }
!LLVMIR-DAG: @_QFEw = internal global { float, float } undef
!LLVMIR-DAG: @_QFEx = internal global i32 undef
!LLVMIR-DAG: @_QFEy = internal global float undef
!LLVMIR-DAG: @_QFEz = internal global i32 undef

!LLVMIR-DAG: @_QFEa.cache = common global i8** null
!LLVMIR-DAG: @_QFEb.cache = common global i8** null
!LLVMIR-DAG: @_QFEw.cache = common global i8** null
!LLVMIR-DAG: @_QFEx.cache = common global i8** null
!LLVMIR-DAG: @_QFEy.cache = common global i8** null
!LLVMIR-DAG: @_QFEz.cache = common global i8** null
  !$omp threadprivate(x, y, z, w, a, b, w, a, b)

  call sub(a, b)

!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR3]] : !fir.ref<i32>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR4]] : !fir.ref<f32>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR5]] : !fir.ref<!fir.logical<4>>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR2]] : !fir.ref<!fir.complex<4>>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR1]] : !fir.ref<!fir.box<!fir.heap<f32>>>

!LLVMIR:  [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB0:[0-9]+]])
!LLVMIR-DAG:  [[CACHE0:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB0:[0-9]+]], i32 [[TMP0:%.*]], i8* bitcast ({ i32*, i64, i32, i8, i8, i8, i8 }* @_QFEa to i8*), i64 24, i8*** @_QFEa.cache)
!LLVMIR-DAG:  [[LOAD0:%.*]] = bitcast i8* [[CACHE0]] to { i32*, i64, i32, i8, i8, i8, i8 }*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP1:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB1:[0-9]+]])
!LLVMIR-DAG:  [[CACHE1:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB1:[0-9]+]], i32 [[TMP1:%.*]], i8* bitcast ({ float*, i64, i32, i8, i8, i8, i8 }* @_QFEb to i8*), i64 24, i8*** @_QFEb.cache)
!LLVMIR-DAG:  [[LOAD1:%.*]] = bitcast i8* [[CACHE1]] to { float*, i64, i32, i8, i8, i8, i8 }*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP2:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB2:[0-9]+]])
!LLVMIR-DAG:  [[CACHE2:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB2:[0-9]+]], i32 [[TMP2:%.*]], i8* bitcast ({ float, float }* @_QFEw to i8*), i64 8, i8*** @_QFEw.cache)
!LLVMIR-DAG:  [[LOAD2:%.*]] = bitcast i8* [[CACHE2]] to { float, float }*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP2:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB3:[0-9]+]])
!LLVMIR-DAG:  [[CACHE3:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB3:[0-9]+]], i32 [[TMP3:%.*]], i8* bitcast (i32* @_QFEx to i8*), i64 4, i8*** @_QFEx.cache)
!LLVMIR-DAG:  [[LOAD3:%.*]] = bitcast i8* [[CACHE3]] to i32*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP4:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB4:[0-9]+]])
!LLVMIR-DAG:  [[CACHE4:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB4:[0-9]+]], i32 [[TMP4:%.*]], i8* bitcast (float* @_QFEy to i8*), i64 4, i8*** @_QFEy.cache)
!LLVMIR-DAG:  [[LOAD4:%.*]] = bitcast i8* [[CACHE4]] to float*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP5:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB4:[0-9]+]])
!LLVMIR-DAG:  [[CACHE5:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB5:[0-9]+]], i32 [[TMP5:%.*]], i8* bitcast (i32* @_QFEz to i8*), i64 4, i8*** @_QFEz.cache)
!LLVMIR-DAG:  [[LOAD5:%.*]] = bitcast i8* [[CACHE5]] to i32*, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = load i32, i32* [[LOAD3]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = load float, float* [[LOAD4]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = load i32, i32* [[LOAD5]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = load { float, float }, { float, float }* [[LOAD2]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = getelementptr { i32*, i64, i32, i8, i8, i8, i8 }, { i32*, i64, i32, i8, i8, i8, i8 }* [[LOAD0]], i32 0, i32 0, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = getelementptr { float*, i64, i32, i8, i8, i8, i8 }, { float*, i64, i32, i8, i8, i8, i8 }* [[LOAD1]], i32 0, i32 0, !dbg !{{.*}}
  print *, x, y, z, w, a, b

  !$omp parallel
!FIRDialect-DAG:    [[ADDR68:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.box<!fir.ptr<i32>>> -> !fir.ref<!fir.box<!fir.ptr<i32>>>
!FIRDialect-DAG:    [[ADDR69:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<!fir.box<!fir.heap<f32>>> -> !fir.ref<!fir.box<!fir.heap<f32>>>
!FIRDialect-DAG:    [[ADDR70:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<!fir.complex<4>> -> !fir.ref<!fir.complex<4>>
!FIRDialect-DAG:    [[ADDR71:%.*]] = omp.threadprivate [[ADDR3]] : !fir.ref<i32> -> !fir.ref<i32>
!FIRDialect-DAG:    [[ADDR72:%.*]] = omp.threadprivate [[ADDR4]] : !fir.ref<f32> -> !fir.ref<f32>
!FIRDialect-DAG:    [[ADDR73:%.*]] = omp.threadprivate [[ADDR5]] : !fir.ref<!fir.logical<4>> -> !fir.ref<!fir.logical<4>>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR71]] : !fir.ref<i32>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR72]] : !fir.ref<f32>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR73]] : !fir.ref<!fir.logical<4>>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR70]] : !fir.ref<!fir.complex<4>>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR68]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR69]] : !fir.ref<!fir.box<!fir.heap<f32>>>
    print *, x, y, z, w, a, b
  !$omp end parallel

!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR3]] : !fir.ref<i32>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR4]] : !fir.ref<f32>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR5]] : !fir.ref<!fir.logical<4>>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR2]] : !fir.ref<!fir.complex<4>>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR1]] : !fir.ref<!fir.box<!fir.heap<f32>>>

!LLVMIR-DAG:  %{{.*}} = load i32, i32* [[LOAD3]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = load float, float* [[LOAD4]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = load i32, i32* [[LOAD5]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = load { float, float }, { float, float }* [[LOAD2]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = getelementptr { i32*, i64, i32, i8, i8, i8, i8 }, { i32*, i64, i32, i8, i8, i8, i8 }* [[LOAD0]], i32 0, i32 0, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = getelementptr { float*, i64, i32, i8, i8, i8, i8 }, { float*, i64, i32, i8, i8, i8, i8 }* [[LOAD1]], i32 0, i32 0, !dbg !{{.*}}
  print *, x, y, z, w, a, b

!FIRDialect:  return

!FIRDialect-DAG: fir.global internal @_QFEa : !fir.box<!fir.ptr<i32>> {
!FIRDialect-DAG:   [[Z0:%.*]] = fir.zero_bits !fir.ptr<i32>
!FIRDialect-DAG:   [[E0:%.*]] = fir.embox [[Z0]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
!FIRDialect-DAG:   fir.has_value [[E0]] : !fir.box<!fir.ptr<i32>>
!FIRDialect-DAG: }
!FIRDialect-DAG: fir.global internal @_QFEb : !fir.box<!fir.heap<f32>> {
!FIRDialect-DAG:   [[Z1:%.*]] = fir.zero_bits !fir.heap<f32>
!FIRDialect-DAG:   [[E1:%.*]] = fir.embox [[Z1]] : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
!FIRDialect-DAG:   fir.has_value [[E1]] : !fir.box<!fir.heap<f32>>
!FIRDialect-DAG: }
!FIRDialect-DAG: fir.global internal @_QFEw : !fir.complex<4> {
!FIRDialect-DAG:   [[Z2:%.*]] = fir.undefined !fir.complex<4>
!FIRDialect-DAG:   fir.has_value [[Z2]] : !fir.complex<4>
!FIRDialect-DAG: }
!FIRDialect-DAG: fir.global internal @_QFEx : i32 {
!FIRDialect-DAG:   [[Z3:%.*]] = fir.undefined i32
!FIRDialect-DAG:   fir.has_value [[Z3]] : i32
!FIRDialect-DAG: }
!FIRDialect-DAG: fir.global internal @_QFEy : f32 {
!FIRDialect-DAG:   [[Z4:%.*]] = fir.undefined f32
!FIRDialect-DAG:   fir.has_value [[Z4]] : f32
!FIRDialect-DAG: }
!FIRDialect-DAG: fir.global internal @_QFEz : !fir.logical<4> {
!FIRDialect-DAG:   [[Z5:%.*]] = fir.undefined !fir.logical<4>
!FIRDialect-DAG:   fir.has_value [[Z5]] : !fir.logical<4>
!FIRDialect-DAG: }

! LLVMIR-LABEL: omp.par.region{{.*}}
!LLVMIR:  [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB0:[0-9]+]])
!LLVMIR-DAG:  [[CACHE0:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB0:[0-9]+]], i32 [[TMP0:%.*]], i8* bitcast ({ i32*, i64, i32, i8, i8, i8, i8 }* @_QFEa to i8*), i64 24, i8*** @_QFEa.cache)
!LLVMIR-DAG:  [[LOAD0:%.*]] = bitcast i8* [[CACHE0]] to { i32*, i64, i32, i8, i8, i8, i8 }*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP1:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB1:[0-9]+]])
!LLVMIR-DAG:  [[CACHE1:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB1:[0-9]+]], i32 [[TMP1:%.*]], i8* bitcast ({ float*, i64, i32, i8, i8, i8, i8 }* @_QFEb to i8*), i64 24, i8*** @_QFEb.cache)
!LLVMIR-DAG:  [[LOAD1:%.*]] = bitcast i8* [[CACHE1]] to { float*, i64, i32, i8, i8, i8, i8 }*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP2:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB2:[0-9]+]])
!LLVMIR-DAG:  [[CACHE2:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB2:[0-9]+]], i32 [[TMP2:%.*]], i8* bitcast ({ float, float }* @_QFEw to i8*), i64 8, i8*** @_QFEw.cache)
!LLVMIR-DAG:  [[LOAD2:%.*]] = bitcast i8* [[CACHE2]] to { float, float }*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP2:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB3:[0-9]+]])
!LLVMIR-DAG:  [[CACHE3:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB3:[0-9]+]], i32 [[TMP3:%.*]], i8* bitcast (i32* @_QFEx to i8*), i64 4, i8*** @_QFEx.cache)
!LLVMIR-DAG:  [[LOAD3:%.*]] = bitcast i8* [[CACHE3]] to i32*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP4:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB4:[0-9]+]])
!LLVMIR-DAG:  [[CACHE4:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB4:[0-9]+]], i32 [[TMP4:%.*]], i8* bitcast (float* @_QFEy to i8*), i64 4, i8*** @_QFEy.cache)
!LLVMIR-DAG:  [[LOAD4:%.*]] = bitcast i8* [[CACHE4]] to float*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP5:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB4:[0-9]+]])
!LLVMIR-DAG:  [[CACHE5:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB5:[0-9]+]], i32 [[TMP5:%.*]], i8* bitcast (i32* @_QFEz to i8*), i64 4, i8*** @_QFEz.cache)
!LLVMIR-DAG:  [[LOAD5:%.*]] = bitcast i8* [[CACHE5]] to i32*, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = load i32, i32* [[LOAD3]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = load float, float* [[LOAD4]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = load i32, i32* [[LOAD5]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = load { float, float }, { float, float }* [[LOAD2]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = getelementptr { i32*, i64, i32, i8, i8, i8, i8 }, { i32*, i64, i32, i8, i8, i8, i8 }* [[LOAD0]], i32 0, i32 0, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = getelementptr { float*, i64, i32, i8, i8, i8, i8 }, { float*, i64, i32, i8, i8, i8, i8 }* [[LOAD1]], i32 0, i32 0, !dbg !{{.*}}
end
