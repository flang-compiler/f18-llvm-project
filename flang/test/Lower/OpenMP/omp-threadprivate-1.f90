! This test checks lowering of OpenMP Threadprivate Directive.
! Test for variables with different kind.

! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | tco | FileCheck %s --check-prefix=LLVMIR

program test
  integer, save :: i
  integer(kind=1), save :: i1
  integer(kind=2), save :: i2
  integer(kind=4), save :: i4
  integer(kind=8), save :: i8
  integer(kind=16), save :: i16

!FIRDialect-DAG:  [[ADDR0:%.*]] = fir.address_of(@_QFEi) : !fir.ref<i32>
!FIRDialect-DAG:  [[NEWADDR0:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<i32> -> !fir.ref<i32>
!FIRDialect-DAG:  [[ADDR1:%.*]] = fir.address_of(@_QFEi1) : !fir.ref<i8>
!FIRDialect-DAG:  [[NEWADDR1:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<i8> -> !fir.ref<i8>
!FIRDialect-DAG:  [[ADDR2:%.*]] = fir.address_of(@_QFEi16) : !fir.ref<i128>
!FIRDialect-DAG:  [[NEWADDR2:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<i128> -> !fir.ref<i128>
!FIRDialect-DAG:  [[ADDR3:%.*]] = fir.address_of(@_QFEi2) : !fir.ref<i16>
!FIRDialect-DAG:  [[NEWADDR3:%.*]] = omp.threadprivate [[ADDR3]] : !fir.ref<i16> -> !fir.ref<i16>
!FIRDialect-DAG:  [[ADDR4:%.*]] = fir.address_of(@_QFEi4) : !fir.ref<i32>
!FIRDialect-DAG:  [[NEWADDR4:%.*]] = omp.threadprivate [[ADDR4]] : !fir.ref<i32> -> !fir.ref<i32>
!FIRDialect-DAG:  [[ADDR5:%.*]] = fir.address_of(@_QFEi8) : !fir.ref<i64>
!FIRDialect-DAG:  [[NEWADDR5:%.*]] = omp.threadprivate [[ADDR5]] : !fir.ref<i64> -> !fir.ref<i64>

!LLVMIR-DAG: @_QFEi = internal global i32 undef
!LLVMIR-DAG: @_QFEi1 = internal global i8 undef
!LLVMIR-DAG: @_QFEi16 = internal global i128 undef
!LLVMIR-DAG: @_QFEi2 = internal global i16 undef
!LLVMIR-DAG: @_QFEi4 = internal global i32 undef
!LLVMIR-DAG: @_QFEi8 = internal global i64 undef
  !$omp threadprivate(i, i1, i2, i4, i8, i16)

!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR0]] : !fir.ref<i32>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR1]] : !fir.ref<i8>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR2]] : !fir.ref<i128>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR3]] : !fir.ref<i16>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR4]] : !fir.ref<i32>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR5]] : !fir.ref<i64>

!LLVMIR-DAG: @_QFEi.cache = common global i8** null
!LLVMIR-DAG: @_QFEi1.cache = common global i8** null
!LLVMIR-DAG: @_QFEi16.cache = common global i8** null
!LLVMIR-DAG: @_QFEi2.cache = common global i8** null
!LLVMIR-DAG: @_QFEi4.cache = common global i8** null
!LLVMIR-DAG: @_QFEi8.cache = common global i8** null

! CHECK-LABEL: @_QQmain()
!LLVMIR: [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB0:[0-9]+]])
!LLVMIR-DAG: [[CACHE0:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB0:[0-9]+]], i32 [[TMP0:%.*]], i8* bitcast (i32* @_QFEi to i8*), i64 4, i8*** @_QFEi.cache)
!LLVMIR-DAG: [[LOAD0:%.*]] = bitcast i8* [[CACHE0]] to i32*, !dbg !{{.*}}
!LLVMIR-DAG: [[TMP1:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB1:[0-9]+]])
!LLVMIR-DAG: [[CACHE1:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB1:[0-9]+]], i32 [[TMP1:%.*]], i8* @_QFEi1, i64 1, i8*** @_QFEi1.cache)
!LLVMIR-DAG: [[TMP2:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB2:[0-9]+]])
!LLVMIR-DAG: [[CACHE2:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB2:[0-9]+]], i32 [[TMP2:%.*]], i8* bitcast (i16* @_QFEi2 to i8*), i64 2, i8*** @_QFEi2.cache)
!LLVMIR-DAG: [[LOAD2:%.*]] = bitcast i8* [[CACHE2]] to i16*, !dbg !{{.*}}
!LLVMIR-DAG: [[TMP3:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB3:[0-9]+]])
!LLVMIR-DAG: [[CACHE3:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB3:[0-9]+]], i32 [[TMP3:%.*]], i8* bitcast (i32* @_QFEi4 to i8*), i64 4, i8*** @_QFEi4.cache)
!LLVMIR-DAG: [[LOAD3:%.*]] = bitcast i8* [[CACHE3]] to i32*, !dbg !{{.*}}
!LLVMIR-DAG: [[TMP4:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB4:[0-9]+]])
!LLVMIR-DAG: [[CACHE4:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB4:[0-9]+]], i32 [[TMP4:%.*]], i8* bitcast (i64* @_QFEi8 to i8*), i64 8, i8*** @_QFEi8.cache)
!LLVMIR-DAG: [[LOAD4:%.*]] = bitcast i8* [[CACHE4]] to i64*, !dbg !{{.*}}
!LLVMIR-DAG: [[TMP5:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB5:[0-9]+]])
!LLVMIR-DAG: [[CACHE5:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB5:[0-9]+]], i32 [[TMP5:%.*]], i8* bitcast (i128* @_QFEi16 to i8*), i64 16, i8*** @_QFEi16.cache)
!LLVMIR-DAG: [[LOAD5:%.*]] = bitcast i8* [[CACHE5]] to i128*, !dbg !{{.*}}
!LLVMIR-DAG: {{.*}} = load i32, i32* [[LOAD0]], align 4, !dbg !{{.*}}
!LLVMIR-DAG: {{.*}} = load i8, i8* [[CACHE1]], align 1, !dbg !{{.*}}
!LLVMIR-DAG: {{.*}} = load i16, i16* [[LOAD2]], align 2, !dbg !{{.*}}
!LLVMIR-DAG: {{.*}} = load i32, i32* [[LOAD3]], align 4, !dbg !{{.*}}
!LLVMIR-DAG: {{.*}} = load i64, i64* [[LOAD4]], align 4, !dbg !{{.*}}
!LLVMIR-DAG: {{.*}} = load i128, i128* [[LOAD5]], align 4, !dbg !{{.*}}
  print *, i, i1, i2, i4, i8, i16

  !$omp parallel
!FIRDialect-DAG:    [[ADDR39:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<i32> -> !fir.ref<i32>
!FIRDialect-DAG:    [[ADDR40:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<i8> -> !fir.ref<i8>
!FIRDialect-DAG:    [[ADDR41:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<i128> -> !fir.ref<i128>
!FIRDialect-DAG:    [[ADDR42:%.*]] = omp.threadprivate [[ADDR3]] : !fir.ref<i16> -> !fir.ref<i16>
!FIRDialect-DAG:    [[ADDR43:%.*]] = omp.threadprivate [[ADDR4]] : !fir.ref<i32> -> !fir.ref<i32>
!FIRDialect-DAG:    [[ADDR44:%.*]] = omp.threadprivate [[ADDR5]] : !fir.ref<i64> -> !fir.ref<i64>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR39]] : !fir.ref<i32>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR40]] : !fir.ref<i8>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR41]] : !fir.ref<i128>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR42]] : !fir.ref<i16>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR43]] : !fir.ref<i32>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR44]] : !fir.ref<i64>
    print *, i, i1, i2, i4, i8, i16
  !$omp end parallel

!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR0]] : !fir.ref<i32>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR1]] : !fir.ref<i8>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR2]] : !fir.ref<i128>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR3]] : !fir.ref<i16>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR4]] : !fir.ref<i32>
!FIRDialect-DAG:  %{{.*}} = fir.load [[NEWADDR5]] : !fir.ref<i64>

!LLVMIR-DAG: {{.*}} = load i32, i32* [[LOAD0]], align 4, !dbg !{{.*}}
!LLVMIR-DAG: {{.*}} = load i8, i8* [[CACHE1]], align 1, !dbg !{{.*}}
!LLVMIR-DAG: {{.*}} = load i16, i16* [[LOAD2]], align 2, !dbg !{{.*}}
!LLVMIR-DAG: {{.*}} = load i32, i32* [[LOAD3]], align 4, !dbg !{{.*}}
!LLVMIR-DAG: {{.*}} = load i64, i64* [[LOAD4]], align 4, !dbg !{{.*}}
!LLVMIR-DAG: {{.*}} = load i128, i128* [[LOAD5]], align 4, !dbg !{{.*}}
  print *, i, i1, i2, i4, i8, i16

!FIRDialect-DAG: fir.global internal @_QFEi : i32 {
!FIRDialect-DAG: fir.global internal @_QFEi1 : i8 {
!FIRDialect-DAG: fir.global internal @_QFEi16 : i128 {
!FIRDialect-DAG: fir.global internal @_QFEi2 : i16 {
!FIRDialect-DAG: fir.global internal @_QFEi4 : i32 {
!FIRDialect-DAG: fir.global internal @_QFEi8 : i64 {

! LLVMIR-LABEL: omp.par.region{{.*}}
!LLVMIR: [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB0:[0-9]+]])
!LLVMIR-DAG: [[CACHE0:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB0:[0-9]+]], i32 [[TMP0:%.*]], i8* bitcast (i32* @_QFEi to i8*), i64 4, i8*** @_QFEi.cache)
!LLVMIR-DAG: [[LOAD0:%.*]] = bitcast i8* [[CACHE0]] to i32*, !dbg !{{.*}}
!LLVMIR-DAG: [[TMP1:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB1:[0-9]+]])
!LLVMIR-DAG: [[CACHE1:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB1:[0-9]+]], i32 [[TMP1:%.*]], i8* @_QFEi1, i64 1, i8*** @_QFEi1.cache)
!LLVMIR-DAG: [[TMP2:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB2:[0-9]+]])
!LLVMIR-DAG: [[CACHE2:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB2:[0-9]+]], i32 [[TMP2:%.*]], i8* bitcast (i16* @_QFEi2 to i8*), i64 2, i8*** @_QFEi2.cache)
!LLVMIR-DAG: [[LOAD2:%.*]] = bitcast i8* [[CACHE2]] to i16*, !dbg !{{.*}}
!LLVMIR-DAG: [[TMP3:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB3:[0-9]+]])
!LLVMIR-DAG: [[CACHE3:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB3:[0-9]+]], i32 [[TMP3:%.*]], i8* bitcast (i32* @_QFEi4 to i8*), i64 4, i8*** @_QFEi4.cache)
!LLVMIR-DAG: [[LOAD3:%.*]] = bitcast i8* [[CACHE3]] to i32*, !dbg !{{.*}}
!LLVMIR-DAG: [[TMP4:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB4:[0-9]+]])
!LLVMIR-DAG: [[CACHE4:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB4:[0-9]+]], i32 [[TMP4:%.*]], i8* bitcast (i64* @_QFEi8 to i8*), i64 8, i8*** @_QFEi8.cache)
!LLVMIR-DAG: [[LOAD4:%.*]] = bitcast i8* [[CACHE4]] to i64*, !dbg !{{.*}}
!LLVMIR-DAG: [[TMP5:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB5:[0-9]+]])
!LLVMIR-DAG: [[CACHE5:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB5:[0-9]+]], i32 [[TMP5:%.*]], i8* bitcast (i128* @_QFEi16 to i8*), i64 16, i8*** @_QFEi16.cache)
!LLVMIR-DAG: [[LOAD5:%.*]] = bitcast i8* [[CACHE5]] to i128*, !dbg !{{.*}}
!LLVMIR-DAG: {{.*}} = load i32, i32* [[LOAD0]], align 4, !dbg !{{.*}}
!LLVMIR-DAG: {{.*}} = load i8, i8* [[CACHE1]], align 1, !dbg !{{.*}}
!LLVMIR-DAG: {{.*}} = load i16, i16* [[LOAD2]], align 2, !dbg !{{.*}}
!LLVMIR-DAG: {{.*}} = load i32, i32* [[LOAD3]], align 4, !dbg !{{.*}}
!LLVMIR-DAG: {{.*}} = load i64, i64* [[LOAD4]], align 4, !dbg !{{.*}}
!LLVMIR-DAG: {{.*}} = load i128, i128* [[LOAD5]], align 4, !dbg !{{.*}}
end
