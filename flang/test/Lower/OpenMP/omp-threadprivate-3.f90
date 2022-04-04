! This test checks lowering of OpenMP Threadprivate Directive.
! Test for character, array, and character array.

! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | tco | FileCheck %s --check-prefix=LLVMIR

module test
  character :: x
  integer :: y(5)
  character(5) :: z(5)

  !$omp threadprivate(x, y, z)

!FIRDialect-DAG: fir.global @_QMtestEx : !fir.char<1> {
!FIRDialect-DAG: fir.global @_QMtestEy : !fir.array<5xi32> {
!FIRDialect-DAG: fir.global @_QMtestEz : !fir.array<5x!fir.char<1,5>> {

!LLVMIR-DAG: @_QMtestEx = global [1 x i8] undef
!LLVMIR-DAG: @_QMtestEy = global [5 x i32] undef
!LLVMIR-DAG: @_QMtestEz = global [5 x [5 x i8]] undef
!LLVMIR-DAG: @_QMtestEx.cache = common global i8** null
!LLVMIR-DAG: @_QMtestEy.cache = common global i8** null
!LLVMIR-DAG: @_QMtestEz.cache = common global i8** null

contains
  subroutine sub()
!FIRDialect-DAG:  [[ADDR0:%.*]] = fir.address_of(@_QMtestEx) : !fir.ref<!fir.char<1>>
!FIRDialect-DAG:  [[NEWADDR0:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.char<1>> -> !fir.ref<!fir.char<1>>
!FIRDialect-DAG:  [[ADDR1:%.*]] = fir.address_of(@_QMtestEy) : !fir.ref<!fir.array<5xi32>>
!FIRDialect-DAG:  [[NEWADDR1:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<!fir.array<5xi32>> -> !fir.ref<!fir.array<5xi32>>
!FIRDialect-DAG:  [[ADDR2:%.*]] = fir.address_of(@_QMtestEz) : !fir.ref<!fir.array<5x!fir.char<1,5>>>
!FIRDialect-DAG:  [[NEWADDR2:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<!fir.array<5x!fir.char<1,5>>> -> !fir.ref<!fir.array<5x!fir.char<1,5>>>
!FIRDialect-DAG:  %{{.*}} = fir.convert [[NEWADDR0]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
!FIRDialect-DAG:  %{{.*}} = fir.embox [[NEWADDR1]](%{{.*}}) : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<5xi32>>
!FIRDialect-DAG:  %{{.*}} = fir.embox [[NEWADDR2]](%{{.*}}) : (!fir.ref<!fir.array<5x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<5x!fir.char<1,5>>>

! CHECK-LABEL: @_QMtestPsub()
!LLVMIR:  [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB0:[0-9]+]])
!LLVMIR-DAG:  [[CACHE0:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB0:[0-9]+]], i32 [[TMP0:%.*]], i8* getelementptr inbounds ([1 x i8], [1 x i8]* @_QMtestEx, i32 0, i32 0), i64 1, i8*** @_QMtestEx.cache)
!LLVMIR-DAG:  [[LOAD0:%.*]] = bitcast i8* [[CACHE0]] to [1 x i8]*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP1:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB1:[0-9]+]])
!LLVMIR-DAG:  [[CACHE1:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB1:[0-9]+]], i32 [[TMP1:%.*]], i8* bitcast ([5 x i32]* @_QMtestEy to i8*), i64 20, i8*** @_QMtestEy.cache)
!LLVMIR-DAG:  [[LOAD1:%.*]] = bitcast i8* [[CACHE1]] to [5 x i32]*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP2:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB2:[0-9]+]])
!LLVMIR-DAG:  [[CACHE2:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB2:[0-9]+]], i32 [[TMP2:%.*]], i8* getelementptr inbounds ([5 x [5 x i8]], [5 x [5 x i8]]* @_QMtestEz, i32 0, i32 0, i32 0), i64 25, i8*** @_QMtestEz.cache)
!LLVMIR-DAG:  [[LOAD2:%.*]] = bitcast i8* [[CACHE2]] to [5 x [5 x i8]]*, !dbg !{{.*}}
!LLVMIR-DAG:  {{.*}} = bitcast [1 x i8]* [[LOAD0]] to i8*, !dbg !{{.*}}
!LLVMIR-DAG:  {{.*}} = insertvalue { [5 x i32]*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } { [5 x i32]* undef, i64 4, i32 20180515, i8 1, i8 9, i8 0, i8 0
!LLVMIR-DAG:  {{.*}} = insertvalue { [5 x [5 x i8]]*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } { [5 x [5 x i8]]* undef, i64 5, i32 20180515, i8 1, i8 40, i8 0, i8 0
    print *, x, y, z

    !$omp parallel
!FIRDialect-DAG:    [[ADDR33:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.char<1>> -> !fir.ref<!fir.char<1>>
!FIRDialect-DAG:    [[ADDR34:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<!fir.array<5xi32>> -> !fir.ref<!fir.array<5xi32>>
!FIRDialect-DAG:    [[ADDR35:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<!fir.array<5x!fir.char<1,5>>> -> !fir.ref<!fir.array<5x!fir.char<1,5>>>
!FIRDialect-DAG:    %{{.*}} = fir.convert [[ADDR33]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
!FIRDialect-DAG:    %{{.*}} = fir.embox [[ADDR34]](%{{.*}}) : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<5xi32>>
!FIRDialect-DAG:    %{{.*}} = fir.embox [[ADDR35]](%{{.*}}) : (!fir.ref<!fir.array<5x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<5x!fir.char<1,5>>>
      print *, x, y, z
    !$omp end parallel

!FIRDialect-DAG:  %{{.*}} = fir.convert [[NEWADDR0]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
!FIRDialect-DAG:  %{{.*}} = fir.embox [[NEWADDR1]](%{{.*}}) : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<5xi32>>
!FIRDialect-DAG:  %{{.*}} = fir.embox [[NEWADDR2]](%{{.*}}) : (!fir.ref<!fir.array<5x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<5x!fir.char<1,5>>>
    print *, x, y, z

! LLVMIR-LABEL: omp.par.region{{.*}}
!LLVMIR:  [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB0:[0-9]+]])
!LLVMIR-DAG:  [[CACHE0:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB0:[0-9]+]], i32 [[TMP0:%.*]], i8* getelementptr inbounds ([1 x i8], [1 x i8]* @_QMtestEx, i32 0, i32 0), i64 1, i8*** @_QMtestEx.cache)
!LLVMIR-DAG:  [[LOAD0:%.*]] = bitcast i8* [[CACHE0]] to [1 x i8]*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP1:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB1:[0-9]+]])
!LLVMIR-DAG:  [[CACHE1:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB1:[0-9]+]], i32 [[TMP1:%.*]], i8* bitcast ([5 x i32]* @_QMtestEy to i8*), i64 20, i8*** @_QMtestEy.cache)
!LLVMIR-DAG:  [[LOAD1:%.*]] = bitcast i8* [[CACHE1]] to [5 x i32]*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP2:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB2:[0-9]+]])
!LLVMIR-DAG:  [[CACHE2:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB2:[0-9]+]], i32 [[TMP2:%.*]], i8* getelementptr inbounds ([5 x [5 x i8]], [5 x [5 x i8]]* @_QMtestEz, i32 0, i32 0, i32 0), i64 25, i8*** @_QMtestEz.cache)
!LLVMIR-DAG:  [[LOAD2:%.*]] = bitcast i8* [[CACHE2]] to [5 x [5 x i8]]*, !dbg !{{.*}}
!LLVMIR-DAG:  {{.*}} = bitcast [1 x i8]* [[LOAD0]] to i8*, !dbg !{{.*}}
!LLVMIR-DAG:  {{.*}} = insertvalue { [5 x i32]*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } { [5 x i32]* undef, i64 4, i32 20180515, i8 1, i8 9, i8 0, i8 0
!LLVMIR-DAG:  {{.*}} = insertvalue { [5 x [5 x i8]]*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } { [5 x [5 x i8]]* undef, i64 5, i32 20180515, i8 1, i8 40, i8 0, i8 0
  end
end
