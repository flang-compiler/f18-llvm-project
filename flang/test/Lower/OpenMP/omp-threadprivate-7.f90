! This test checks lowering of OpenMP Threadprivate Directive.
! Test for threadprivate variable usage across module and main program and
! the module and main program are in one file.

! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | tco | FileCheck %s --check-prefix=LLVMIR

!FIRDialect-DAG: fir.global common @_QBblk(dense<0> : vector<24xi8>) : !fir.array<24xi8>
!FIRDialect-DAG: fir.global @_QMtestEy : f32 {

!LLVMIR-DAG: @_QBblk = common global [24 x i8] zeroinitializer
!LLVMIR-DAG: @_QMtestEy = global float undef
!LLVMIR-DAG: @_QBblk.cache = common global i8** null
!LLVMIR-DAG: @_QMtestEy.cache = common global i8** null

module test
  integer :: x
  real :: y, z(5)
  common /blk/ x, z

  !$omp threadprivate(y, /blk/)

contains
  subroutine sub()
! FIRDialect-LABEL: @_QMtestPsub
!FIRDialect-DAG:   [[ADDR0:%.*]] = fir.address_of(@_QBblk) : !fir.ref<!fir.array<24xi8>>
!FIRDialect-DAG:   [[NEWADDR0:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.array<24xi8>> -> !fir.ref<!fir.array<24xi8>>
!FIRDialect-DAG:   [[ADDR1:%.*]] = fir.address_of(@_QMtestEy) : !fir.ref<f32>
!FIRDialect-DAG:   [[NEWADDR1:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<f32> -> !fir.ref<f32>

    !$omp parallel
!FIRDialect-DAG:    [[ADDR2:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.array<24xi8>> -> !fir.ref<!fir.array<24xi8>>
!FIRDialect-DAG:    [[ADDR3:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<f32> -> !fir.ref<f32>
!FIRDialect-DAG:    [[ADDR4:%.*]] = fir.convert [[ADDR2]] : (!fir.ref<!fir.array<24xi8>>) -> !fir.ref<!fir.array<?xi8>>
!FIRDialect-DAG:    [[ADDR5:%.*]] = fir.coordinate_of [[ADDR4]], %{{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!FIRDialect-DAG:    [[ADDR6:%.*]] = fir.convert [[ADDR5:%.*]] : (!fir.ref<i8>) -> !fir.ref<i32>
!FIRDialect-DAG:    [[ADDR7:%.*]] = fir.convert [[ADDR2]] : (!fir.ref<!fir.array<24xi8>>) -> !fir.ref<!fir.array<?xi8>>
!FIRDialect-DAG:    [[ADDR8:%.*]] = fir.coordinate_of [[ADDR7]], %{{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!FIRDialect-DAG:    [[ADDR9:%.*]] = fir.convert [[ADDR8:%.*]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<5xf32>>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR6]] : !fir.ref<i32>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR3]] : !fir.ref<f32>
!FIRDialect-DAG:    %{{.*}} = fir.embox [[ADDR9]](%{{.*}}) : (!fir.ref<!fir.array<5xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<5xf32>>

! LLVMIR-LABEL: @_QMtestPsub..omp_par
! LLVMIR-LABEL: omp.par.region{{.*}}
!LLVMIR-DAG:  [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB0:[0-9]+]])
!LLVMIR-DAG:  [[CACHE0:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB0:[0-9]+]], i32 [[TMP0:%.*]], i8* getelementptr inbounds ([24 x i8], [24 x i8]* @_QBblk, i32 0, i32 0), i64 24, i8*** @_QBblk.cache)
!LLVMIR-DAG:  [[INS4:%.*]] = bitcast i8* [[CACHE0]] to [24 x i8]*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP1:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB1:[0-9]+]])
!LLVMIR-DAG:  [[CACHE1:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB1:[0-9]+]], i32 [[TMP1:%.*]], i8* bitcast (float* @_QMtestEy to i8*), i64 4, i8*** @_QMtestEy.cache)
!LLVMIR-DAG:  [[INS7:%.*]] = bitcast i8* [[CACHE1]] to float*, !dbg !{{.*}}
!LLVMIR-DAG:  [[INS8:%.*]] = bitcast [24 x i8]* [[INS4]] to i8*, !dbg !{{.*}}
!LLVMIR-DAG:  [[INS9:%.*]] = getelementptr i8, i8* [[INS8]], i64 0, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = bitcast i8* [[INS9]] to i32*, !dbg !{{.*}}
!LLVMIR-DAG:  [[INS11:%.*]] = getelementptr i8, i8* [[INS8]], i64 4, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = bitcast i8* [[INS11]] to [5 x float]*, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = load float, float* [[INS7]], align 4, !dbg !{{.*}}
      print *, x, y, z
    !$omp end parallel
  end
end

program main
  use test
  integer :: x1
  real :: z1(5)
  common /blk/ x1, z1

  !$omp threadprivate(/blk/)

  call sub()

! FIRDialect-LABEL: @_QQmain()
!FIRDialect-DAG:  [[ADDR0:%.*]] = fir.address_of(@_QBblk) : !fir.ref<!fir.array<24xi8>>
!FIRDialect-DAG:  [[NEWADDR0:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.array<24xi8>> -> !fir.ref<!fir.array<24xi8>>
!FIRDialect-DAG:  [[ADDR1:%.*]] = fir.address_of(@_QBblk) : !fir.ref<!fir.array<24xi8>>
!FIRDialect-DAG:  [[NEWADDR1:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<!fir.array<24xi8>> -> !fir.ref<!fir.array<24xi8>>
!FIRDialect-DAG:  [[ADDR2:%.*]] = fir.address_of(@_QMtestEy) : !fir.ref<f32>
!FIRDialect-DAG:  [[NEWADDR2:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<f32> -> !fir.ref<f32>

  !$omp parallel
!FIRDialect-DAG:    [[ADDR4:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<!fir.array<24xi8>> -> !fir.ref<!fir.array<24xi8>>
!FIRDialect-DAG:    [[ADDR5:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<f32> -> !fir.ref<f32>
!FIRDialect-DAG:    [[ADDR6:%.*]] = fir.convert [[ADDR4]] : (!fir.ref<!fir.array<24xi8>>) -> !fir.ref<!fir.array<?xi8>>
!FIRDialect-DAG:    [[ADDR7:%.*]] = fir.coordinate_of [[ADDR6]], %{{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!FIRDialect-DAG:    [[ADDR8:%.*]] = fir.convert [[ADDR7:%.*]] : (!fir.ref<i8>) -> !fir.ref<i32>
!FIRDialect-DAG:    [[ADDR9:%.*]] = fir.convert [[ADDR4]] : (!fir.ref<!fir.array<24xi8>>) -> !fir.ref<!fir.array<?xi8>>
!FIRDialect-DAG:    [[ADDR10:%.*]] = fir.coordinate_of [[ADDR9]], %{{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!FIRDialect-DAG:    [[ADDR11:%.*]] = fir.convert [[ADDR10:%.*]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<5xf32>>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR8]] : !fir.ref<i32>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR5]] : !fir.ref<f32>
!FIRDialect-DAG:    %{{.*}} = fir.embox [[ADDR11]](%{{.*}}) : (!fir.ref<!fir.array<5xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<5xf32>>

! LLVMIR-LABEL: @_QQmain..omp_par
! LLVMIR-LABEL: omp.par.region{{.*}}
!LLVMIR-DAG:  [[TMP1:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB1:[0-9]+]])
!LLVMIR-DAG:  [[INS6:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB1:[0-9]+]], i32 [[TMP1:%.*]], i8* getelementptr inbounds ([24 x i8], [24 x i8]* @_QBblk, i32 0, i32 0), i64 24, i8*** @_QBblk.cache)
!LLVMIR-DAG:  [[INS7:%.*]] = bitcast i8* [[INS6]] to [24 x i8]*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP2:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB2:[0-9]+]])
!LLVMIR-DAG:  [[INS9:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB2:[0-9]+]], i32 [[TMP1:%.*]], i8* bitcast (float* @_QMtestEy to i8*), i64 4, i8*** @_QMtestEy.cache)
!LLVMIR-DAG:  [[INS10:%.*]] = bitcast i8* [[INS9]] to float*, !dbg !{{.*}}
!LLVMIR-DAG:  [[INS11:%.*]] = bitcast [24 x i8]* [[INS7]] to i8*, !dbg !{{.*}}
!LLVMIR-DAG:  [[INS12:%.*]] = getelementptr i8, i8* [[INS11]], i64 0, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = bitcast i8* [[INS12]] to i32*, !dbg !{{.*}}
!LLVMIR-DAG:  [[INS14:%.*]] = getelementptr i8, i8* [[INS11]], i64 4, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = bitcast i8* [[INS14]] to [5 x float]*, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = load float, float* [[INS10]], align 4, !dbg !{{.*}}
    print *, x1, y, z1
  !$omp end parallel

end
