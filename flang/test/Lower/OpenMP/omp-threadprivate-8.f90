! This test checks lowering of OpenMP Threadprivate Directive.
! Test for threadprivate variable usage across module and main program and
! the module and main program are split in two file to test use association.

! RUN: split-file %s %t
! RUN: bbc -emit-fir -fopenmp %t/mod.f90 -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect-MOD
! RUN: bbc -emit-fir -fopenmp %t/mod.f90 -o - | \
! RUN:   tco | FileCheck %s --check-prefix=LLVMIR-MOD
! RUN: bbc -emit-fir -fopenmp %t/use.f90 -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect-USE
! RUN: bbc -emit-fir -fopenmp %t/use.f90 -o - | \
! RUN:   tco | FileCheck %s --check-prefix=LLVMIR-USE


!--- mod.f90

module test
  integer :: x
  real :: y, z(5)
  common /blk/ x, z

  !$omp threadprivate(y, /blk/)

contains
  subroutine sub()
    !$omp parallel
      print *, x, y, z
    !$omp end parallel
  end
end

!FIRDialect-MOD-DAG: fir.global common @_QBblk(dense<0> : vector<24xi8>) : !fir.array<24xi8>
!FIRDialect-MOD-DAG: fir.global @_QMtestEy : f32 {

! FIRDialect-MOD-LABEL: @_QMtestPsub()
!FIRDialect-MOD-DAG:   [[ADDR0:%.*]] = fir.address_of(@_QBblk) : !fir.ref<!fir.array<24xi8>>
!FIRDialect-MOD-DAG:   [[NEWADDR0:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.array<24xi8>> -> !fir.ref<!fir.array<24xi8>>
!FIRDialect-MOD-DAG:   [[ADDR1:%.*]] = fir.address_of(@_QMtestEy) : !fir.ref<f32>
!FIRDialect-MOD-DAG:   [[NEWADDR1:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<f32> -> !fir.ref<f32>
!FIRDialect-MOD-DAG:    [[ADDR2:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.array<24xi8>> -> !fir.ref<!fir.array<24xi8>>
!FIRDialect-MOD-DAG:    [[ADDR3:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<f32> -> !fir.ref<f32>
!FIRDialect-MOD-DAG:    [[ADDR4:%.*]] = fir.convert [[ADDR2]] : (!fir.ref<!fir.array<24xi8>>) -> !fir.ref<!fir.array<?xi8>>
!FIRDialect-MOD-DAG:    [[ADDR5:%.*]] = fir.coordinate_of [[ADDR4]], %{{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!FIRDialect-MOD-DAG:    [[ADDR6:%.*]] = fir.convert [[ADDR5:%.*]] : (!fir.ref<i8>) -> !fir.ref<i32>
!FIRDialect-MOD-DAG:    [[ADDR7:%.*]] = fir.convert [[ADDR2]] : (!fir.ref<!fir.array<24xi8>>) -> !fir.ref<!fir.array<?xi8>>
!FIRDialect-MOD-DAG:    [[ADDR8:%.*]] = fir.coordinate_of [[ADDR7]], %{{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!FIRDialect-MOD-DAG:    [[ADDR9:%.*]] = fir.convert [[ADDR8:%.*]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<5xf32>>
!FIRDialect-MOD-DAG:    %{{.*}} = fir.load [[ADDR6]] : !fir.ref<i32>
!FIRDialect-MOD-DAG:    %{{.*}} = fir.load [[ADDR3]] : !fir.ref<f32>
!FIRDialect-MOD-DAG:    %{{.*}} = fir.embox [[ADDR9]](%{{.*}}) : (!fir.ref<!fir.array<5xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<5xf32>>

!LLVMIR-MOD-DAG: @_QBblk = common global [24 x i8] zeroinitializer
!LLVMIR-MOD-DAG: @_QMtestEy = global float undef
!LLVMIR-MOD-DAG: @_QBblk.cache = common global i8** null
!LLVMIR-MOD-DAG: @_QMtestEy.cache = common global i8** null

! LLVMIR-MOD-LABEL: @_QMtestPsub..omp_par
! LLVMIR-MOD-LABEL: omp.par.region{{.*}}
!LLVMIR-MOD-DAG:  [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB0:[0-9]+]])
!LLVMIR-MOD-DAG:  [[CACHE0:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB0:[0-9]+]], i32 [[TMP0:%.*]], i8* getelementptr inbounds ([24 x i8], [24 x i8]* @_QBblk, i32 0, i32 0), i64 24, i8*** @_QBblk.cache)
!LLVMIR-MOD-DAG:  [[INS4:%.*]] = bitcast i8* [[CACHE0]] to [24 x i8]*, !dbg !{{.*}}
!LLVMIR-MOD-DAG:  [[TMP1:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB1:[0-9]+]])
!LLVMIR-MOD-DAG:  [[CACHE1:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB1:[0-9]+]], i32 [[TMP1:%.*]], i8* bitcast (float* @_QMtestEy to i8*), i64 4, i8*** @_QMtestEy.cache)
!LLVMIR-MOD-DAG:  [[INS7:%.*]] = bitcast i8* [[CACHE1]] to float*, !dbg !{{.*}}
!LLVMIR-MOD-DAG:  [[INS8:%.*]] = bitcast [24 x i8]* [[INS4]] to i8*, !dbg !{{.*}}
!LLVMIR-MOD-DAG:  [[INS9:%.*]] = getelementptr i8, i8* [[INS8]], i64 0, !dbg !{{.*}}
!LLVMIR-MOD-DAG:  %{{.*}} = bitcast i8* [[INS9]] to i32*, !dbg !{{.*}}
!LLVMIR-MOD-DAG:  [[INS11:%.*]] = getelementptr i8, i8* [[INS8]], i64 4, !dbg !{{.*}}
!LLVMIR-MOD-DAG:  %{{.*}} = bitcast i8* [[INS11]] to [5 x float]*, !dbg !{{.*}}
!LLVMIR-MOD-DAG:  %{{.*}} = load float, float* [[INS7]], align 4, !dbg !{{.*}}


!--- use.f90

program main
  use test
  integer :: x1
  real :: z1(5)
  common /blk/ x1, z1

  !$omp threadprivate(/blk/)

  call sub()

  !$omp parallel
    print *, x1, y, z1
  !$omp end parallel

end

! FIRDialect-USE-LABEL: @_QQmain
!FIRDialect-USE-DAG:  [[ADDR0:%.*]] = fir.address_of(@_QBblk) : !fir.ref<!fir.array<24xi8>>
!FIRDialect-USE-DAG:  [[NEWADDR0:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.array<24xi8>> -> !fir.ref<!fir.array<24xi8>>
!FIRDialect-USE-DAG:  [[ADDR1:%.*]] = fir.address_of(@_QBblk) : !fir.ref<!fir.array<24xi8>>
!FIRDialect-USE-DAG:  [[NEWADDR1:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<!fir.array<24xi8>> -> !fir.ref<!fir.array<24xi8>>
!FIRDialect-USE-DAG:  [[ADDR2:%.*]] = fir.address_of(@_QMtestEy) : !fir.ref<f32>
!FIRDialect-USE-DAG:  [[NEWADDR2:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<f32> -> !fir.ref<f32>
!FIRDialect-USE-DAG:    [[ADDR4:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<!fir.array<24xi8>> -> !fir.ref<!fir.array<24xi8>>
!FIRDialect-USE-DAG:    [[ADDR5:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<f32> -> !fir.ref<f32>
!FIRDialect-USE-DAG:    [[ADDR6:%.*]] = fir.convert [[ADDR4]] : (!fir.ref<!fir.array<24xi8>>) -> !fir.ref<!fir.array<?xi8>>
!FIRDialect-USE-DAG:    [[ADDR7:%.*]] = fir.coordinate_of [[ADDR6]], %{{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!FIRDialect-USE-DAG:    [[ADDR8:%.*]] = fir.convert [[ADDR7:%.*]] : (!fir.ref<i8>) -> !fir.ref<i32>
!FIRDialect-USE-DAG:    [[ADDR9:%.*]] = fir.convert [[ADDR4]] : (!fir.ref<!fir.array<24xi8>>) -> !fir.ref<!fir.array<?xi8>>
!FIRDialect-USE-DAG:    [[ADDR10:%.*]] = fir.coordinate_of [[ADDR9]], %{{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!FIRDialect-USE-DAG:    [[ADDR11:%.*]] = fir.convert [[ADDR10:%.*]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<5xf32>>
!FIRDialect-USE-DAG:    %{{.*}} = fir.load [[ADDR8]] : !fir.ref<i32>
!FIRDialect-USE-DAG:    %{{.*}} = fir.load [[ADDR5]] : !fir.ref<f32>
!FIRDialect-USE-DAG:    %{{.*}} = fir.embox [[ADDR11]](%{{.*}}) : (!fir.ref<!fir.array<5xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<5xf32>>

!FIRDialect-USE-DAG: fir.global common @_QBblk(dense<0> : vector<24xi8>) : !fir.array<24xi8>
!FIRDialect-USE-DAG: fir.global @_QMtestEy : f32

!LLVMIR-USE-DAG: @_QBblk = common global [24 x i8] zeroinitializer
!LLVMIR-USE-DAG: @_QMtestEy = external global float
!LLVMIR-USE-DAG: @_QBblk.cache = common global i8** null
!LLVMIR-USE-DAG: @_QMtestEy.cache = common global i8** null

! LLVMIR-USE-LABEL: @_QQmain..omp_par
! LLVMIR-USE-LABEL: omp.par.region{{.*}}
!LLVMIR-USE-DAG:  [[TMP1:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB1:[0-9]+]])
!LLVMIR-USE-DAG:  [[INS6:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB1:[0-9]+]], i32 [[TMP1:%.*]], i8* getelementptr inbounds ([24 x i8], [24 x i8]* @_QBblk, i32 0, i32 0), i64 24, i8*** @_QBblk.cache)
!LLVMIR-USE-DAG:  [[INS7:%.*]] = bitcast i8* [[INS6]] to [24 x i8]*, !dbg !{{.*}}
!LLVMIR-USE-DAG:  [[TMP2:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB2:[0-9]+]])
!LLVMIR-USE-DAG:  [[INS9:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB2:[0-9]+]], i32 [[TMP1:%.*]], i8* bitcast (float* @_QMtestEy to i8*), i64 4, i8*** @_QMtestEy.cache)
!LLVMIR-USE-DAG:  [[INS10:%.*]] = bitcast i8* [[INS9]] to float*, !dbg !{{.*}}
!LLVMIR-USE-DAG:  [[INS11:%.*]] = bitcast [24 x i8]* [[INS7]] to i8*, !dbg !{{.*}}
!LLVMIR-USE-DAG:  [[INS12:%.*]] = getelementptr i8, i8* [[INS11]], i64 0, !dbg !{{.*}}
!LLVMIR-USE-DAG:  %{{.*}} = bitcast i8* [[INS12]] to i32*, !dbg !{{.*}}
!LLVMIR-USE-DAG:  [[INS14:%.*]] = getelementptr i8, i8* [[INS11]], i64 4, !dbg !{{.*}}
!LLVMIR-USE-DAG:  %{{.*}} = bitcast i8* [[INS14]] to [5 x float]*, !dbg !{{.*}}
!LLVMIR-USE-DAG:  %{{.*}} = load float, float* [[INS10]], align 4, !dbg !{{.*}}
