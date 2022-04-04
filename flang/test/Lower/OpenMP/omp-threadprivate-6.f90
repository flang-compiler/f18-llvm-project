! This test checks lowering of OpenMP Threadprivate Directive.
! Test for common block.

! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | tco | FileCheck %s --check-prefix=LLVMIR

module test
  integer:: a
  real :: b(2)
  complex, pointer :: c, d(:)
  character(5) :: e, f(2)
  common /blk/ a, b, c, d, e, f

  !$omp threadprivate(/blk/)

!FIRDialect: fir.global common @_QBblk(dense<0> : vector<103xi8>) : !fir.array<103xi8>

!LLVMIR-DAG: @_QBblk = common global [103 x i8] zeroinitializer
!LLVMIR-DAG: @_QBblk.cache = common global i8** null

contains
  subroutine sub()
!FIRDialect:  [[ADDR0:%.*]] = fir.address_of(@_QBblk) : !fir.ref<!fir.array<103xi8>>
!FIRDialect:  [[NEWADDR0:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.array<103xi8>> -> !fir.ref<!fir.array<103xi8>>
!FIRDialect-DAG:  [[ADDR1:%.*]] = fir.convert [[NEWADDR0]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!FIRDialect-DAG:  [[C0:%.*]] = arith.constant 0 : index
!FIRDialect-DAG:  [[ADDR2:%.*]] = fir.coordinate_of [[ADDR1]], [[C0]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!FIRDialect-DAG:  [[ADDR3:%.*]] = fir.convert [[ADDR2]] : (!fir.ref<i8>) -> !fir.ref<i32>
!FIRDialect-DAG:  [[ADDR4:%.*]] = fir.convert [[NEWADDR0]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!FIRDialect-DAG:  [[C1:%.*]] = arith.constant 4 : index
!FIRDialect-DAG:  [[ADDR5:%.*]] = fir.coordinate_of [[ADDR4]], [[C1]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!FIRDialect-DAG:  [[ADDR6:%.*]] = fir.convert [[ADDR5]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<2xf32>>
!FIRDialect-DAG:  [[ADDR7:%.*]] = fir.convert [[NEWADDR0]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!FIRDialect-DAG:  [[C2:%.*]] = arith.constant 16 : index
!FIRDialect-DAG:  [[ADDR8:%.*]] = fir.coordinate_of [[ADDR7]], [[C2]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!FIRDialect-DAG:  [[ADDR9:%.*]] = fir.convert [[ADDR8]] : (!fir.ref<i8>) -> !fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>
!FIRDialect-DAG:  [[ADDR10:%.*]] = fir.convert [[NEWADDR0]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!FIRDialect-DAG:  [[C3:%.*]] = arith.constant 40 : index
!FIRDialect-DAG:  [[ADDR11:%.*]] = fir.coordinate_of [[ADDR10]], [[C3]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!FIRDialect-DAG:  [[ADDR12:%.*]] = fir.convert [[ADDR11]] : (!fir.ref<i8>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>>
!FIRDialect-DAG:  [[ADDR13:%.*]] = fir.convert [[NEWADDR0]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!FIRDialect-DAG:  [[C4:%.*]] = arith.constant 88 : index
!FIRDialect-DAG:  [[ADDR14:%.*]] = fir.coordinate_of [[ADDR13]], [[C4]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!FIRDialect-DAG:  [[ADDR15:%.*]] = fir.convert [[ADDR14]] : (!fir.ref<i8>) -> !fir.ref<!fir.char<1,5>>
!FIRDialect-DAG:  [[ADDR16:%.*]] = fir.convert [[NEWADDR0]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!FIRDialect-DAG:  [[C5:%.*]] = arith.constant 93 : index
!FIRDialect-DAG:  [[ADDR17:%.*]] = fir.coordinate_of [[ADDR16]], [[C5]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!FIRDialect-DAG:  [[ADDR18:%.*]] = fir.convert [[ADDR17]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<2x!fir.char<1,5>>>
!FIRDialect-DAG:  %{{.*}} = fir.load [[ADDR3]] : !fir.ref<i32>
!FIRDialect-DAG:  %{{.*}} = fir.embox [[ADDR6]](%{{.*}}) : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf32>>
!FIRDialect-DAG:  %{{.*}} = fir.load [[ADDR9]] : !fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>
!FIRDialect-DAG:  %{{.*}} = fir.load [[ADDR12]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>>
!FIRDialect-DAG:  %{{.*}} = fir.convert [[ADDR15]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
!FIRDialect-DAG:  %{{.*}} = fir.embox [[ADDR18]](%{{.*}}) : (!fir.ref<!fir.array<2x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.char<1,5>>>

!LLVMIR:  [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB0:[0-9]+]])
!LLVMIR:  [[TMP5:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB0]], i32 [[TMP0]], i8* getelementptr inbounds ([103 x i8], [103 x i8]* @_QBblk, i32 0, i32 0), i64 103, i8*** @_QBblk.cache)
!LLVMIR:  [[TMP6:%.*]] = bitcast i8* [[TMP5]] to [103 x i8]*, !dbg !{{.*}}
!LLVMIR:  [[TMP7:%.*]] = bitcast [103 x i8]* [[TMP6]] to i8*, !dbg !{{.*}}
!LLVMIR:  [[TMP8:%.*]] = getelementptr i8, i8* [[TMP7]], i64 0, !dbg !{{.*}}
!LLVMIR:  [[TMP9:%.*]] = bitcast i8* [[TMP8]] to i32*, !dbg !{{.*}}
!LLVMIR:  [[TMP10:%.*]] = getelementptr i8, i8* [[TMP7]], i64 4, !dbg !{{.*}}
!LLVMIR:  [[TMP11:%.*]] = bitcast i8* [[TMP10]] to [2 x float]*, !dbg !{{.*}}
!LLVMIR:  [[TMP12:%.*]] = getelementptr i8, i8* [[TMP7]], i64 16, !dbg !{{.*}}
!LLVMIR:  [[TMP13:%.*]] = bitcast i8* [[TMP12]] to { { float, float }*, i64, i32, i8, i8, i8, i8 }*, !dbg !{{.*}}
!LLVMIR:  [[TMP14:%.*]] = getelementptr i8, i8* [[TMP7]], i64 40, !dbg !{{.*}}
!LLVMIR:  [[TMP15:%.*]] = bitcast i8* [[TMP14]] to { { float, float }*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }*, !dbg !{{.*}}
!LLVMIR:  [[TMP16:%.*]] = getelementptr i8, i8* [[TMP7]], i64 88, !dbg !{{.*}}
!LLVMIR:  [[TMP17:%.*]] = bitcast i8* [[TMP16]] to [5 x i8]*, !dbg !{{.*}}
!LLVMIR:  [[TMP18:%.*]] = getelementptr i8, i8* [[TMP7]], i64 93, !dbg !{{.*}}
!LLVMIR:  [[TMP19:%.*]] = bitcast i8* [[TMP18]] to [2 x [5 x i8]]*, !dbg !{{.*}}
!LLVMIR:  %{{.*}} = load i32, i32* [[TMP9]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = insertvalue { [2 x float]*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } { [2 x float]* undef, i64 4, i32 20180515, i8 1, i8 27, i8 0, i8 0
!LLVMIR-DAG:  %{{.*}} = getelementptr { { float, float }*, i64, i32, i8, i8, i8, i8 }, { { float, float }*, i64, i32, i8, i8, i8, i8 }* [[TMP13]], i32 0, i32 0, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = getelementptr { { float, float }*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, { { float, float }*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* [[TMP15]], i32 0, i32 7, i64 0, i32 0, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = bitcast [5 x i8]* [[TMP17]] to i8*, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = insertvalue { [2 x [5 x i8]]*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } { [2 x [5 x i8]]* undef, i64 5, i32 20180515, i8 1, i8 40, i8 0, i8 0
    print *, a, b, c, d, e, f

    !$omp parallel
!FIRDialect:    [[ADDR77:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.array<103xi8>> -> !fir.ref<!fir.array<103xi8>>
!FIRDialect-DAG:    [[ADDR78:%.*]] = fir.convert [[ADDR77]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!FIRDialect-DAG:    [[ADDR79:%.*]] = fir.coordinate_of [[ADDR78]], [[C0:%.*]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!FIRDialect-DAG:    [[ADDR80:%.*]] = fir.convert [[ADDR79:%.*]] : (!fir.ref<i8>) -> !fir.ref<i32>
!FIRDialect-DAG:    [[ADDR81:%.*]] = fir.convert [[ADDR77]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!FIRDialect-DAG:    [[ADDR82:%.*]] = fir.coordinate_of [[ADDR81]], [[C1:%.*]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!FIRDialect-DAG:    [[ADDR83:%.*]] = fir.convert [[ADDR82:%.*]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<2xf32>>
!FIRDialect-DAG:    [[ADDR84:%.*]] = fir.convert [[ADDR77]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!FIRDialect-DAG:    [[ADDR85:%.*]] = fir.coordinate_of [[ADDR84]], [[C2:%.*]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!FIRDialect-DAG:    [[ADDR86:%.*]] = fir.convert [[ADDR85:%.*]] : (!fir.ref<i8>) -> !fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>
!FIRDialect-DAG:    [[ADDR87:%.*]] = fir.convert [[ADDR77]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!FIRDialect-DAG:    [[ADDR88:%.*]] = fir.coordinate_of [[ADDR87]], [[C3:%.*]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!FIRDialect-DAG:    [[ADDR89:%.*]] = fir.convert [[ADDR88:%.*]] : (!fir.ref<i8>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>>
!FIRDialect-DAG:    [[ADDR90:%.*]] = fir.convert [[ADDR77]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!FIRDialect-DAG:    [[ADDR91:%.*]] = fir.coordinate_of [[ADDR90]], [[C4:%.*]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!FIRDialect-DAG:    [[ADDR92:%.*]] = fir.convert [[ADDR91:%.*]] : (!fir.ref<i8>) -> !fir.ref<!fir.char<1,5>>
!FIRDialect-DAG:    [[ADDR93:%.*]] = fir.convert [[ADDR77]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!FIRDialect-DAG:    [[ADDR94:%.*]] = fir.coordinate_of [[ADDR93]], [[C5:%.*]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!FIRDialect-DAG:    [[ADDR95:%.*]] = fir.convert [[ADDR94:%.*]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<2x!fir.char<1,5>>>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR80]] : !fir.ref<i32>
!FIRDialect-DAG:    %{{.*}} = fir.embox [[ADDR83]](%{{.*}}) : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf32>>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR86]] : !fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>
!FIRDialect-DAG:    %{{.*}} = fir.load [[ADDR89]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>>
!FIRDialect-DAG:    %{{.*}} = fir.convert [[ADDR92]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
!FIRDialect-DAG:    %{{.*}} = fir.embox [[ADDR95]](%{{.*}}) : (!fir.ref<!fir.array<2x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.char<1,5>>>
      print *, a, b, c, d, e, f
    !$omp end parallel

!FIRDialect-DAG:  %{{.*}} = fir.load [[ADDR3]] : !fir.ref<i32>
!FIRDialect-DAG:  %{{.*}} = fir.embox [[ADDR6]](%{{.*}}) : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf32>>
!FIRDialect-DAG:  %{{.*}} = fir.load [[ADDR9]] : !fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>
!FIRDialect-DAG:  %{{.*}} = fir.load [[ADDR12]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>>
!FIRDialect-DAG:  %{{.*}} = fir.convert [[ADDR15]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
!FIRDialect-DAG:  %{{.*}} = fir.embox [[ADDR18]](%{{.*}}) : (!fir.ref<!fir.array<2x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.char<1,5>>>

!LLVMIR-DAG:  %{{.*}} = load i32, i32* [[TMP9]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = getelementptr { { float, float }*, i64, i32, i8, i8, i8, i8 }, { { float, float }*, i64, i32, i8, i8, i8, i8 }* [[TMP13]], i32 0, i32 0, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = getelementptr { { float, float }*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, { { float, float }*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* [[TMP15]], i32 0, i32 7, i64 0, i32 0, !dbg !{{.*}}
    print *, a, b, c, d, e, f

! LLVMIR-LABEL: omp.par.region{{.*}}
!LLVMIR:  [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB0:[0-9]+]])
!LLVMIR:  [[TMP5:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB0]], i32 [[TMP0]], i8* getelementptr inbounds ([103 x i8], [103 x i8]* @_QBblk, i32 0, i32 0), i64 103, i8*** @_QBblk.cache)
!LLVMIR-DAG:  [[TMP6:%.*]] = bitcast i8* [[TMP5]] to [103 x i8]*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP7:%.*]] = bitcast [103 x i8]* [[TMP6]] to i8*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP8:%.*]] = getelementptr i8, i8* [[TMP7]], i64 0, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP9:%.*]] = bitcast i8* [[TMP8:%.*]] to i32*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP10:%.*]] = getelementptr i8, i8* [[TMP7]], i64 4, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP11:%.*]] = bitcast i8* [[TMP10:%.*]] to [2 x float]*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP12:%.*]] = getelementptr i8, i8* [[TMP7]], i64 16, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP13:%.*]] = bitcast i8* [[TMP12:%.*]] to { { float, float }*, i64, i32, i8, i8, i8, i8 }*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP14:%.*]] = getelementptr i8, i8* [[TMP7]], i64 40, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP15:%.*]] = bitcast i8* [[TMP14:%.*]] to { { float, float }*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP16:%.*]] = getelementptr i8, i8* [[TMP7]], i64 88, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP17:%.*]] = bitcast i8* [[TMP16:%.*]] to [5 x i8]*, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP18:%.*]] = getelementptr i8, i8* [[TMP7]], i64 93, !dbg !{{.*}}
!LLVMIR-DAG:  [[TMP19:%.*]] = bitcast i8* [[TMP18:%.*]] to [2 x [5 x i8]]*, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = load i32, i32* [[TMP9]], align 4, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = insertvalue { [2 x float]*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } { [2 x float]* undef, i64 4, i32 20180515, i8 1, i8 27, i8 0, i8 0
!LLVMIR-DAG:  %{{.*}} = getelementptr { { float, float }*, i64, i32, i8, i8, i8, i8 }, { { float, float }*, i64, i32, i8, i8, i8, i8 }* [[TMP13]], i32 0, i32 0, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = getelementptr { { float, float }*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, { { float, float }*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* [[TMP15]], i32 0, i32 7, i64 0, i32 0, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = bitcast [5 x i8]* [[TMP17]] to i8*, !dbg !{{.*}}
!LLVMIR-DAG:  %{{.*}} = insertvalue { [2 x [5 x i8]]*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } { [2 x [5 x i8]]* undef, i64 5, i32 20180515, i8 1, i8 40, i8 0, i8 0
  end
end
