! This test checks passing an array slice in a parallel region

! RUN: bbc -fopenmp %s -o - | tco 2>&1 | FileCheck %s

! CHECK-LABEL: @_QPsb1..omp_par
! CHECK-LABEL: omp.par.region1
! CHECK:   %[[SB1_MAT:.*]] = alloca { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8
! CHECK:   %[[SB1_MAT1:.*]] = getelementptr { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %[[SB1_MAT]], i32 0, i32 0
! CHECK:   %[[SB1_MAT2:.*]] = load i32*, i32** %[[SB1_MAT1]], align 8
! CHECK:   %[[SB1_MAT3:.*]] = bitcast i32* %[[SB1_MAT2]] to [3 x i32]*
! CHECK:   call void @_QPouter_src_calc([3 x i32]* %[[SB1_MAT3]])

subroutine sb1
  IMPLICIT NONE
  INTEGER, DIMENSION(3, 3) :: mat
  INTEGER :: k

  !$OMP PARALLEL
  DO k = 1, 2
     CALL outer_src_calc (  mat(:,k) )
  END DO
  !$OMP END PARALLEL
end subroutine

! CHECK-LABEL: @_QPsb2..omp_par
! CHECK-LABEL: omp.par.region1
! CHECK:   %[[SB2_MAT:.*]] = alloca { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8
! CHECK:   call i32 @__kmpc_master
! CHECK:   %[[SB2_MAT1:.*]] = getelementptr { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %[[SB2_MAT]], i32 0, i32 0
! CHECK:   %[[SB2_MAT2:.*]] = load i32*, i32** %[[SB2_MAT1]], align 8
! CHECK:   %[[SB2_MAT3:.*]] = bitcast i32* %[[SB2_MAT2]] to [3 x i32]*
! CHECK:   call void @_QPouter_src_calc([3 x i32]* %[[SB2_MAT3]])
subroutine sb2
  IMPLICIT NONE
  INTEGER, DIMENSION(3, 3) :: mat
  INTEGER :: k

  !$OMP PARALLEL
  !$OMP MASTER
  DO k = 1, 2
     CALL outer_src_calc (  mat(:,k) )
  END DO
  !$OMP END MASTER
  !$OMP END PARALLEL
end subroutine

! CHECK-LABEL: @_QPsb3
! CHECK:   %[[SB3_MAT:.*]] = alloca { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8
! CHECK:   call i32 @__kmpc_master
! CHECK:   %[[SB3_MAT1:.*]] = getelementptr { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %[[SB3_MAT]], i32 0, i32 0
! CHECK:   %[[SB3_MAT2:.*]] = load i32*, i32** %[[SB3_MAT1]], align 8
! CHECK:   %[[SB3_MAT3:.*]] = bitcast i32* %[[SB3_MAT2]] to [3 x i32]*
! CHECK:   call void @_QPouter_src_calc([3 x i32]* %[[SB3_MAT3]])
subroutine sb3
  IMPLICIT NONE
  INTEGER, DIMENSION(3, 3) :: mat
  INTEGER :: k

  !$OMP MASTER
  DO k = 1, 2
     CALL outer_src_calc (  mat(:,k) )
  END DO
  !$OMP END MASTER
end subroutine
