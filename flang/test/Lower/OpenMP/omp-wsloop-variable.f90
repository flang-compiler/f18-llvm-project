! This test checks lowering of OpenMP DO Directive(Worksharing) for different
! types of loop iteration variable, lower bound, upper bound, and step.

! RUN: bbc -fopenmp -emit-fir %s -o - 2>&1 | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

program wsloop_variable
  integer(kind=1) :: i1_lb, i1_ub
  integer(kind=2) :: i2, i2_ub, i2_s
  integer(kind=4) :: i4_s
  integer(kind=8) :: i8, i8_s
  integer(kind=16) :: i16, i16_lb
  real :: x

! FIRDialect:  OpenMP loop iteration variable cannot have more than 64 bits size and will be narrowed into 64 bits.

! FIRDialect:  [[TMP0:%.*]] = arith.constant 1 : i32
! FIRDialect:  [[TMP1:%.*]] = arith.constant 100 : i32
! FIRDialect:  [[TMP2:%.*]] = fir.convert [[TMP0]] : (i32) -> i64
! FIRDialect:  [[TMP3:%.*]] = fir.convert %{{.*}} : (i8) -> i64
! FIRDialect:  [[TMP4:%.*]] = fir.convert %{{.*}} : (i16) -> i64
! FIRDialect:  [[TMP5:%.*]] = fir.convert %{{.*}} : (i128) -> i64
! FIRDialect:  [[TMP6:%.*]] = fir.convert [[TMP1]] : (i32) -> i64
! FIRDialect:  [[TMP7:%.*]] = fir.convert %{{.*}} : (i32) -> i64
! FIRDialect:  omp.wsloop ([[TMP8:%.*]], [[TMP9:%.*]]) : i64 = ([[TMP2]], [[TMP5]]) to ([[TMP3]], [[TMP6]]) inclusive step ([[TMP4]], [[TMP7]]) collapse(2)  {
! FIRDialect:    [[TMP10:%.*]] = arith.addi [[TMP8]], [[TMP9]] : i64
! FIRDialect:    [[TMP11:%.*]] = fir.convert [[TMP10]] : (i64) -> f32
! FIRDialect:    fir.store [[TMP11]] to %{{.*}} : !fir.ref<f32>
! FIRDialect:    omp.yield
! FIRDialect:  }
  !$omp do collapse(2)
  do i2 = 1, i1_ub, i2_s
    do i8 = i16_lb, 100, i4_s
      x = i2 + i8
    end do
  end do
  !$omp end do

! FIRDialect:  [[TMP12:%.*]] = arith.constant 1 : i32
! FIRDialect:  [[TMP13:%.*]] = fir.convert %{{.*}} : (i8) -> i32
! FIRDialect:  [[TMP14:%.*]] = fir.convert %{{.*}} : (i64) -> i32
! FIRDialect:  omp.wsloop ([[TMP15:%.*]]) : i32 = ([[TMP12]]) to ([[TMP13]]) inclusive step ([[TMP14]])  {
! FIRDialect:    [[TMP16:%.*]] = fir.convert [[TMP15]] : (i32) -> f32
! FIRDialect:    fir.store [[TMP16]] to %{{.*}} : !fir.ref<f32>
! FIRDialect:    omp.yield
! FIRDialect:  }
  !$omp do
  do i2 = 1, i1_ub, i8_s
    x = i2
  end do
  !$omp end do

! FIRDialect:  [[TMP17:%.*]] = fir.convert %{{.*}} : (i8) -> i64
! FIRDialect:  [[TMP18:%.*]] = fir.convert %{{.*}} : (i16) -> i64
! FIRDialect:  [[TMP19:%.*]] = fir.convert %{{.*}} : (i32) -> i64
! FIRDialect:  omp.wsloop ([[TMP20:%.*]]) : i64 = ([[TMP17]]) to ([[TMP18]]) inclusive step ([[TMP19]])  {
! FIRDialect:    [[TMP21:%.*]] = fir.convert [[TMP20]] : (i64) -> f32
! FIRDialect:    fir.store [[TMP21]] to %{{.*}} : !fir.ref<f32>
! FIRDialect:    omp.yield
! FIRDialect:  }
  !$omp do
  do i16 = i1_lb, i2_ub, i4_s
    x = i16
  end do
  !$omp end do

end program wsloop_variable
