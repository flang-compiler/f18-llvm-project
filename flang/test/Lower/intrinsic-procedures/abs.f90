! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPabs_testi
! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_1:.*]]: !fir.ref<i32>
subroutine abs_testi(a, b)
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:  %[[VAL_3:.*]] = arith.constant 31 : i32
! CHECK:  %[[VAL_4:.*]] = arith.shrsi %[[VAL_2]], %[[VAL_3]] : i32
! CHECK:  %[[VAL_5:.*]] = arith.xori %[[VAL_2]], %[[VAL_4]] : i32
! CHECK:  %[[VAL_6:.*]] = arith.subi %[[VAL_5]], %[[VAL_4]] : i32
! CHECK:  fir.store %[[VAL_6]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:  return
  integer :: a, b
  b = abs(a)
end subroutine

! CHECK-LABEL: func @_QPabs_testr(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<f32>{{.*}}, %[[VAL_1:.*]]: !fir.ref<f32>{{.*}}) {
subroutine abs_testr(a, b)
! CHECK: %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<f32>
! CHECK: %[[VAL_3:.*]] = fir.call @llvm.fabs.f32(%[[VAL_2]]) : (f32) -> f32
! CHECK: fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<f32>
! CHECK: return
  real :: a, b
  b = abs(a)
end subroutine

! CHECK-LABEL: func @_QPabs_testz(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.complex<4>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<f32>{{.*}}) {
subroutine abs_testz(a, b)
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.complex<4>>
! CHECK:  %[[VAL_3:.*]] = fir.extract_value %[[VAL_2]], [0 : index] : (!fir.complex<4>) -> f32
! CHECK:  %[[VAL_4:.*]] = fir.extract_value %[[VAL_2]], [1 : index] : (!fir.complex<4>) -> f32
! CHECK:  %[[VAL_5:.*]] = fir.call @__mth_i_hypot(%[[VAL_3]], %[[VAL_4]]) : (f32, f32) -> f32
! CHECK:  fir.store %[[VAL_5]] to %[[VAL_1]] : !fir.ref<f32>
! CHECK:  return
  complex :: a
  real :: b
  b = abs(a)
end subroutine abs_testz
