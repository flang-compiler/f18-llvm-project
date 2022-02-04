! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPaimag_test(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.complex<4>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<f32>{{.*}}) {
subroutine aimag_test(a, b)
! CHECK: %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.complex<4>>
! CHECK: %[[VAL_3:.*]] = fir.extract_value %[[VAL_2]], [1 : index] : (!fir.complex<4>) -> f32
! CHECK: fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<f32>
! CHECK: return
  complex :: a
  real :: b
  b = aimag(a)
end subroutine
