! RUN: bbc %s -o - | FileCheck %s

! CHECK-LABEL: @_QPprint_test
! CHECK-SAME: (%[[A:.*]]: !fir.ref<!fir.complex<2>>) {
! CHECK:  %[[LIST_IO:.*]] = fir.call @_FortranAioBeginExternalListOutput
! CHECK:  %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<!fir.complex<2>>
! CHECK:  %[[A_REAL:.*]] = fir.extract_value %[[A_VAL]], [0 : index] : (!fir.complex<2>) -> f16
! CHECK:  %[[A_IMAG:.*]] = fir.extract_value %[[A_VAL]], [1 : index] : (!fir.complex<2>) -> f16
! CHECK:  %[[A_REAL_CVT:.*]] = fir.convert %[[A_REAL]] : (f16) -> f32
! CHECK:  %[[A_IMAG_CVT:.*]] = fir.convert %[[A_IMAG]] : (f16) -> f32
! CHECK:  %{{.*}} = fir.call @_FortranAioOutputComplex32(%[[LIST_IO]], %[[A_REAL_CVT]], %[[A_IMAG_CVT]]) : (!fir.ref<i8>, f32, f32) -> i1
subroutine print_test(a)
  complex(kind=2) :: a
  print *, a
end subroutine print_test
