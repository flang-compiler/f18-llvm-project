! RUN: bbc %s -o - | FileCheck %s

! ABS
! CHECK-LABEL: abs_testi
subroutine abs_testi(a, b)
  integer :: a, b
  ! CHECK: shift_right_signed
  ! CHECK: xor
  ! CHECK: subi
  b = abs(a)
end subroutine

! CHECK-LABEL: abs_testr
subroutine abs_testr(a, b)
  real :: a, b
  ! CHECK: call @llvm.fabs.f32
  b = abs(a)
end subroutine

! CHECK-LABEL: abs_testz
subroutine abs_testz(a, b)
  complex :: a
  real :: b
  ! CHECK: fir.extract_value
  ! CHECK: fir.extract_value
  ! CHECK: call @{{.*}}hypot
  b = abs(a)
end subroutine

! AIMAG
! CHECK-LABEL: aimag_test
subroutine aimag_test(a, b)
  complex :: a
  real :: b
  ! CHECK: fir.extract_value
  b = aimag(a)
end subroutine

! DBLE
! CHECK-LABEL: dble_test
subroutine dble_test(a)
  real :: a
  ! CHECK: fir.convert {{.*}} : (f32) -> f64
  print *, dble(a)
end subroutine

! CONJG
! CHECK-LABEL: conjg_test
subroutine conjg_test(z1, z2)
  complex :: z1, z2
  ! CHECK: fir.extract_value 
  ! CHECK: fir.negf
  ! CHECK: fir.insert_value 
  z2 = conjg(z1)
end subroutine

! ICHAR
! CHECK-LABEL: ichar_test
subroutine ichar_test(c)
  character(1) :: c
  ! CHECK: fir.convert {{.*}} : (!fir.char<1>) -> i32
  print *, ichar(c)
end subroutine

! LEN
! CHECK-LABEL: len_test
subroutine len_test(i, c)
  integer :: i
  character(*) :: c
  ! CHECK: fir.boxchar_len
  i = len(c)
end subroutine


! SIGN
! CHECK-LABEL: sign_testi
subroutine sign_testi(a, b, c)
  integer a, b, c
  ! CHECK: shift_right_signed
  ! CHECK: xor
  ! CHECK: subi
  ! CHECK-DAG: subi
  ! CHECK-DAG: cmpi "slt"
  ! CHECK: select
  c = sign(a, b)
end subroutine

! CHECK-LABEL: sign_testr
subroutine sign_testr(a, b, c)
  real a, b, c
  ! CHECK-DAG: call {{.*}}fabs
  ! CHECK-DAG: fir.negf
  ! CHECK-DAG: fir.cmpf "olt"
  ! CHECK: select
  c = sign(a, b)
end subroutine

! SQRT
! CHECK-LABEL: sqrt_testr
subroutine sqrt_testr(a, b)
  real :: a, b
  ! CHECK: call {{.*}}sqrt
  b = sqrt(a)
end subroutine
