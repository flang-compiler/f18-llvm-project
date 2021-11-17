! Test lowering of IO read SIZE control-spec (12.6.2.15)
! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: @_QPtest_read_size(
! CHECK-SAME: %[[sizeVar:[^:]+]]: !fir.ref<i32>,
subroutine test_read_size(size, c1, c2, unit, stat)
  integer :: unit, size, stat
  character(*) :: c1, c2
  ! CHECK: %[[cookie:.*]] = fir.call @_FortranAioBeginExternalFormattedInput(
  ! CHECK: fir.call @_FortranAioEnableHandlers(
  ! CHECK: %[[ok1:.*]] = fir.call @_FortranAioSetAdvance(
  ! CHECK: fir.if %[[ok1]] {
  ! CHECK:   fir.if %[[ok1]] {
  ! CHECK:     %[[ok2:.*]] = fir.call @_FortranAioInputAscii(
  ! CHECK:     fir.if %[[ok2]] {
  ! CHECK:       fir.call @_FortranAioInputAscii(
  ! CHECK:     }
  ! CHECK:   }
  ! CHECK: }
  ! CHECK: %[[sizeValue:.*]] = fir.call @_FortranAioGetSize(%[[cookie]]) : (!fir.ref<i8>) -> i64
  ! CHECK: %[[sizeCast:.*]] = fir.convert %[[sizeValue]] : (i64) -> i32
  ! CHECK: fir.store %[[sizeCast]] to %[[sizeVar]] : !fir.ref<i32>
  ! CHECK: fir.call @_FortranAioEndIoStatement(%[[cookie]]) : (!fir.ref<i8>) -> i32
  READ(unit, '(A)', ADVANCE='NO', SIZE=size, IOSTAT=stat) c1, c2
end subroutine

  integer :: unit
  character(7) :: c1
  character(4) :: c2
  integer :: size = 0
  integer :: stat = 0
  OPEN(NEWUNIT=unit,ACCESS='SEQUENTIAL',ACTION='READWRITE',&
    FORM='FORMATTED',STATUS='SCRATCH')
  WRITE(unit, '(A)') "ABCDEF"
  WRITE(unit, '(A)') "GHIJKL"
  REWIND(unit)
  call test_read_size(size, c1, c2, unit, stat)
  print *, stat, size, c1
  CLOSE(unit)
end
