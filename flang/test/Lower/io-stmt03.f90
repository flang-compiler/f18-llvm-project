! RUN: bbc -emit-fir -o - %s | FileCheck %s

subroutine s ! focus on control flow -- stick to scalars
! CHECK-NOT: fir.if
! CHECK: BeginExternalFormattedInput
! CHECK-NOT: fir.if
! CHECK: SetAdvance
! CHECK-NOT: fir.if
! CHECK: InputReal32
! CHECK-NOT: fir.if
! CHECK: InputReal32
! CHECK-NOT: fir.if
! CHECK: fir.do_loop
! CHECK-NOT: fir.if
! CHECK: InputReal32
! CHECK-NOT: fir.if
! CHECK: fir.do_loop
! CHECK-NOT: fir.if
! CHECK: InputReal32
! CHECK-NOT: fir.if
! CHECK: InputReal32
! CHECK-NOT: fir.if
! CHECK: InputReal32
! CHECK-NOT: fir.if
! CHECK: InputReal32
! CHECK-NOT: fir.if
! CHECK: EndIoStatement
! CHECK-NOT: fir.if
read(*,'(F7.2)', advance='no') a, b, (c, (d, e, k=1,n), f, j=1,n), g
end
