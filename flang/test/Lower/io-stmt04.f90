! RUN: bbc -emit-fir -o - %s | FileCheck %s

subroutine s ! focus on control flow -- stick to scalars
! CHECK: BeginExternalFormattedInput
! CHECK: EnableHandlers
! CHECK: SetAdvance
! CHECK: fir.if
! CHECK: InputReal32
! CHECK: fir.if
! CHECK: InputReal32
! CHECK: fir.if
! CHECK: fir.iterate_while
! CHECK: fir.if
! CHECK: InputReal32
! CHECK: fir.if
! CHECK: fir.iterate_while
! CHECK: fir.if
! CHECK: InputReal32
! CHECK: fir.if
! CHECK: InputReal32
! CHECK: fir.if
! CHECK: InputReal32
! CHECK: fir.if
! CHECK: InputReal32
! CHECK: EndIoStatement
read(*,'(F7.2)', iostat=mm, advance='no') a, b, (c, (d, e, k=1,n), f, j=1,n), g
end
