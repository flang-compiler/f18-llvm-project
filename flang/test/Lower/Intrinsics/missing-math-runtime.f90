! There is no quad math runtime available in lowering
! for now. Test that the TODO are emitted correctly.
! RUN: bbc -emit-fir %s -o /dev/null 2>&1 | FileCheck %s

 real(16) :: a, b
 complex(16) :: z
! CHECK: no math runtime available for 'f128 ** f128'
 call next(a**b)
! CHECK: no math runtime available for 'acos(f128)'
 call next(acos(a))
! CHECK: no math runtime available for 'atan2(f128, f128)'
 call next(atan2(a, b))
! CHECK: no math runtime available for '!fir.complex<16> ** !fir.complex<16>'
 call nextc(a**z)
end

