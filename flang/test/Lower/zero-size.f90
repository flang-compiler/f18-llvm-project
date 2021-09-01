! RUN: bbc -o - %s | FileCheck %s

! CHECK-LABEL: _QPzero
subroutine zero(aa)
  real, dimension(:) :: aa
  print*, size(aa)
end

! CHECK-LABEL: _QQmain
program prog
  real nada(2:-1)
  interface
    subroutine zero(aa)
      real, dimension(:) :: aa
    end
  end interface
  ! CHECK: %[[shape:[0-9]*]] = fir.shape_shift %c2, %c0 : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %2 = fir.embox %0(%[[shape]]) : (!fir.ref<!fir.array<0xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<0xf32>>
  call zero(nada)
end
