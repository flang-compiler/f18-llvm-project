! RUN: bbc -emit-fir %s -o - | FileCheck %s

program nested_where
  integer :: a(3) = 0
  logical :: mask1(3) = (/ .true.,.false.,.true. /)
  logical :: mask2(3) = (/ .true.,.true.,.false. /)
  forall (i=1:3)
    where (mask1)
      where (mask2)
        a = 1
      end where
    endwhere
  end forall
end program nested_where

! CHECK: %{{.*}} = fir.if %{{.*}} -> (!fir.array<3xi32>) {
! CHECK:   %{{.*}} = fir.if %{{.*}} -> (!fir.array<3xi32>) {
! CHECK:     %{{.*}} = fir.array_update %{{.*}}, %{{.*}}, %{{.*}} : (!fir.array<3xi32>, i32, index) -> !fir.array<3xi32>
! CHECK:     fir.result %{{.*}} : !fir.array<3xi32>
! CHECK:   } else {
! CHECK:     fir.result %{{.*}} : !fir.array<3xi32>
! CHECK:   }
! CHECK:   fir.result %68 : !fir.array<3xi32>
! CHECK: } else {
! CHECK:   fir.result %arg3 : !fir.array<3xi32>
! CHECK: }
