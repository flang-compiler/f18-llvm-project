! RUN: bbc %s -o - | FileCheck %s

  ! CHECK: %[[aa:[0-9]+]] = fir.address_of(@_QEaa) : !fir.ref<!fir.array<2650000xf32>>{{$}}
  integer, parameter :: N = 2650000
  real aa(N)
  ! CHECK: %7 = fir.array_coor %[[aa]](%1) %6 : (!fir.ref<!fir.array<2650000xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  aa = -2
  ! CHECK: %[[temp:[0-9]+]] = fir.allocmem !fir.array<2650000xf32>
  ! CHECK: %15 = fir.array_coor %[[aa]](%1) %14 : (!fir.ref<!fir.array<2650000xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: %16 = fir.array_coor %[[temp]](%1) %14 : (!fir.heap<!fir.array<2650000xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: %24 = fir.array_coor %[[aa]](%1) [%19] %23 : (!fir.ref<!fir.array<2650000xf32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f32>
  ! CHECK: %27 = fir.array_coor %[[temp]](%1) [%9] %23 : (!fir.heap<!fir.array<2650000xf32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f32>
  ! CHECK: %33 = fir.array_coor %[[temp]](%1) %32 : (!fir.heap<!fir.array<2650000xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: %34 = fir.array_coor %[[aa]](%1) %32 : (!fir.ref<!fir.array<2650000xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.freemem %[[temp]] : !fir.heap<!fir.array<2650000xf32>>
  aa(2:N) = aa(1:N-1) + 7.0
  ! CHECK: %40 = fir.coordinate_of %[[aa]], %c0_i64 : (!fir.ref<!fir.array<2650000xf32>>, i64) -> !fir.ref<f32>
  ! CHECK: %43 = fir.coordinate_of %[[aa]], %c2649999_i64 : (!fir.ref<!fir.array<2650000xf32>>, i64) -> !fir.ref<f32>
  print*, aa(1), aa(N)
end
