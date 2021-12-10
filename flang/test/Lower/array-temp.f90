! RUN: bbc %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPss1()
subroutine ss1
  ! CHECK: %[[aa:[0-9]+]] = fir.alloca !fir.array<2650000xf32> {bindc_name = "aa", uniq_name = "_QFss1Eaa"}
  ! CHECK: %[[shape:[0-9]+]] = fir.shape {{.*}} : (index) -> !fir.shape<1>
  integer, parameter :: N = 2650000
  real aa(N)
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) {{.*}} : (!fir.ref<!fir.array<2650000xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  aa = -2
  ! CHECK: %[[temp:[0-9]+]] = fir.allocmem !fir.array<2650000xf32>
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) {{.*}} : (!fir.ref<!fir.array<2650000xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[temp]](%[[shape]]) {{.*}} : (!fir.heap<!fir.array<2650000xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) [{{.*}}] {{.*}} : (!fir.ref<!fir.array<2650000xf32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[temp]](%[[shape]]) [{{.*}}] {{.*}} : (!fir.heap<!fir.array<2650000xf32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[temp]](%[[shape]]) {{.*}} : (!fir.heap<!fir.array<2650000xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) {{.*}} : (!fir.ref<!fir.array<2650000xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.freemem %[[temp]] : !fir.heap<!fir.array<2650000xf32>>
  aa(2:N) = aa(1:N-1) + 7.0
! print*, aa(1:2), aa(N-1:N)
end

! CHECK-LABEL: func @_QPss2
subroutine ss2(N)
  ! CHECK: %[[arg:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
  ! CHECK: %[[n:[0-9]+]] = fir.convert %[[arg]] : (i32) -> index
  ! CHECK: %[[aa:[0-9]+]] = fir.alloca !fir.array<?xf32>, %[[n]] {bindc_name = "aa", uniq_name = "_QFss2Eaa"}
  real aa(N)
  ! CHECK: %[[shape:[0-9]+]] = fir.shape %[[n]] : (index) -> !fir.shape<1>
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) {{.*}} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  aa = -2
  ! CHECK: %[[temp:[0-9]+]] = fir.allocmem !fir.array<?xf32>, %[[n]]
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) {{.*}} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[temp]](%[[shape]]) {{.*}} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) [{{.*}}] {{.*}} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[temp]](%[[shape]]) [{{.*}}] {{.*}} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[temp]](%[[shape]]) {{.*}} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) {{.*}} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.freemem %[[temp]] : !fir.heap<!fir.array<?xf32>>
  aa(2:N) = aa(1:N-1) + 7.0
! print*, aa(1:2), aa(N-1:N)
end

! CHECK-LABEL: func @_QPss3
subroutine ss3(N)
  ! CHECK: %[[arg:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
  ! CHECK: %[[n:[0-9]+]] = fir.convert %[[arg]] : (i32) -> index
  ! CHECK: %[[aa:[0-9]+]] = fir.alloca !fir.array<2x?xf32>, %[[n]] {bindc_name = "aa", uniq_name = "_QFss3Eaa"}
  real aa(2,N)
  ! CHECK: %[[shape:[0-9]+]] = fir.shape {{.*}} %[[n]] : (index, index) -> !fir.shape<2>
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) {{.*}}, {{.*}} : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
  aa = -2
  ! CHECK: %[[temp:[0-9]+]] = fir.allocmem !fir.array<2x?xf32>, %[[n]]
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) {{.*}}, {{.*}} : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[temp]](%[[shape]]) {{.*}}, {{.*}} : (!fir.heap<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) [{{.*}}] {{.*}}, {{.*}} : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[temp]](%[[shape]]) [{{.*}}] {{.*}}, {{.*}} : (!fir.heap<!fir.array<2x?xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[temp]](%[[shape]]) {{.*}}, {{.*}} : (!fir.heap<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) {{.*}}, {{.*}} : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
  ! CHECK: fir.freemem %[[temp]] : !fir.heap<!fir.array<2x?xf32>>
  aa(:,2:N) = aa(:,1:N-1) + 7.0
! print*, aa(:,1:2), aa(:,N-1:N)
end

! CHECK-LABEL: func @_QPss4
subroutine ss4(N)
  ! CHECK: %[[arg:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
  ! CHECK: %[[n:[0-9]+]] = fir.convert %[[arg]] : (i32) -> index
  ! CHECK: %[[aa:[0-9]+]] = fir.alloca !fir.array<?x2xf32>, %[[n]] {bindc_name = "aa", uniq_name = "_QFss4Eaa"}
  real aa(N,2)
  ! CHECK: %[[shape:[0-9]+]] = fir.shape %[[n]], {{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) {{.*}}, {{.*}} : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
  aa = -2
  ! CHECK: %[[temp:[0-9]+]] = fir.allocmem !fir.array<?x2xf32>, %[[n]]
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) {{.*}}, {{.*}} : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[temp]](%[[shape]]) {{.*}}, {{.*}} : (!fir.heap<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) [{{.*}}] {{.*}}, {{.*}} : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[temp]](%[[shape]]) [{{.*}}] {{.*}}, {{.*}} : (!fir.heap<!fir.array<?x2xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[temp]](%[[shape]]) {{.*}}, {{.*}} : (!fir.heap<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) {{.*}}, {{.*}} : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
  ! CHECK: fir.freemem %[[temp]] : !fir.heap<!fir.array<?x2xf32>>
  aa(2:N,:) = aa(1:N-1,:) + 7.0
! print*, aa(1:2,:), aa(N-1:N,:)
end

! CHECK-LABEL: func @_QPtt1
subroutine tt1
  ! CHECK: fir.call @_FortranAioBeginExternalListOutput
  ! CHECK: %[[temp3:[0-9]+]] = fir.allocmem !fir.array<3xf32>
  ! CHECK: br ^bb1(%[[temp3]]
  ! CHECK-NEXT: ^bb1(%[[temp3arg:[0-9]+]]: !fir.heap<!fir.array<3xf32>>
  ! CHECK: %[[temp1:[0-9]+]] = fir.allocmem !fir.array<1xf32>
  ! CHECK: fir.call @_QFtt1Pr
  ! CHECK: fir.call @realloc
  ! CHECK: fir.freemem %[[temp1]] : !fir.heap<!fir.array<1xf32>>
  ! CHECK: %[[temp3x:[0-9]+]] = fir.allocmem !fir.array<3xf32>
  ! CHECK: fir.call @_FortranAioOutputDescriptor
  ! CHECK-NEXT: fir.freemem %[[temp3x]] : !fir.heap<!fir.array<3xf32>>
  ! CHECK-NEXT: fir.freemem %[[temp3arg]] : !fir.heap<!fir.array<3xf32>>
  ! CHECK-NEXT: fir.call @_FortranAioEndIoStatement
  print*, [(r([7.0]),i=1,3)]
contains
  ! CHECK-LABEL: func @_QFtt1Pr
  function r(x)
    real x(:)
    r = x(1)
  end
end

program p
! call ss1
! call ss2(2650000)
! call ss3(2650000)
! call ss4(2650000)
! call tt1
end
