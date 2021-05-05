! RUN: bbc %s -o - | FileCheck %s

!  Constant array ctor.
! CHECK-LABEL: func @_QPtest1(
subroutine test1(a, b)
  real :: a(3)
  integer :: b(4)
  integer, parameter :: constant_array(4) = [6, 7, 42, 9]

  ! Array ctors for constant arrays should be outlined as constant globals.

  !  Look at inline constructor case
  ! CHECK: %{{.*}} = fir.address_of(@_QQro.3xr4.6e55f044605a4991f15fd4505d83faf4) : !fir.ref<!fir.array<3xf32>>
  a = (/ 1.0, 2.0, 3.0 /)

  !  Look at PARAMETER case
  ! CHECK: %{{.*}} = fir.address_of(@_QQro.4xi4.6a6af0eea868c84da59807d34f7e1a86) : !fir.ref<!fir.array<4xi32>>
  b = constant_array
end subroutine test1

!  Dynamic array ctor with constant extent.
! CHECK-LABEL: func @_QPtest2(
! CHECK-SAME: %[[a:[^:]*]]: !fir.ref<!fir.array<5xf32>>,
! CHECK-SAME: %[[b:[^:]*]]: !fir.ref<f32>)
subroutine test2(a, b)
  real :: a(5), b
  real, external :: f

  !  Look for the 5 store patterns
  ! CHECK: %[[tmp:.*]] = fir.allocmem !fir.array<5xf32>
  ! CHECK: %[[val:.*]] = fir.call @_QPf(%[[b]]) : (!fir.ref<f32>) -> f32
  ! CHECK: %[[loc:.*]] = fir.coordinate_of %[[tmp]], %c{{.*}} : (!fir.heap<!fir.array<5xf32>>, index) -> !fir.ref<f32>
  ! CHECK: fir.store %[[val]] to %[[loc]] : !fir.ref<f32>
  ! CHECK: fir.call @_QPf(%{{.*}}) : (!fir.ref<f32>) -> f32
  ! CHECK: fir.coordinate_of %[[tmp]], %c{{.*}} : (!fir.heap<!fir.array<5xf32>>, index) -> !fir.ref<f32>
  ! CHECK: fir.store
  ! CHECK: fir.call @_QPf(
  ! CHECK: fir.coordinate_of %[[tmp]], %c
  ! CHECK: fir.store
  ! CHECK: fir.call @_QPf(
  ! CHECK: fir.coordinate_of %[[tmp]], %c
  ! CHECK: fir.store
  ! CHECK: fir.call @_QPf(
  ! CHECK: fir.coordinate_of %[[tmp]], %c
  ! CHECK: fir.store

  !  After the ctor done, loop to copy result to `a`
  ! CHECK-DAG: fir.array_coor %[[tmp]](%
  ! CHECK-DAG: %[[ai:.*]] = fir.array_coor %[[a]](%
  ! CHECK: fir.store %{{.*}} to %[[ai]] : !fir.ref<f32>
  ! CHECK: fir.freemem %[[tmp]] : !fir.heap<!fir.array<5xf32>>

  a = [f(b), f(b+1), f(b+2), f(b+5), f(b+11)]
end subroutine test2

!  Dynamic array ctor with dynamic extent.
! CHECK-LABEL: func @_QPtest3(
! CHECK-SAME: %[[a:.*]]: !fir.box<!fir.array<?xf32>>)
subroutine test3(a)
  real :: a(:)
  real, allocatable :: b(:), c(:)
  interface
    subroutine test3b(x)
      real, allocatable :: x(:)
    end subroutine test3b
  end interface
  interface
    function test3c
      real, allocatable :: test3c(:)
    end function test3c
  end interface

  ! CHECK: fir.call @_QPtest3b
  call test3b(b)
  !a = (/ b, test3c() /)
end subroutine test3

! CHECK-LABEL: func @_QPtest4(
subroutine test4(a, b, n1, m1)
  real :: a(:)
  real :: b(:,:)
  integer, external :: f1, f2, f3

  ! Dynamic array ctor with dynamic extent using implied do loops.
  !a = [ ((b(i,j), j=f1(i),f2(n1),f3(m1+i)), i=1,n1,m1) ]
end subroutine test4

! CHECK: fir.global internal @_QQro.3xr4.6e55f044605a4991f15fd4505d83faf4 constant : !fir.array<3xf32>
! CHECK: constant 1.0
! CHECK: constant 2.0
! CHECK: constant 3.0

! CHECK: fir.global internal @_QQro.4xi4.6a6af0eea868c84da59807d34f7e1a86 constant : !fir.array<4xi32>
! CHECK: constant 6
! CHECK: constant 7
! CHECK: constant 42
! CHECK: constant 9
