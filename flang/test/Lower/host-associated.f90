! Test internal procedure host association lowering.
! RUN: bbc %s -o - | FileCheck %s

! -----------------------------------------------------------------------------
!     Test non character intrinsic scalars
! -----------------------------------------------------------------------------

!!! Test scalar (with implicit none)

! CHECK: func @_QPtest1(
subroutine test1
  implicit none
  integer i
  ! CHECK-DAG: %[[i:.*]] = fir.alloca i32 {{.*}}uniq_name = "_QFtest1Ei"
  ! CHECK-DAG: %[[tup:.*]] = fir.alloca tuple<!fir.ptr<i32>>
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[tup]], %c0
  ! CHECK: %[[ii:.*]] = fir.convert %[[i]]
  ! CHECK: fir.store %[[ii]] to %[[addr]] : !fir.ref<!fir.ptr<i32>>
  ! CHECK: fir.call @_QFtest1Ptest1_internal(%[[tup]]) : (!fir.ref<tuple<!fir.ptr<i32>>>) -> ()
  call test1_internal
  print *, i
contains
  ! CHECK: func @_QFtest1Ptest1_internal(%[[arg:[^:]*]]: !fir.ref<tuple<!fir.ptr<i32>>> {fir.host_assoc}) {
  ! CHECK: %[[iaddr:.*]] = fir.coordinate_of %[[arg]], %c0
  ! CHECK: %[[i:.*]] = fir.load %[[iaddr]] : !fir.ref<!fir.ptr<i32>>
  ! CHECK: %[[val:.*]] = fir.call @_QPifoo() : () -> i32
  ! CHECK: fir.store %[[val]] to %[[i]] : !fir.ptr<i32>
  subroutine test1_internal
    integer, external :: ifoo
    i = ifoo()
  end subroutine test1_internal
end subroutine test1

!!! Test scalar

! CHECK: func @_QPtest2() {
subroutine test2
  a = 1.0
  b = 2.0
  ! CHECK: %[[tup:.*]] = fir.alloca tuple<!fir.ptr<f32>, !fir.ptr<f32>>
  ! CHECK-DAG: %[[a0:.*]] = fir.coordinate_of %[[tup]], %c0
  ! CHECK-DAG: %[[p0:.*]] = fir.convert %{{.*}} : (!fir.ref<f32>) -> !fir.ptr<f32>
  ! CHECK: fir.store %[[p0]] to %[[a0]] : !fir.ref<!fir.ptr<f32>>
  ! CHECK-DAG: %[[b0:.*]] = fir.coordinate_of %[[tup]], %c1
  ! CHECK-DAG: %[[p1:.*]] = fir.convert %{{.*}} : (!fir.ref<f32>) -> !fir.ptr<f32>
  ! CHECK: fir.store %[[p1]] to %[[b0]] : !fir.ref<!fir.ptr<f32>>
  ! CHECK: fir.call @_QFtest2Ptest2_internal(%[[tup]]) : (!fir.ref<tuple<!fir.ptr<f32>, !fir.ptr<f32>>>) -> ()
  call test2_internal
  print *, a, b
contains
  ! CHECK: func @_QFtest2Ptest2_internal(%[[arg:[^:]*]]: !fir.ref<tuple<!fir.ptr<f32>, !fir.ptr<f32>>> {fir.host_assoc}) {
  subroutine test2_internal
    ! CHECK: %[[a:.*]] = fir.coordinate_of %[[arg]], %c0
    ! CHECK: %[[aa:.*]] = fir.load %[[a]] : !fir.ref<!fir.ptr<f32>>
    ! CHECK: %[[b:.*]] = fir.coordinate_of %[[arg]], %c1
    ! CHECK: %{{.*}} = fir.load %[[b]] : !fir.ref<!fir.ptr<f32>>
    ! CHECK: fir.alloca
    ! CHECK: fir.load %[[aa]] : !fir.ptr<f32>
    c = a
    a = b
    b = c
    call test2_inner
  end subroutine test2_internal

  ! CHECK: func @_QFtest2Ptest2_inner(%[[arg:[^:]*]]: !fir.ref<tuple<!fir.ptr<f32>, !fir.ptr<f32>>> {fir.host_assoc}) {
  subroutine test2_inner
    ! CHECK: %[[a:.*]] = fir.coordinate_of %[[arg]], %c0
    ! CHECK: %[[aa:.*]] = fir.load %[[a]] : !fir.ref<!fir.ptr<f32>>
    ! CHECK: %[[b:.*]] = fir.coordinate_of %[[arg]], %c1
    ! CHECK: %[[bb:.*]] = fir.load %[[b]] : !fir.ref<!fir.ptr<f32>>
    ! CHECK-DAG: %[[bd:.*]] = fir.load %[[bb]] : !fir.ptr<f32>
    ! CHECK-DAG: %[[ad:.*]] = fir.load %[[aa]] : !fir.ptr<f32>
    ! CHECK: %{{.*}} = arith.cmpf ogt, %[[ad]], %[[bd]] : f32
    if (a > b) then
       b = b + 2.0
    end if
  end subroutine test2_inner
end subroutine test2

! -----------------------------------------------------------------------------
!     Test non character scalars
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPtest6(
! CHECK-SAME: %[[c:.*]]: !fir.boxchar<1>
subroutine test6(c)
  character(*) :: c
  ! CHECK: %[[cunbox:.*]]:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[tup:.*]] = fir.alloca tuple<!fir.boxchar<1>>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.boxchar<1>>>, i32) -> !fir.ref<!fir.boxchar<1>>
  ! CHECK: %[[emboxchar:.*]] = fir.emboxchar %[[cunbox]]#0, %[[cunbox]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.store %[[emboxchar]] to %[[coor]] : !fir.ref<!fir.boxchar<1>>
  ! CHECK: fir.call @_QFtest6Ptest6_inner(%[[tup]]) : (!fir.ref<tuple<!fir.boxchar<1>>>) -> ()
  call test6_inner
  print *, c

contains
  ! CHECK-LABEL: func @_QFtest6Ptest6_inner(
  ! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<!fir.boxchar<1>>> {fir.host_assoc}) {
  subroutine test6_inner
    ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.boxchar<1>>>, i32) -> !fir.ref<!fir.boxchar<1>>
    ! CHECK: %[[load:.*]] = fir.load %[[coor]] : !fir.ref<!fir.boxchar<1>>
    ! CHECK: fir.unboxchar %[[load]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    c = "Hi there"
  end subroutine test6_inner
end subroutine test6

! -----------------------------------------------------------------------------
!     Test non allocatable and pointer arrays
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPtest3(
! CHECK-SAME: %[[p:[^:]+]]: !fir.box<!fir.array<?xf32>>,
! CHECK-SAME: %[[q:.*]]: !fir.box<!fir.array<?xf32>>,
! CHECK-SAME: %[[i:.*]]: !fir.ref<i64>
subroutine test3(p,q,i)
  integer(8) :: i
  real :: p(i:)
  real :: q(:)
  ! CHECK: %[[iload:.*]] = fir.load %[[i]] : !fir.ref<i64>
  ! CHECK: %[[icast:.*]] = fir.convert %[[iload]] : (i64) -> index
  ! CHECK: %[[tup:.*]] = fir.alloca tuple<!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK: %[[ptup:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.box<!fir.ptr<!fir.array<?xf32>>>>>, i32) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK: %[[pshift:.*]] = fir.shift %[[icast]] : (index) -> !fir.shift<1>
  ! CHECK: %[[pbox:.*]] = fir.rebox %[[p]](%[[pshift]]) : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[pbox]] to %[[ptup]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK: %[[qtup:.*]] = fir.coordinate_of %[[tup]], %c1{{.*}} : (!fir.ref<tuple<!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.box<!fir.ptr<!fir.array<?xf32>>>>>, i32) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK: %[[qbox:.*]] = fir.rebox %[[q]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[qbox]] to %[[qtup]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>

  i = i + 1
  q = -42.0

  ! CHECK: fir.call @_QFtest3Ptest3_inner(%[[tup]]) : (!fir.ref<tuple<!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.box<!fir.ptr<!fir.array<?xf32>>>>>) -> ()
  call test3_inner

  if (p(2) .ne. -42.0) then
     print *, "failed"
  end if
  
contains
  ! CHECK-LABEL: func @_QFtest3Ptest3_inner(
  ! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.box<!fir.ptr<!fir.array<?xf32>>>>> {fir.host_assoc}) {
  subroutine test3_inner
    ! CHECK: %[[pcoor:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.box<!fir.ptr<!fir.array<?xf32>>>>>, i32) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    ! CHECK: %[[p:.*]] = fir.load %[[pcoor]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    ! CHECK: %[[pbounds:.]]:3 = fir.box_dims %[[p]], %c0{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
    ! CHECK: %[[qcoor:.*]] = fir.coordinate_of %[[tup]], %c1{{.*}} : (!fir.ref<tuple<!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.box<!fir.ptr<!fir.array<?xf32>>>>>, i32) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    ! CHECK: %[[q:.*]] = fir.load %[[qcoor]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    ! CHECK: %[[qbounds:.]]:3 = fir.box_dims %[[q]], %c0{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)


    ! CHECK: %[[qlb:.*]] = fir.convert %[[qbounds]]#0 : (index) -> i64
    ! CHECK: %[[qoffset:.*]] = arith.subi %c1{{.*}}, %[[qlb]] : i64
    ! CHECK: %[[qelt:.*]] = fir.coordinate_of %[[q]], %[[qoffset]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, i64) -> !fir.ref<f32>
    ! CHECK: %[[qload:.*]] = fir.load %[[qelt]] : !fir.ref<f32>
    ! CHECK: %[[plb:.*]] = fir.convert %[[pbounds]]#0 : (index) -> i64
    ! CHECK: %[[poffset:.*]] = arith.subi %c2{{.*}}, %[[plb]] : i64
    ! CHECK: %[[pelt:.*]] = fir.coordinate_of %[[p]], %[[poffset]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, i64) -> !fir.ref<f32>
    ! CHECK: fir.store %[[qload]] to %[[pelt]] : !fir.ref<f32>
    p(2) = q(1)
  end subroutine test3_inner
end subroutine test3

! CHECK-LABEL: func @_QPtest3a(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.array<10xf32>>) {
subroutine test3a(p)
  real :: p(10)
  real :: q(10)
  ! CHECK: %[[q:.*]] = fir.alloca !fir.array<10xf32> {bindc_name = "q", uniq_name = "_QFtest3aEq"}
  ! CHECK: %[[tup:.*]] = fir.alloca tuple<!fir.box<!fir.ptr<!fir.array<10xf32>>>, !fir.box<!fir.ptr<!fir.array<10xf32>>>>
  ! CHECK: %[[ptup:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.box<!fir.ptr<!fir.array<10xf32>>>, !fir.box<!fir.ptr<!fir.array<10xf32>>>>>, i32) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<10xf32>>>>
  ! CHECK: %[[shape:.*]] = fir.shape %c10{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[pbox:.*]] = fir.embox %[[p]](%[[shape]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<10xf32>>>
  ! CHECK: fir.store %[[pbox]] to %[[ptup]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<10xf32>>>>
  ! CHECK: %[[qtup:.*]] = fir.coordinate_of %[[tup]], %c1{{.*}} : (!fir.ref<tuple<!fir.box<!fir.ptr<!fir.array<10xf32>>>, !fir.box<!fir.ptr<!fir.array<10xf32>>>>>, i32) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<10xf32>>>>
  ! CHECK: %[[qbox:.*]] = fir.embox %[[q]](%[[shape]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<10xf32>>>
  ! CHECK: fir.store %[[qbox]] to %[[qtup]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<10xf32>>>>

  q = -42.0
  ! CHECK: fir.call @_QFtest3aPtest3a_inner(%[[tup]]) : (!fir.ref<tuple<!fir.box<!fir.ptr<!fir.array<10xf32>>>, !fir.box<!fir.ptr<!fir.array<10xf32>>>>>) -> ()
  call test3a_inner

  if (p(1) .ne. -42.0) then
     print *, "failed"
  end if
  
contains
  ! CHECK: func @_QFtest3aPtest3a_inner(
  ! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<!fir.box<!fir.ptr<!fir.array<10xf32>>>, !fir.box<!fir.ptr<!fir.array<10xf32>>>>> {fir.host_assoc}) {
  subroutine test3a_inner
    ! CHECK: %[[pcoor:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.box<!fir.ptr<!fir.array<10xf32>>>, !fir.box<!fir.ptr<!fir.array<10xf32>>>>>, i32) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<10xf32>>>>
    ! CHECK: %[[p:.*]] = fir.load %[[pcoor]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<10xf32>>>>
    ! CHECK: %[[paddr:.*]] = fir.box_addr %[[p]] : (!fir.box<!fir.ptr<!fir.array<10xf32>>>) -> !fir.ptr<!fir.array<10xf32>>
    ! CHECK: %[[qcoor:.*]] = fir.coordinate_of %[[tup]], %c1{{.*}} : (!fir.ref<tuple<!fir.box<!fir.ptr<!fir.array<10xf32>>>, !fir.box<!fir.ptr<!fir.array<10xf32>>>>>, i32) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<10xf32>>>>
    ! CHECK: %[[q:.*]] = fir.load %[[qcoor]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<10xf32>>>>
    ! CHECK: %[[qaddr:.*]] = fir.box_addr %[[q]] : (!fir.box<!fir.ptr<!fir.array<10xf32>>>) -> !fir.ptr<!fir.array<10xf32>>

    ! CHECK: %[[qelt:.*]] = fir.coordinate_of %[[qaddr]], %c0{{.*}} : (!fir.ptr<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
    ! CHECK: %[[qload:.*]] = fir.load %[[qelt]] : !fir.ref<f32>
    ! CHECK: %[[pelt:.*]] = fir.coordinate_of %[[paddr]], %c0{{.*}} : (!fir.ptr<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
    ! CHECK: fir.store %[[qload]] to %[[pelt]] : !fir.ref<f32>
    p(1) = q(1)
  end subroutine test3a_inner
end subroutine test3a

! -----------------------------------------------------------------------------
!     Test allocatable and pointer scalars
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPtest4() {
subroutine test4
  real, pointer :: p
  real, allocatable, target :: ally
  ! CHECK: %[[ally:.*]] = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = "ally", fir.target, uniq_name = "_QFtest4Eally"}
  ! CHECK: %[[p:.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "p", uniq_name = "_QFtest4Ep"}
  ! CHECK: %[[tup:.*]] = fir.alloca tuple<!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>>
  ! CHECK: %[[ptup:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<f32>>>>
  ! CHECK: fir.store %[[p]] to %[[ptup]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<f32>>>>
  ! CHECK: %[[atup:.*]] = fir.coordinate_of %[[tup]], %c1{{.*}} : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<f32>>>>
  ! CHECK: fir.store %[[ally]] to %[[atup]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<f32>>>>
  ! CHECK: fir.call @_QFtest4Ptest4_inner(%[[tup]]) : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>>>) -> ()

  allocate(ally)
  ally = -42.0
  call test4_inner

  if (p .ne. -42.0) then
     print *, "failed"
  end if
  
contains
  ! CHECK-LABEL: func @_QFtest4Ptest4_inner(
  ! CHECK-SAME:%[[tup:.*]]: !fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>>> {fir.host_assoc}) {
  subroutine test4_inner
    ! CHECK: %[[ptup:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<f32>>>>
    ! CHECK: %[[p:.*]] = fir.load %[[ptup]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<f32>>>>
    ! CHECK: %[[atup:.*]] = fir.coordinate_of %[[tup]], %c1{{.*}} : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<f32>>>>
    ! CHECK: %[[a:.*]] = fir.load %[[atup]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<f32>>>>
    ! CHECK: %[[abox:.*]] = fir.load %[[a]] : !fir.ref<!fir.box<!fir.heap<f32>>>
    ! CHECK: %[[addr:.*]] = fir.box_addr %[[abox]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
    ! CHECK: %[[ptr:.*]] = fir.embox %[[addr]] : (!fir.heap<f32>) -> !fir.box<!fir.ptr<f32>>
    ! CHECK: fir.store %[[ptr]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
    p => ally
  end subroutine test4_inner
end subroutine test4

! -----------------------------------------------------------------------------
!     Test allocatable and pointer arrays
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPtest5() {
subroutine test5
  real, pointer :: p(:)
  real, allocatable, target :: ally(:)

  ! CHECK: %[[ally:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "ally", fir.target
  ! CHECK: %[[p:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>> {bindc_name = "p"
  ! CHECK: %[[tup:.*]] = fir.alloca tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  ! CHECK: %[[ptup:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>>
  ! CHECK: fir.store %[[p]] to %[[ptup]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>>
  ! CHECK: %[[atup:.*]] = fir.coordinate_of %[[tup]], %c1{{.*}} : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  ! CHECK: fir.store %[[ally]] to %[[atup]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  ! CHECK: fir.call @_QFtest5Ptest5_inner(%[[tup]]) : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>) -> ()

  allocate(ally(10))
  ally = -42.0
  call test5_inner

  if (p(1) .ne. -42.0) then
     print *, "failed"
  end if
  
contains
  ! CHECK-LABEL: func @_QFtest5Ptest5_inner(
  ! CHECK-SAME:%[[tup:.*]]: !fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>> {fir.host_assoc}) {
  subroutine test5_inner
    ! CHECK: %[[ptup:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>>
    ! CHECK: %[[p:.*]] = fir.load %[[ptup]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>>
    ! CHECK: %[[atup:.*]] = fir.coordinate_of %[[tup]], %c1{{.*}} : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
    ! CHECK: %[[a:.*]] = fir.load %[[atup]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
    ! CHECK: %[[abox:.*]] = fir.load %[[a]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    ! CHECK-DAG: %[[adims:.*]]:3 = fir.box_dims %[[abox]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
    ! CHECK-DAG: %[[addr:.*]] = fir.box_addr %[[abox]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
    ! CHECK-DAG: %[[ashape:.*]] = fir.shape_shift %[[adims]]#0, %[[adims]]#1 : (index, index) -> !fir.shapeshift<1>

    ! CHECK: %[[ptr:.*]] = fir.embox %[[addr]](%[[ashape]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
    ! CHECK: fir.store %[[ptr]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    p => ally
  end subroutine test5_inner
end subroutine test5


! -----------------------------------------------------------------------------
!     Test elemental internal procedure
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPtest7(
! CHECK-SAME: %[[j:.*]]: !fir.ref<i32>,
! CHECK-SAME: %[[k:.*]]: !fir.box<!fir.array<?xi32>>
subroutine test7(j, k)
  implicit none
  integer :: j
  integer :: k(:)
  ! CHECK: %[[tup:.*]] = fir.alloca tuple<!fir.ptr<i32>>
  ! CHECK: %[[jtup:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.ptr<i32>>>, i32) -> !fir.ref<!fir.ptr<i32>>
  ! CHECK: %[[jptr:.*]] = fir.convert %[[j]] : (!fir.ref<i32>) -> !fir.ptr<i32>
  ! CHECK: fir.store %[[jptr]] to %[[jtup]] : !fir.ref<!fir.ptr<i32>>

  ! CHECK: %[[kelem:.*]] = fir.array_coor %[[k]] %{{.*}} : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
  ! CHECK: fir.call @_QFtest7Ptest7_inner(%[[kelem]], %[[tup]]) : (!fir.ref<i32>, !fir.ref<tuple<!fir.ptr<i32>>>) -> i32
  k = test7_inner(k)
contains

! CHECK-LABEL: func @_QFtest7Ptest7_inner(
! CHECK-SAME: %[[i:.*]]: !fir.ref<i32>,
! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<!fir.ptr<i32>>> {fir.host_assoc}) -> i32 {
elemental integer function test7_inner(i)
  implicit none
  integer, intent(in) :: i
  ! CHECK: %[[jtup:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.ptr<i32>>>, i32) -> !fir.ref<!fir.ptr<i32>>
  ! CHECK: %[[jptr:.*]] = fir.load %[[jtup]] : !fir.ref<!fir.ptr<i32>>
  ! CHECK-DAG: %[[iload:.*]] = fir.load %[[i]] : !fir.ref<i32>
  ! CHECK-DAG: %[[jload:.*]] = fir.load %[[jptr]] : !fir.ptr<i32>
  ! CHECK: addi %[[iload]], %[[jload]] : i32
  test7_inner = i + j
end function
end subroutine

subroutine issue990()
  ! Test that host symbols used in statement functions inside an internal
  ! procedure are correctly captured from the host.
  implicit none
  integer :: captured
  call bar()
contains
! CHECK-LABEL: func @_QFissue990Pbar(
! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<!fir.ptr<i32>>> {fir.host_assoc}) {
subroutine bar()
  integer :: stmt_func, i
  stmt_func(i) = i + captured
  ! CHECK: %[[tupAddr:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.ptr<i32>>>, i32) -> !fir.ref<!fir.ptr<i32>>
  ! CHECK: %[[addr:.*]] = fir.load %[[tupAddr]] : !fir.ref<!fir.ptr<i32>>
  ! CHECK: %[[value:.*]] = fir.load %[[addr]] : !fir.ptr<i32>
  ! CHECK: arith.addi %{{.*}}, %[[value]] : i32
  print *, stmt_func(10)
end subroutine
end subroutine

subroutine issue990b()
  ! Test when an internal procedure uses a statement function from its host
  ! which uses host variables that are otherwise not used by the internal
  ! procedure.
  implicit none
  integer :: captured, captured_stmt_func, i
  captured_stmt_func(i) = i + captured
  call bar()
contains
! CHECK-LABEL: func @_QFissue990bPbar(
! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<!fir.ptr<i32>>> {fir.host_assoc}) {
subroutine bar()
  ! CHECK: %[[tupAddr:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.ptr<i32>>>, i32) -> !fir.ref<!fir.ptr<i32>>
  ! CHECK: %[[addr:.*]] = fir.load %[[tupAddr]] : !fir.ref<!fir.ptr<i32>>
  ! CHECK: %[[value:.*]] = fir.load %[[addr]] : !fir.ptr<i32>
  ! CHECK: arith.addi %{{.*}}, %[[value]] : i32
  print *, captured_stmt_func(10)
end subroutine
end subroutine

! Test capture of dummy procedure functions.
subroutine test8(dummy_proc)
 implicit none
 interface
   real function dummy_proc(x)
    real :: x
   end function
 end interface
 call bar()
contains
! CHECK-LABEL: func @_QFtest8Pbar(
! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<() -> ()>> {fir.host_assoc}) {
subroutine bar()
  ! CHECK: %[[tupAddr:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<() -> ()>>, i32) -> !fir.ref<() -> ()>
  ! CHECK: %[[dummyProc:.*]] = fir.load %[[tupAddr]] : !fir.ref<() -> ()>
  ! CHECK: %[[dummyProcCast:.*]] = fir.convert %[[dummyProc]] : (() -> ()) -> ((!fir.ref<f32>) -> f32)
  ! CHECK: fir.call %[[dummyProcCast]](%{{.*}}) : (!fir.ref<f32>) -> f32
 print *, dummy_proc(42.)
end subroutine
end subroutine

! Test capture of dummy subroutines.
subroutine test9(dummy_proc)
 implicit none
 interface
   subroutine dummy_proc()
   end subroutine
 end interface
 call bar()
contains
! CHECK-LABEL: func @_QFtest9Pbar(
! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<() -> ()>> {fir.host_assoc}) {
subroutine bar()
  ! CHECK: %[[tupAddr:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<() -> ()>>, i32) -> !fir.ref<() -> ()>
  ! CHECK: %[[dummyProc:.*]] = fir.load %[[tupAddr]] : !fir.ref<() -> ()>
  ! CHECK: fir.call %[[dummyProc]]() : () -> ()
  call dummy_proc()
end subroutine
end subroutine

! Test capture of namelist
! CHECK-LABEL: func @_QPtest10(
! CHECK-SAME: %[[i:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) {
subroutine test10(i)
 implicit none
 integer, pointer :: i(:)
 namelist /a_namelist/ i
 ! CHECK: %[[tupAddr:.*]] = fir.coordinate_of %[[tup:.*]], %c0{{.*}} : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>
 ! CHECK: fir.store %[[i]] to %[[tupAddr]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>
 ! CHECK: fir.call @_QFtest10Pbar(%[[tup]]) : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>>) -> ()
 call bar()
contains
! CHECK-LABEL: func @_QFtest10Pbar(
! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>> {fir.host_assoc}) {
subroutine bar()
  ! CHECK: %[[tupAddr:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>
  ! CHECK: fir.load %[[tupAddr]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>
  read (88, NML = a_namelist) 
end subroutine
end subroutine
