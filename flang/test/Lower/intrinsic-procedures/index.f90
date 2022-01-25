! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPindex_test(
! CHECK-SAME: %[[s:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[ss:[^:]+]]: !fir.boxchar<1>{{.*}}) -> i32
integer function index_test(s1, s2)
  character(*) :: s1, s2
  ! CHECK: %[[st:[^:]*]]:2 = fir.unboxchar %[[s]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[sst:[^:]*]]:2 = fir.unboxchar %[[ss]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[a1:.*]] = fir.convert %[[st]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: %[[a2:.*]] = fir.convert %[[st]]#1 : (index) -> i64
  ! CHECK: %[[a3:.*]] = fir.convert %[[sst]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: %[[a4:.*]] = fir.convert %[[sst]]#1 : (index) -> i64
  ! CHECK: = fir.call @_FortranAIndex1(%[[a1]], %[[a2]], %[[a3]], %[[a4]], %{{.*}}) : (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
  index_test = index(s1, s2)
end function index_test

! CHECK-LABEL: func @_QPindex_test2(
! CHECK-SAME: %[[s:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[ss:[^:]+]]: !fir.boxchar<1>{{.*}}) -> i32
integer function index_test2(s1, s2)
  character(*) :: s1, s2
  ! CHECK: %[[mut:.*]] = fir.alloca !fir.box<!fir.heap<i32>>
  ! CHECK: %[[st:[^:]*]]:2 = fir.unboxchar %[[s]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[sst:[^:]*]]:2 = fir.unboxchar %[[ss]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[sb:.*]] = fir.embox %[[st]]#0 typeparams %[[st]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
  ! CHECK: %[[ssb:.*]] = fir.embox %[[sst]]#0 typeparams %[[sst]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
  ! CHECK: %[[back:.*]] = fir.embox %{{.*}} : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
  ! CHECK: %[[hb:.*]] = fir.embox %{{.*}} : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  ! CHECK: %[[a0:.*]] = fir.convert %[[mut]] : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: %[[a1:.*]] = fir.convert %[[sb]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
  ! CHECK: %[[a2:.*]] = fir.convert %[[ssb]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
  ! CHECK: %[[a3:.*]] = fir.convert %[[back]] : (!fir.box<!fir.logical<4>>) -> !fir.box<none>
  ! CHECK: %[[a5:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK:  fir.call @_FortranAIndex(%[[a0]], %[[a1]], %[[a2]], %[[a3]], %{{.*}}, %[[a5]], %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
  index_test2 = index(s1, s2, .true., 4)
  ! CHECK: %[[ld1:.*]] = fir.load %[[mut]] : !fir.ref<!fir.box<!fir.heap<i32>>>
  ! CHECK: %[[ad1:.*]] = fir.box_addr %[[ld1]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  ! CHECK: %[[ld2:.*]] = fir.load %[[ad1]] : !fir.heap<i32>
  ! CHECK: fir.freemem %[[ad1]]
end function index_test2

! CHECK-LABEL: func @_QPindex_test3
integer function index_test3(s, i)
  character(*) :: s
  integer :: i
  ! CHECK: %[[tmpChar:.*]] = fir.alloca !fir.char<1>
  ! CHECK: fir.store %{{.*}} to %[[tmpChar]] : !fir.ref<!fir.char<1>>
  ! CHECK: %[[tmpCast:.*]] = fir.convert %[[tmpChar]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
  ! CHECK: fir.call @_FortranAIndex1(%{{.*}}, %{{.*}}, %[[tmpCast]], %{{.*}}, %{{.*}})
  index_test3 = index(s, char(i))
end function


