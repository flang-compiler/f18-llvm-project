! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtest1(
! CHECK-SAME: %[[a:.*]]: !fir.ref<!fir.array<100xf32>>,
! CHECK-SAME: %[[v:.*]]: !fir.box<!fir.array<?xi32>>)
subroutine test1(a,v)
  real a(100)
  integer v(:)
  ! CHECK: %[[vdim:.*]]:3 = fir.box_dims %[[v]], %{{.*}} : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[tmp:.*]] = fir.allocmem !fir.array<?xf32>, %[[vdim]]#1 {name = ".array.expr"}
  ! CHECK-DAG: %[[shape:.*]] = fir.shape %[[vdim]]#1 : (index) -> !fir.shape<1>
  ! CHECK: %[[tmparr:.*]] = fir.array_load %[[tmp]](%[[shape]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
  ! CHECK: %[[varr:.*]] = fir.array_load %[[v]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
  ! CHECK: %[[aarr:.*]] = fir.array_load %[[a]](%{{.*}}) [%{{.*}}] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<100xf32>
   associate (name => a(v))
    ! CHECK: %[[loop:.*]] = fir.do_loop %[[i:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[t:.*]] = %[[tmparr]]) -> (!fir.array<?xf32>) {
    ! CHECK: = fir.array_fetch %[[varr]], %[[i]] : (!fir.array<?xi32>, index) -> i32
    ! CHECK: %[[aval:.*]] = fir.array_fetch %[[aarr]], %{{.*}} : (!fir.array<100xf32>, index) -> f32
    ! CHECK: %[[res:.*]] = fir.array_update %[[t]], %[[aval]], %[[i]] : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
    ! CHECK: fir.result %[[res]] : !fir.array<?xf32>
    ! CHECK: fir.array_merge_store %[[tmparr]], %[[loop]] to %[[tmp]] : !fir.heap<!fir.array<?xf32>>
    ! CHECK: %[[x:.*]] = fir.convert %[[tmp]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
    ! CHECK: fir.call @_QPbob(%[[x]]) : (!fir.ref<!fir.array<?xf32>>) -> ()
    ! CHECK: fir.freemem %[[tmp]] : !fir.heap<!fir.array<?xf32>>
    call bob(name)
  end associate
end subroutine test1

! CHECK-LABEL: func @_QPtest2(
! CHECK-SAME: %[[nadd:.*]]: !fir.ref<i32>)
subroutine test2(n)
  integer :: n
  integer, external :: foo
  ! CHECK: %[[n:.*]] = fir.load %[[nadd]] : !fir.ref<i32>
  ! CHECK: %[[n10:.*]] = addi %[[n]], %c10{{.*}} : i32
  ! CHECK: fir.store %[[n10]] to %{{.*}} : !fir.ref<i32>
  ! CHECK: %[[foo:.*]] = fir.call @_QPfoo(%{{.*}}) : (!fir.ref<i32>) -> i32
  ! CHECK: fir.store %[[foo]] to %{{.*}} : !fir.ref<i32>
  associate (i => n, j => n + 10, k => foo(20))
    print *, i, j, k, n
  end associate
end subroutine test2
