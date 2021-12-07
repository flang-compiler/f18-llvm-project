! Test that module variables with an initializer are only defined once,
! even in context where this is compiler generated symbols/namelists (*).

! RUN: split-file %s %t
! RUN: bbc -emit-fir %t/definition-a.f90 -o - | FileCheck %s --check-prefix=CHECK-A-DEF
! RUN: bbc -emit-fir %t/definition-b.f90 -o - | FileCheck %s --check-prefix=CHECK-B-DEF
! RUN: bbc -emit-fir %t/use.f90 -o - | FileCheck %s --check-prefix=CHECK-USE


! (*) compiler generated symbols, namelist members are special because
! the symbol on the use site are not symbols with semantics::UseDetails,
! but directly the symbols from the module scope.

!--- definition-a.f90

! Test definition of `atype` derived type descriptor.
module define_a
  type atype
    real :: x
  end type
end module

! CHECK-A-DEF: fir.global @_QMdefine_aE.dt.atype : !fir.type<{{.*}}> {
! CHECK-A-DEF: fir.has_value
! CHECK-A-DEF: }

!--- definition-b.f90

! Test define_b `i` is defined here as well as `btype` derived type
! descriptor, but ensure `atype` derived type descriptor is not redefined
! (only declared) while defining `btype` descriptors that depends on it.
module define_b
  use :: define_a
  type btype
    type(atype) :: atype
  end type
  integer :: i = 42
  namelist /some_namelist/ i
end module

! CHECK-B-DEF: fir.global @_QMdefine_bEi : i32 {
! CHECK-B-DEF: fir.has_value %{{.*}} : i32
! CHECK-B-DEF: }

! CHECK-B-DEF: fir.global @_QMdefine_bE.dt.btype : !fir.type<{{.*}}> {
! CHECK-B-DEF: fir.has_value
! CHECK-B-DEF: }

! CHECK-B-DEF: fir.global @_QMdefine_aE.dt.atype : !fir.type<{{.*}}>{{$}}


!--- use.f90

! Test  define_b `i` is declared but not defined here and that derived types
! descriptors are not redefined here.
subroutine foo()
  use :: define_b
  type(btype) :: somet
  print *, somet
  write(*, some_namelist)
end subroutine
! CHECK-USE-NOT: fir.global @_QMdefine_bE.dt.btype

! CHECK-USE: fir.global @_QMdefine_bEi : i32{{$}}
! CHECK-USE-NOT: fir.has_value %{{.*}} : i32

! CHECK-USE-NOT: fir.global @_QMdefine_bE.dt.btype
