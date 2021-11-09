! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: associated_test
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.ptr<f32>>>,
! CHECK-SAME: %[[arg1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
subroutine associated_test(scalar, array)
  real, pointer :: scalar, array(:)
  real, target :: ziel
  ! CHECK: %[[ziel:.*]] = fir.alloca f32 {bindc_name = "ziel"
  ! CHECK: %[[scalar:.*]] = fir.load %[[arg0]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
  ! CHECK: %[[addr0:.*]] = fir.box_addr %[[scalar]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  ! CHECK: %[[addrToInt0:.*]] = fir.convert %[[addr0]]
  ! CHECK: cmpi ne, %[[addrToInt0]], %c0{{.*}}
  print *, associated(scalar)
  ! CHECK: %[[array:.*]] = fir.load %[[arg1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK: %[[addr1:.*]] = fir.box_addr %[[array]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
  ! CHECK: %[[addrToInt1:.*]] = fir.convert %[[addr1]]
  ! CHECK: cmpi ne, %[[addrToInt1]], %c0{{.*}}
  print *, associated(array)
  ! CHECK: %[[zbox0:.*]] = fir.embox %[[ziel]] : (!fir.ref<f32>) -> !fir.box<f32>
  ! CHECK: %[[scalar:.*]] = fir.load %[[arg0]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
  ! CHECK: %[[sbox:.*]] = fir.convert %[[scalar]] : (!fir.box<!fir.ptr<f32>>) -> !fir.box<none>
  ! CHECK: %[[zbox:.*]] = fir.convert %[[zbox0]] : (!fir.box<f32>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAPointerIsAssociatedWith(%[[sbox]], %[[zbox]]) : (!fir.box<none>, !fir.box<none>) -> i1
  print *, associated(scalar, ziel)
end subroutine

subroutine test_func_results()
  interface
    function get_pointer()
      real, pointer :: get_pointer(:) 
    end function
  end interface
  ! CHECK: %[[result:.*]] = fir.call @_QPget_pointer() : () -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.save_result %[[result]] to %[[box_storage:.*]] : !fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK: %[[box:.*]] = fir.load %[[box_storage]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
  ! CHECK: %[[addr_cast:.*]] = fir.convert %[[addr]] : (!fir.ptr<!fir.array<?xf32>>) -> i64
  ! CHECK:  arith.cmpi ne, %[[addr_cast]], %c0{{.*}} : i64
  print *, associated(get_pointer())
end subroutine
