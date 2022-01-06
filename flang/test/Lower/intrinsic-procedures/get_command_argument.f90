! RUN: flang-new -fc1 -emit-fir %s -o - | FileCheck %s
! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPget_command_argument_test1(
! CHECK-SAME: %[[num:.*]]: !fir.ref<i32>) {
subroutine get_command_argument_test1(num)
    integer :: num
    call get_command_argument(num)
! CHECK: %[[numUnbox:.*]] = fir.load %[[num]] : !fir.ref<i32>
! CHECK: %[[arg:.*]] = fir.absent !fir.box<none>
! CHECK: %[[errmsg:.*]] = fir.absent !fir.box<none>
! CHECK: %{{[0-9]+}} = fir.call @_FortranAArgumentValue(%[[numUnbox]], %[[arg]], %[[errmsg]]) : (i32, !fir.box<none>, !fir.box<none>) -> i32
end subroutine get_command_argument_test1

! CHECK-LABEL: func @_QPget_command_argument_test2(
! CHECK-SAME: %[[num:.*]]: !fir.ref<i32>,
! CHECK-SAME: %[[arg:.*]]: !fir.boxchar<1>) {
subroutine get_command_argument_test2(num, arg)
integer :: num
character(len=32) :: arg
call get_command_argument(num, arg)
! CHECK: %0:2 = fir.unboxchar %[[arg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %c32 = arith.constant 32 : index
! CHECK: %[[numUnbox:.*]] = fir.load %[[num]] : !fir.ref<i32>
! CHECK: %2 = fir.embox %0#0 typeparams %c32 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK: %[[errmsg:.*]] = fir.absent !fir.box<none>
! CHECK: %[[argUnbox:.*]] = fir.convert %2 : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK: %5 = fir.call @_FortranAArgumentValue(%[[numUnbox]], %[[argUnbox]], %[[errmsg]]) : (i32, !fir.box<none>, !fir.box<none>) -> i32
end subroutine get_command_argument_test2

! CHECK-LABEL: func @_QPget_command_argument_test3(
! CHECK-SAME: %[[num:[^:]*]]: !fir.ref<i32>,
! CHECK-SAME: %[[arg:.*]]: !fir.boxchar<1>,
! CHECK-SAME: %[[length:[^:]*]]: !fir.ref<i32>,
! CHECK-SAME: %[[status:.*]]: !fir.ref<i32>,
! CHECK-SAME: %[[errmsg:.*]]: !fir.boxchar<1>) {
subroutine get_command_argument_test3(num, arg, length, status, errmsg)
    integer :: num, length, status
    character(len=32) :: arg, errmsg
    call get_command_argument(num, arg, length, status, errmsg)
! CHECK: %[[argUnboxed:.*]]:2 = fir.unboxchar %[[arg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[argLen:.*]] = arith.constant 32 : index
! CHECK: %[[errmsgUnboxed:.*]]:2 = fir.unboxchar %[[errmsg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[errmsgLen:.*]] = arith.constant 32 : index
! CHECK: %[[numUnboxed:.*]] = fir.load %[[num]] : !fir.ref<i32>
! CHECK: %[[argBoxed:.*]] = fir.embox %[[argUnboxed]]#0 typeparams %[[argLen]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK: %[[errmsgBoxed:.*]] = fir.embox %[[errmsgUnboxed]]#0 typeparams %[[errmsgLen]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK: %[[argBuffer:.*]] = fir.convert %[[argBoxed]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK: %[[errmsgBuffer:.*]] = fir.convert %[[errmsgBoxed]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK: %[[valueResult:.*]] = fir.call @_FortranAArgumentValue(%[[numUnboxed]], %[[argBuffer]], %[[errmsgBuffer]]) : (i32, !fir.box<none>, !fir.box<none>) -> i32
! CHECK: fir.store %[[valueResult]] to %[[status]] : !fir.ref<i32>
! CHECK: %[[lengthResult:.*]] = fir.call @_FortranAArgumentLength(%[[numUnboxed]]) : (i32) -> i64
! CHECK: %[[lengthCast:.*]] = fir.convert %[[lengthResult]] : (i64) -> i32
! CHECK: fir.store %[[lengthCast]] to %[[length]] : !fir.ref<i32>
end subroutine get_command_argument_test3








