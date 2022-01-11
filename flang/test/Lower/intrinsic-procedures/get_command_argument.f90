! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes=CHECK,CHECK-32 -DDEFAULT_INTEGER_SIZE=32 %s
! RUN: flang-new -fc1 -fdefault-integer-8 -emit-fir %s -o - | FileCheck --check-prefixes=CHECK,CHECK-64 -DDEFAULT_INTEGER_SIZE=64 %s

! CHECK-LABEL: func @_QPget_command_argument_test1(
! CHECK-32-SAME: %[[num:.*]]: !fir.ref<i32>) {
! CHECK-64-SAME: %[[num:.*]]: !fir.ref<i64>) {
subroutine get_command_argument_test1(num)
    integer :: num
    call get_command_argument(num)
! CHECK-NOT: fir.call @_FortranAArgumentValue
! CHECK-NOT: fir.call @_FortranAArgumentLength
! CHECK-NEXT: return
end subroutine get_command_argument_test1

! CHECK-LABEL: func @_QPget_command_argument_test2(
! CHECK-SAME: %[[num:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>,
! CHECK-SAME: %[[arg:.*]]: !fir.boxchar<1>) {
subroutine get_command_argument_test2(num, arg)
integer :: num
character(len=32) :: arg
call get_command_argument(num, arg)
! CHECK: %[[argUnboxed:.*]]:2 = fir.unboxchar %[[arg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[argLength:.*]] = arith.constant 32 : index
! CHECK-NEXT: %[[numUnbox:.*]] = fir.load %[[num]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-NEXT: %[[argBoxed:.*]] = fir.embox %[[argUnboxed]]#0 typeparams %[[argLength]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-NEXT: %[[errmsg:.*]] = fir.absent !fir.box<none>
! CHECK-64-NEXT: %[[numCast:.*]] = fir.convert %[[numUnbox]] : (i64) -> i32
! CHECK-NEXT: %[[argCast:.*]] = fir.convert %[[argBoxed]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-32-NEXT: %{{[0-9]+}} = fir.call @_FortranAArgumentValue(%[[numUnbox]], %[[argCast]], %[[errmsg]]) : (i32, !fir.box<none>, !fir.box<none>) -> i32
! CHECK-64-NEXT: %{{[0-9]+}} = fir.call @_FortranAArgumentValue(%[[numCast]], %[[argCast]], %[[errmsg]]) : (i32, !fir.box<none>, !fir.box<none>) -> i32
! CHECK-NOT: fir.call @_FortranAArgumentLength
end subroutine get_command_argument_test2

! CHECK-LABEL: func @_QPget_command_argument_test3(
! CHECK-SAME: %[[num:[^:]*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>,
! CHECK-SAME: %[[arg:.*]]: !fir.boxchar<1>,
! CHECK-SAME: %[[length:[^:]*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>,
! CHECK-SAME: %[[status:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>,
! CHECK-SAME: %[[errmsg:.*]]: !fir.boxchar<1>) {
subroutine get_command_argument_test3(num, arg, length, status, errmsg)
    integer :: num, length, status
    character(len=32) :: arg, errmsg
    call get_command_argument(num, arg, length, status, errmsg)
! CHECK: %[[argUnboxed:.*]]:2 = fir.unboxchar %[[arg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[argLen:.*]] = arith.constant 32 : index
! CHECK-NEXT: %[[errmsgUnboxed:.*]]:2 = fir.unboxchar %[[errmsg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[errmsgLen:.*]] = arith.constant 32 : index
! CHECK-NEXT: %[[numUnboxed:.*]] = fir.load %[[num]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-NEXT: %[[argBoxed:.*]] = fir.embox %[[argUnboxed]]#0 typeparams %[[argLen]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-NEXT: %[[errmsgBoxed:.*]] = fir.embox %[[errmsgUnboxed]]#0 typeparams %[[errmsgLen]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-64-NEXT: %[[numCast:.*]] = fir.convert %[[numUnboxed]] : (i64) -> i32
! CHECK-NEXT: %[[argBuffer:.*]] = fir.convert %[[argBoxed]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-NEXT: %[[errmsgBuffer:.*]] = fir.convert %[[errmsgBoxed]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-32-NEXT: %[[valueResult:.*]] = fir.call @_FortranAArgumentValue(%[[numUnboxed]], %[[argBuffer]], %[[errmsgBuffer]]) : (i32, !fir.box<none>, !fir.box<none>) -> i32
! CHECK-64-NEXT: %[[valueResult32:.*]] = fir.call @_FortranAArgumentValue(%[[numCast]], %[[argBuffer]], %[[errmsgBuffer]]) : (i32, !fir.box<none>, !fir.box<none>) -> i32
! CHECK-64-NEXT: %[[valueResult:.*]] = fir.convert %[[valueResult32]] : (i32) -> i64
! CHECK-NEXT: fir.store %[[valueResult]] to %[[status]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-64-NEXT: %[[numCast:.*]] = fir.convert %[[numUnboxed]] : (i64) -> i32
! CHECK-32-NEXT: %[[lengthResult64:.*]] = fir.call @_FortranAArgumentLength(%[[numUnboxed]]) : (i32) -> i64
! CHECK-64-NEXT: %[[lengthResult:.*]] = fir.call @_FortranAArgumentLength(%[[numCast]]) : (i32) -> i64
! CHECK-32-NEXT: %[[lengthResult:.*]] = fir.convert %[[lengthResult64]] : (i64) -> i32
! CHECK-NEXT: fir.store %[[lengthResult]] to %[[length]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
end subroutine get_command_argument_test3

! CHECK-LABEL: func @_QPget_command_argument_test4(
! CHECK-SAME: %[[num:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>,
! CHECK-SAME: %[[length:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>) {
subroutine get_command_argument_test4(num, length)
    integer :: num, length
    call get_command_argument(num, LENGTH=length)
! CHECK-NOT: fir.call @_FortranAArgumentValue
! CHECK: %[[numLoaded:.*]] = fir.load %[[num]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-64-NEXT: %[[numCast:.*]] = fir.convert %[[numLoaded]] : (i64) -> i32
! CHECK-32-NEXT: %[[result64:.*]] = fir.call @_FortranAArgumentLength(%[[numLoaded]]) : (i32) -> i64
! CHECK-64-NEXT: %[[result:.*]] = fir.call @_FortranAArgumentLength(%[[numCast]]) : (i32) -> i64
! CHECK-32-NEXT: %[[result:.*]] = fir.convert %[[result64]] : (i64) -> i32
! CHECK-NEXT: fir.store %[[result]] to %[[length]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-NEXT: return
end subroutine get_command_argument_test4

! CHECK-LABEL: func @_QPget_command_argument_test5(
! CHECK-SAME: %[[num:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>,
! CHECK-SAME: %[[status:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>) {
subroutine get_command_argument_test5(num, status)
    integer :: num, status
    call get_command_argument(num, STATUS=status)
! CHECK: %[[numLoaded:.*]] = fir.load %[[num]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-NEXT: %[[arg:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[errmsg:.*]] = fir.absent !fir.box<none>
! CHECK-64-NEXT: %[[numCast:.*]] = fir.convert %[[numLoaded]] : (i64) -> i32
! CHECK-32-NEXT: %[[result:.*]] = fir.call @_FortranAArgumentValue(%[[numLoaded]], %[[arg]], %[[errmsg]]) : (i32, !fir.box<none>, !fir.box<none>) -> i32
! CHECK-64-NEXT: %[[result32:.*]] = fir.call @_FortranAArgumentValue(%[[numCast]], %[[arg]], %[[errmsg]]) : (i32, !fir.box<none>, !fir.box<none>) -> i32
! CHECK-64-NEXT: %[[result:.*]] = fir.convert %[[result32]] : (i32) -> i64
! CHECK-32-NEXT: fir.store %[[result]] to %[[status]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-NOT: fir.call @_FortranAArgumentLength
end subroutine get_command_argument_test5

! CHECK-LABEL: func @_QPget_command_argument_test6(
! CHECK-SAME: %[[num:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>,
! CHECK-SAME: %[[errmsg:.*]]: !fir.boxchar<1>) {
subroutine get_command_argument_test6(num, errmsg)
    integer :: num
    character(len=32) :: errmsg
    call get_command_argument(num, ERRMSG=errmsg)
! CHECK: %[[errmsgUnboxed:.*]]:2 = fir.unboxchar %[[errmsg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[errmsgLength:.*]] = arith.constant 32 : index
! CHECK-NEXT: %[[numUnboxed:.*]] = fir.load %[[num]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-NEXT: %[[arg:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[errmsgBoxed:.*]] = fir.embox %[[errmsgUnboxed]]#0 typeparams %[[errmsgLength]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-64-NEXT: %[[numCast:.*]] = fir.convert %[[numUnboxed]] : (i64) -> i32
! CHECK-NEXT: %[[errmsg:.*]] = fir.convert %[[errmsgBoxed]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-32-NEXT: %{{[0-9]+}} = fir.call @_FortranAArgumentValue(%[[numUnboxed]], %[[arg]], %[[errmsg]]) : (i32, !fir.box<none>, !fir.box<none>) -> i32
! CHECK-64-NEXT: %{{[0-9]+}} = fir.call @_FortranAArgumentValue(%[[numCast]], %[[arg]], %[[errmsg]]) : (i32, !fir.box<none>, !fir.box<none>) -> i32
! CHECK-NOT: fir.call @_FortranAArgumentLength
end subroutine get_command_argument_test6
