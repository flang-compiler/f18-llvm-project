! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: func @_QQmain
program p
  ! CHECK-DAG: [[ccc:%[0-9]+]] = fir.address_of(@_QEccc) : !fir.ref<!fir.array<4x!fir.char<1,3>>>
  ! CHECK-DAG: [[jjj:%[0-9]+]] = fir.alloca i32 {bindc_name = "jjj", uniq_name = "_QEjjj"}
  character*3 ccc(4)
  namelist /nnn/ jjj, ccc
  jjj = 17
  ccc = ["aa ", "bb ", "cc ", "dd "]
  ! CHECK: fir.call @_FortranAioBeginExternalListOutput
  ! CHECK: [[nnn:%[0-9]+]] = fir.address_of(@_QNGnnn) : !fir.ref<tuple<!fir.ptr<i8>, i64, !fir.ptr<!fir.array<2xtuple<!fir.ptr<i8>, !fir.ptr<!fir.box<none>>>>>>>
  ! CHECK: [[nnnlist:%[0-9]+]] = fir.address_of(@_QNGnnn.list) : !fir.ref<!fir.array<2xtuple<!fir.ptr<i8>, !fir.ptr<!fir.box<none>>>>>
  ! CHECK: fir.embox [[jjj]]
  ! CHECK: fir.coordinate_of [[nnnlist]]
  ! CHECK: fir.embox [[ccc]]
  ! CHECK: fir.coordinate_of [[nnnlist]]
  ! CHECK: fir.convert [[nnn]] : (!fir.ref<tuple<!fir.ptr<i8>, i64, !fir.ptr<!fir.array<2xtuple<!fir.ptr<i8>, !fir.ptr<!fir.box<none>>>>>>>) -> !fir.ref<tuple<>>
  ! CHECK: fir.call @_FortranAioOutputNamelist
  ! CHECK: fir.call @_FortranAioEndIoStatement
  write(*, nnn)
  jjj = 27
  ccc(4) = "zz "
  ! CHECK: fir.call @_FortranAioBeginExternalListOutput
  ! CHECK: [[nnn:%[0-9]+]] = fir.address_of(@_QNGnnn) : !fir.ref<tuple<!fir.ptr<i8>, i64, !fir.ptr<!fir.array<2xtuple<!fir.ptr<i8>, !fir.ptr<!fir.box<none>>>>>>>
  ! CHECK: [[nnnlist:%[0-9]+]] = fir.address_of(@_QNGnnn.list) : !fir.ref<!fir.array<2xtuple<!fir.ptr<i8>, !fir.ptr<!fir.box<none>>>>>
  ! CHECK: fir.embox [[jjj]]
  ! CHECK: fir.coordinate_of [[nnnlist]]
  ! CHECK: fir.embox [[ccc]]
  ! CHECK: fir.coordinate_of [[nnnlist]]
  ! CHECK: fir.convert [[nnn]] : (!fir.ref<tuple<!fir.ptr<i8>, i64, !fir.ptr<!fir.array<2xtuple<!fir.ptr<i8>, !fir.ptr<!fir.box<none>>>>>>>) -> !fir.ref<tuple<>>
  ! CHECK: fir.call @_FortranAioOutputNamelist
  ! CHECK: fir.call @_FortranAioEndIoStatement
  write(*, nnn)
end
  ! CHECK: fir.global linkonce @_QQcl.6E6E6E00 constant : !fir.char<1,4>
  ! CHECK: fir.global linkonce @_QQcl.6A6A6A00 constant : !fir.char<1,4>
  ! CHECK: fir.global linkonce @_QQcl.63636300 constant : !fir.char<1,4>
  ! CHECK: fir.global linkonce @_QNGnnn.list : !fir.array<2xtuple<!fir.ptr<i8>, !fir.ptr<!fir.box<none>>>>
  ! CHECK: fir.global linkonce @_QNGnnn constant : tuple<!fir.ptr<i8>, i64, !fir.ptr<!fir.array<2xtuple<!fir.ptr<i8>, !fir.ptr<!fir.box<none>>>>>>
