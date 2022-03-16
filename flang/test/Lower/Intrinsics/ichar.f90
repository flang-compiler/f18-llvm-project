! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: ichar_test
subroutine ichar_test(c)
  character(1) :: c
  character :: str(10)
  ! CHECK-DAG: %[[unbox:.*]]:2 = fir.unboxchar
  ! CHECK-DAG: %[[J:.*]] = fir.alloca i32 {{{.*}}uniq_name = "{{.*}}Ej"}
  ! CHECK-DAG: %[[STR:.*]] = fir.alloca !fir.array{{.*}} {{{.*}}uniq_name = "{{.*}}Estr"}
  ! CHECK: %[[BOX:.*]] = fir.convert %[[unbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1>>
  ! CHECK: %[[PTR:.*]] = fir.load %[[BOX]] : !fir.ref<!fir.char<1>>
  ! CHECK: %[[CHAR:.*]] = fir.extract_value %[[PTR]], [0 : index] :
  ! CHECK: %[[ARG:.*]] = arith.extui %[[CHAR]] : i8 to i32
  ! CHECK: fir.call @{{.*}}OutputInteger32{{.*}}%[[ARG]]
  ! CHECK: fir.call @{{.*}}EndIoStatement
  print *, ichar(c)

  ! CHECK-DAG: %{{.*}} = fir.load %[[J]] : !fir.ref<i32>
  ! CHECK: %[[PTR1:.*]] = fir.coordinate_of %[[STR]], %
  ! CHECK: %[[PTR2:.*]] = fir.load %[[PTR1]] : !fir.ref<!fir.char<1>>
  ! CHECK: %[[CHAR:.*]] = fir.extract_value %[[PTR2]], [0 : index] :
  ! CHECK: %[[ARG:.*]] = arith.extui %[[CHAR]] : i8 to i32
  ! CHECK: fir.call @{{.*}}OutputInteger32{{.*}}%[[ARG]]
  ! CHECK: fir.call @{{.*}}EndIoStatement
  print *, ichar(str(J))
end subroutine
