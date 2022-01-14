// RUN: tco -o - %s | FileCheck %s

!type_defined_elsewhere = type !fir.type<_QMsome_moduleTtype_defined_elsewhere{j:i32}>
fir.global internal @_QFfooEx2 : !fir.box<!fir.heap<!type_defined_elsewhere>> {
  %0 = fir.zero_bits !fir.heap<!type_defined_elsewhere>
  %1 = fir.embox %0 : (!fir.heap<!type_defined_elsewhere>) -> !fir.box<!fir.heap<!type_defined_elsewhere>>
  fir.has_value %1 : !fir.box<!fir.heap<!type_defined_elsewhere>>
}
// CHECK: @_QMsome_moduleE.dt.type_defined_elsewhere = extern_weak global i8
// CHECK: @_QFfooEx2 = internal global { %_QMsome_moduleTtype_defined_elsewhere*, i64, i32, i8, i8, i8, i8, i8*, [1 x i64] }
// CHECK-SAME: { %_QMsome_moduleTtype_defined_elsewhere* null, i64 ptrtoint (%_QMsome_moduleTtype_defined_elsewhere* getelementptr (%_QMsome_moduleTtype_defined_elsewhere, %_QMsome_moduleTtype_defined_elsewhere* null, i32 1) to i64),
// CHECK-SAME: i32 20180515, i8 0, i8 42, i8 2, i8 1, i8* @_QMsome_moduleE.dt.type_defined_elsewhere, [1 x i64] undef }

!sometype = type !fir.type<_QFfooTsometype{num:i32,values:!fir.box<!fir.ptr<!fir.array<?x?xf32>>>}>
fir.global internal @_QFfooE.dt.sometype : i8
fir.global internal @_QFfooEx : !fir.box<!fir.heap<!sometype>> {
  %0 = fir.zero_bits !fir.heap<!sometype>
  %1 = fir.embox %0 : (!fir.heap<!sometype>) -> !fir.box<!fir.heap<!sometype>>
  fir.has_value %1 : !fir.box<!fir.heap<!sometype>>
}
// CHECK: @_QFfooEx = internal global { %_QFfooTsometype*, i64, i32, i8, i8, i8, i8, i8*, [1 x i64] }
// CHECK-SAME: { %_QFfooTsometype* null, i64 ptrtoint (%_QFfooTsometype* getelementptr (%_QFfooTsometype, %_QFfooTsometype* null, i32 1) to i64),
// CHECK-SAME: i32 20180515, i8 0, i8 42, i8 2, i8 1, i8* @_QFfooE.dt.sometype, [1 x i64] undef }
