// RUN: fir-opt %s | tco | FileCheck %s


// CHECK-LABEL: define void @_QPtest_callee({ i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %0)
func @_QPtest_callee(%arg0: !fir.box<!fir.array<?xi32>>) {
  return
}

// CHECK-LABEL: define void @_QPtest_slice()
func @_QPtest_slice() {
// CHECK:  %[[a1:.*]] = alloca { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8, 
// CHECK:  %[[a2:.*]] = alloca [20 x i32], i64 1, align 4, 
// CHECK:  %[[a3:.*]] = getelementptr [20 x i32], [20 x i32]* %[[a2]], i64 0, i64 0, 
// CHECK:  %[[a4:.*]] = insertvalue { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }  
// CHECK:  { i32* undef, i64 4, i32 20180515, i8 1, i8 9, i8 0, i8 0, [1 x [3 x i64]] 
// CHECK: [i64 1, i64 5, i64 8]] }, i32* %[[a3]], 0, 
// CHECK:  store { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %[[a4]], { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %[[a1]], align 8
// CHECK:  call void @_QPtest_callee({ i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %[[a1]]), 
  %c20 = arith.constant 20 : index
  %c1_i64 = arith.constant 1 : i64
  %c10_i64 = arith.constant 10 : i64
  %c2_i64 = arith.constant 2 : i64
  %0 = fir.alloca !fir.array<20xi32> {bindc_name = "x", uniq_name = "_QFtest_sliceEx"}
  %1 = fir.shape %c20 : (index) -> !fir.shape<1>
  %2 = fir.slice %c1_i64, %c10_i64, %c2_i64 : (i64, i64, i64) -> !fir.slice<1>
  %3 = fir.embox %0(%1) [%2] : (!fir.ref<!fir.array<20xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xi32>>
  fir.call @_QPtest_callee(%3) : (!fir.box<!fir.array<?xi32>>) -> ()
  return
}

// CHECK-LABEL: define void @_QPtest_dt_callee({ i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %0)
func @_QPtest_dt_callee(%arg0: !fir.box<!fir.array<?xi32>>) {
  return
}

// CHECK-LABEL: define void @_QPtest_dt_slice()
func @_QPtest_dt_slice() {
// CHECK:  %[[a1:.*]] = alloca { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8, 
// CHECK:  %[[a3:.*]] = alloca [20 x %_QFtest_dt_sliceTt], i64 1, align 8,
// CHECK:  %[[a4:.*]] = getelementptr [20 x %_QFtest_dt_sliceTt], [20 x %_QFtest_dt_sliceTt]* %[[a3]], i64 0, i64 0, i32 0,
// CHECK: %[[a5:.*]] = insertvalue { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }
// CHECK-SAME: { i32* undef, i64 4, i32 20180515, i8 1, i8 9, i8 0, i8 0, [1 x [3 x i64]]
// CHECK-SAME: [i64 1, i64 5, i64 mul
// CHECK-SAME: (i64 ptrtoint (%_QFtest_dt_sliceTt* getelementptr (%_QFtest_dt_sliceTt, %_QFtest_dt_sliceTt* null, i64 1) to i64), i64 2)]] }
// CHECK-SAME: , i32* %[[a4]], 0

// CHECK: store { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %[[a5]], { i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %[[a1]], align 8,
// CHECK:  call void @_QPtest_dt_callee({ i32*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %[[a1]]),
  %c20 = arith.constant 20 : index
  %c1_i64 = arith.constant 1 : i64
  %c10_i64 = arith.constant 10 : i64
  %c2_i64 = arith.constant 2 : i64
  %0 = fir.alloca i32 {bindc_name = "v", uniq_name = "_QFtest_dt_sliceEv"}
  %1 = fir.alloca !fir.array<20x!fir.type<_QFtest_dt_sliceTt{i:i32,j:i32}>> {bindc_name = "x", uniq_name = "_QFtest_dt_sliceEx"}
  %2 = fir.field_index i, !fir.type<_QFtest_dt_sliceTt{i:i32,j:i32}>
  %3 = fir.shape %c20 : (index) -> !fir.shape<1>
  %4 = fir.slice %c1_i64, %c10_i64, %c2_i64 path %2 : (i64, i64, i64, !fir.field) -> !fir.slice<1>
  %5 = fir.embox %1(%3) [%4] : (!fir.ref<!fir.array<20x!fir.type<_QFtest_dt_sliceTt{i:i32,j:i32}>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xi32>>
  fir.call @_QPtest_dt_callee(%5) : (!fir.box<!fir.array<?xi32>>) -> ()
  return
}

func private @takesRank2CharBox(!fir.box<!fir.array<?x?x!fir.char<1,?>>>)

// CHECK-LABEL: define void @emboxSubstring(
// CHECK-SAME: [3 x [2 x [4 x i8]]]* %[[arg0:.*]])
func @emboxSubstring(%arg0: !fir.ref<!fir.array<2x3x!fir.char<1,4>>>) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %0 = fir.shape %c2, %c3 : (index, index) -> !fir.shape<2>
  %1 = fir.slice %c1, %c2, %c1, %c1, %c3, %c1 substr %c1, %c2 : (index, index, index, index, index, index, index, index) -> !fir.slice<2>
  %2 = fir.embox %arg0(%0) [%1] : (!fir.ref<!fir.array<2x3x!fir.char<1,4>>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>

  // CHECK: %[[addr:.*]] = getelementptr [3 x [2 x [4 x i8]]], [3 x [2 x [4 x i8]]]* %[[arg0]], i64 0, i64 0, i64 0
  // CHECK: %[[substringAddr:.*]] = getelementptr [4 x i8], [4 x i8]* %[[addr]], i64 0, i64 1
  // CHECK: insertvalue {[[descriptorType:.*]]} { i8* undef, i64 2, i32 20180515, i8 2, i8 32, i8 0, i8 0,
  // CHECK-SAME: [2 x [3 x i64]] [{{\[}}3 x i64] [i64 1, i64 2, i64 4], [3 x i64] [i64 1, i64 3, i64 8]] },
  // CHECK-SAME: i8* %[[substringAddr]], 0

  fir.call @takesRank2CharBox(%2) : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> ()
  return
}
