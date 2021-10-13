// RUN: tco --cfg-conversion="always-execute-loop-body=true" %s | tco | FileCheck %s
// RUN: fir-opt --cfg-conversion="always-execute-loop-body=true" %s | FileCheck %s --check-prefix=FIROPT
// RUN: fir-opt --cfg-conversion %s | FileCheck %s --check-prefix=NOOPT

// CHECK-LABEL: @x
func @x(%addr : !fir.ref<index>) {
  %bound = constant 452 : index
  %step = constant 1 : index
  // CHECK: %[[phi:.*]] = phi i64 [{{.*}}], [ 1,
  // CHECK: = icmp sgt i64 %[[phi]], 0
  fir.do_loop %iv = %bound to %bound step %step {
    // CHECK: call void @y(i64* %
    fir.call @y(%addr) : (!fir.ref<index>) -> ()
  }
  // CHECK: ret void
  return
}

func private @y(%addr : !fir.ref<index>)

// FIROPT-LABEL:   func @x(
// FIROPT-SAME:            %[[VAL_0:.*]]: !fir.ref<index>) {
// FIROPT:           %[[VAL_1:.*]] = constant 452 : index
// FIROPT:           %[[VAL_2:.*]] = constant 1 : index
// FIROPT:           %[[VAL_3:.*]] = subi %[[VAL_1]], %[[VAL_1]] : index
// FIROPT:           %[[VAL_4:.*]] = addi %[[VAL_3]], %[[VAL_2]] : index
// FIROPT:           %[[VAL_5:.*]] = divi_signed %[[VAL_4]], %[[VAL_2]] : index
// FIROPT:           %[[VAL_6:.*]] = constant 0 : index
// FIROPT:           %[[VAL_7:.*]] = cmpi sle, %[[VAL_5]], %[[VAL_6]] : index
// FIROPT:           %[[VAL_8:.*]] = constant 1 : index
// FIROPT:           %[[VAL_9:.*]] = select %[[VAL_7]], %[[VAL_8]], %[[VAL_5]] : index
// FIROPT:           br ^bb1(%[[VAL_1]], %[[VAL_9]] : index, index)
// FIROPT:         ^bb1(%[[VAL_10:.*]]: index, %[[VAL_11:.*]]: index):
// FIROPT:           %[[VAL_12:.*]] = constant 0 : index
// FIROPT:           %[[VAL_13:.*]] = cmpi sgt, %[[VAL_11]], %[[VAL_12]] : index
// FIROPT:           cond_br %[[VAL_13]], ^bb2, ^bb3
// FIROPT:         ^bb2:
// FIROPT:           fir.call @y(%[[VAL_0]]) : (!fir.ref<index>) -> ()
// FIROPT:           %[[VAL_14:.*]] = addi %[[VAL_10]], %[[VAL_2]] : index
// FIROPT:           %[[VAL_15:.*]] = constant 1 : index
// FIROPT:           %[[VAL_16:.*]] = subi %[[VAL_11]], %[[VAL_15]] : index
// FIROPT:           br ^bb1(%[[VAL_14]], %[[VAL_16]] : index, index)
// FIROPT:         ^bb3:
// FIROPT:           return
// FIROPT:         }
// FIROPT:         func private @y(!fir.ref<index>)

// NOOPT-LABEL:   func @x(
// NOOPT-SAME:            %[[VAL_0:.*]]: !fir.ref<index>) {
// NOOPT:           %[[VAL_1:.*]] = constant 452 : index
// NOOPT:           %[[VAL_2:.*]] = constant 1 : index
// NOOPT:           %[[VAL_3:.*]] = subi %[[VAL_1]], %[[VAL_1]] : index
// NOOPT:           %[[VAL_4:.*]] = addi %[[VAL_3]], %[[VAL_2]] : index
// NOOPT:           %[[VAL_5:.*]] = divi_signed %[[VAL_4]], %[[VAL_2]] : index
// NOOPT:           br ^bb1(%[[VAL_1]], %[[VAL_5]] : index, index)
// NOOPT:         ^bb1(%[[VAL_6:.*]]: index, %[[VAL_7:.*]]: index):
// NOOPT:           %[[VAL_8:.*]] = constant 0 : index
// NOOPT:           %[[VAL_9:.*]] = cmpi sgt, %[[VAL_7]], %[[VAL_8]] : index
// NOOPT:           cond_br %[[VAL_9]], ^bb2, ^bb3
// NOOPT:         ^bb2:
// NOOPT:           fir.call @y(%[[VAL_0]]) : (!fir.ref<index>) -> ()
// NOOPT:           %[[VAL_10:.*]] = addi %[[VAL_6]], %[[VAL_2]] : index
// NOOPT:           %[[VAL_11:.*]] = constant 1 : index
// NOOPT:           %[[VAL_12:.*]] = subi %[[VAL_7]], %[[VAL_11]] : index
// NOOPT:           br ^bb1(%[[VAL_10]], %[[VAL_12]] : index, index)
// NOOPT:         ^bb3:
// NOOPT:           return
// NOOPT:         }
// NOOPT:         func private @y(!fir.ref<index>)
