// RUN: tco %s | FileCheck %s
// RUN: fir-opt --split-input-file --cfg-conversion %s | FileCheck %s --check-prefix=FIROPT

// CHECK-LABEL: @x
func @x(%lb : index, %ub : index, %step : index, %b : i1, %addr : !fir.ref<index>) {
  // CHECK: [[LOOP:[0-9]+]]:
  // CHECK: %[[COND:.*]] = icmp sgt i64 %{{.*}}, 0
  // CHECK: br i1 %[[COND]]
  fir.do_loop %iv = %lb to %ub step %step unordered {
    fir.if %b {
      // CHECK: store i64
      fir.store %iv to %addr : !fir.ref<index>
    } else {
      // CHECK: store i64 0
      %zero = arith.constant 0 : index
      fir.store %zero to %addr : !fir.ref<index>
    }
    // CHECK: br label %[[LOOP]]
  }
  // CHECK: ret void
  return
}

func private @f2() -> i1

// FIROPT:     func @x(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: !fir.ref<index>) {
// FIROPT:       %[[VAL_5:.*]] = arith.subi %[[VAL_1]], %[[VAL_0]] : index
// FIROPT:       %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %[[VAL_2]] : index
// FIROPT:       %[[VAL_7:.*]] = arith.divsi %[[VAL_6]], %[[VAL_2]] : index
// FIROPT:       br ^bb1(%[[VAL_0]], %[[VAL_7]] : index, index)
// FIROPT:     ^bb1(%[[VAL_8:.*]]: index, %[[VAL_9:.*]]: index):
// FIROPT:       %[[VAL_10:.*]] = arith.constant 0 : index
// FIROPT:       %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_9]], %[[VAL_10]] : index
// FIROPT:       cond_br %[[VAL_11]], ^bb2, ^bb6
// FIROPT:     ^bb2:
// FIROPT:       cond_br %[[VAL_3]], ^bb3, ^bb4
// FIROPT:     ^bb3:
// FIROPT:       fir.store %[[VAL_8]] to %[[VAL_4]] : !fir.ref<index>
// FIROPT:       br ^bb5
// FIROPT:     ^bb4:
// FIROPT:       %[[VAL_12:.*]] = arith.constant 0 : index
// FIROPT:       fir.store %[[VAL_12]] to %[[VAL_4]] : !fir.ref<index>
// FIROPT:       br ^bb5
// FIROPT:     ^bb5:
// FIROPT:       %[[VAL_13:.*]] = arith.addi %[[VAL_8]], %[[VAL_2]] : index
// FIROPT:       %[[VAL_14:.*]] = arith.constant 1 : index
// FIROPT:       %[[VAL_15:.*]] = arith.subi %[[VAL_9]], %[[VAL_14]] : index
// FIROPT:       br ^bb1(%[[VAL_13]], %[[VAL_15]] : index, index)
// FIROPT:     ^bb6:
// FIROPT:       return
// FIROPT:     }
// FIROPT:     func private @f2() -> i1

// -----

// CHECK-LABEL: @x2
func @x2(%lo : index, %up : index, %ok : i1) {
  %c1 = arith.constant 1 : index
  // CHECK-DAG: %[[count:.*]] = phi i64
  // CHECK-DAG: %[[exit:.*]] = phi i1
  // CHECK: %[[cond:.*]] = icmp sle i64 %[[count]], %
  // CHECK: %[[and:.*]] = and i1 %[[exit]], %[[cond]]
  // CHECK: br i1 %[[and]]
  %unused = fir.iterate_while (%i = %lo to %up step %c1) and (%ok1 = %ok) {
    %ok2 = fir.call @f2() : () -> i1
    fir.result %ok2 : i1
  }
  // CHECK: ret
  return
}

func private @f3(i16)

// FIROPT:   func @x2(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: i1) {
// FIROPT:     %[[VAL_3:.*]] = arith.constant 1 : index
// FIROPT:     br ^bb1(%[[VAL_0]], %[[VAL_2]] : index, i1)
// FIROPT:   ^bb1(%[[VAL_4:.*]]: index, %[[VAL_5:.*]]: i1):
// FIROPT:     %[[VAL_6:.*]] = arith.constant 0 : index
// FIROPT:     %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_6]], %[[VAL_3]] : index
// FIROPT:     %[[VAL_8:.*]] = arith.cmpi sle, %[[VAL_4]], %[[VAL_1]] : index
// FIROPT:     %[[VAL_9:.*]] = arith.cmpi slt, %[[VAL_3]], %[[VAL_6]] : index
// FIROPT:     %[[VAL_10:.*]] = arith.cmpi sle, %[[VAL_1]], %[[VAL_4]] : index
// FIROPT:     %[[VAL_11:.*]] = arith.andi %[[VAL_7]], %[[VAL_8]] : i1
// FIROPT:     %[[VAL_12:.*]] = arith.andi %[[VAL_9]], %[[VAL_10]] : i1
// FIROPT:     %[[VAL_13:.*]] = arith.ori %[[VAL_11]], %[[VAL_12]] : i1
// FIROPT:     %[[VAL_14:.*]] = arith.andi %[[VAL_5]], %[[VAL_13]] : i1
// FIROPT:     cond_br %[[VAL_14]], ^bb2, ^bb3
// FIROPT:   ^bb2:
// FIROPT:     %[[VAL_15:.*]] = fir.call @f2() : () -> i1
// FIROPT:     %[[VAL_16:.*]] = arith.addi %[[VAL_4]], %[[VAL_3]] : index
// FIROPT:     br ^bb1(%[[VAL_16]], %[[VAL_15]] : index, i1)
// FIROPT:   ^bb3:
// FIROPT:     return
// FIROPT:   }
// FIROPT:   func private @f3(i16)

// -----

// do_loop with an extra loop-carried value
// CHECK-LABEL: @x3
func @x3(%lo : index, %up : index) -> i1 {
  %c1 = arith.constant 1 : index
  %ok1 = arith.constant true
  // CHECK-DAG: %[[ok:.*]] = phi i1
  // CHECK-DAG: %[[count:.*]] = phi i64
  // CHECK: = icmp sgt i64 %[[count]], 0
  %ok2 = fir.do_loop %i = %lo to %up step %c1 iter_args(%j = %ok1) -> i1 {
    %ok = fir.call @f2() : () -> i1
    fir.result %ok : i1
    // CHECK: = sub i64 %[[count]], 1
  }
  // CHECK: ret i1 %[[ok]]
  return %ok2 : i1
}

// FIROPT-LABEL:   func @x3(
// FIROPT-SAME:             %[[VAL_0:.*]]: index,
// FIROPT-SAME:             %[[VAL_1:.*]]: index) -> i1 {
// FIROPT:           %[[VAL_2:.*]] = arith.constant 1 : index
// FIROPT:           %[[VAL_3:.*]] = arith.constant true
// FIROPT:           %[[VAL_4:.*]] = arith.subi %[[VAL_1]], %[[VAL_0]] : index
// FIROPT:           %[[VAL_5:.*]] = arith.addi %[[VAL_4]], %[[VAL_2]] : index
// FIROPT:           %[[VAL_6:.*]] = arith.divsi %[[VAL_5]], %[[VAL_2]] : index
// FIROPT:           br ^bb1(%[[VAL_0]], %[[VAL_3]], %[[VAL_6]] : index, i1, index)
// FIROPT:         ^bb1(%[[VAL_7:.*]]: index, %[[VAL_8:.*]]: i1, %[[VAL_9:.*]]: index):
// FIROPT:           %[[VAL_10:.*]] = arith.constant 0 : index
// FIROPT:           %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_9]], %[[VAL_10]] : index
// FIROPT:           cond_br %[[VAL_11]], ^bb2, ^bb3
// FIROPT:         ^bb2:
// FIROPT:           %[[VAL_12:.*]] = fir.call @f2() : () -> i1
// FIROPT:           %[[VAL_13:.*]] = arith.addi %[[VAL_7]], %[[VAL_2]] : index
// FIROPT:           %[[VAL_14:.*]] = arith.constant 1 : index
// FIROPT:           %[[VAL_15:.*]] = arith.subi %[[VAL_9]], %[[VAL_14]] : index
// FIROPT:           br ^bb1(%[[VAL_13]], %[[VAL_12]], %[[VAL_15]] : index, i1, index)
// FIROPT:         ^bb3:
// FIROPT:           return %[[VAL_8]] : i1
// FIROPT:         }

// -----

// iterate_while with an extra loop-carried value
// CHECK-LABEL: @y3
func @y3(%lo : index, %up : index) -> i1 {
  %c1 = arith.constant 1 : index
  %ok1 = arith.constant true
  // CHECK: %[[ok4:.*]] = call i1 @f2()
  %ok4 = fir.call @f2() : () -> i1
  // CHECK-DAG: %[[count:.*]] = phi i64
  // CHECK-DAG: %[[ok3:.*]] = phi i1 {{.*}}[ true
  // CHECK-DAG: %[[j:.*]] = phi i1 {{.*}}[ %[[ok4]]
  // CHECK: %[[prev:.*]] = icmp sle i64 %[[count]],
  // CHECK: = and i1 %[[ok3]], %[[prev]]
  %ok2:2 = fir.iterate_while (%i = %lo to %up step %c1) and (%ok3 = %ok1) iter_args(%j = %ok4) -> i1 {
    %ok = fir.call @f2() : () -> i1
    fir.result %ok3, %ok : i1, i1
    // CHECK: = add i64 %[[count]], 1
  }
  // CHECK: %[[result:.*]] = and i1 %[[ok3]], %[[j]]
  %andok = arith.andi %ok2#0, %ok2#1 : i1
  // CHECK: ret i1 %[[result]]
  return %andok : i1
}

func private @f4(i32) -> i1

// FIROPT-LABEL:   func @y3(
// FIROPT-SAME:             %[[VAL_0:.*]]: index,
// FIROPT-SAME:             %[[VAL_1:.*]]: index) -> i1 {
// FIROPT:           %[[VAL_2:.*]] = arith.constant 1 : index
// FIROPT:           %[[VAL_3:.*]] = arith.constant true
// FIROPT:           %[[VAL_4:.*]] = fir.call @f2() : () -> i1
// FIROPT:           br ^bb1(%[[VAL_0]], %[[VAL_3]], %[[VAL_4]] : index, i1, i1)
// FIROPT:         ^bb1(%[[VAL_5:.*]]: index, %[[VAL_6:.*]]: i1, %[[VAL_7:.*]]: i1):
// FIROPT:           %[[VAL_8:.*]] = arith.constant 0 : index
// FIROPT:           %[[VAL_9:.*]] = arith.cmpi slt, %[[VAL_8]], %[[VAL_2]] : index
// FIROPT:           %[[VAL_10:.*]] = arith.cmpi sle, %[[VAL_5]], %[[VAL_1]] : index
// FIROPT:           %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_2]], %[[VAL_8]] : index
// FIROPT:           %[[VAL_12:.*]] = arith.cmpi sle, %[[VAL_1]], %[[VAL_5]] : index
// FIROPT:           %[[VAL_13:.*]] = arith.andi %[[VAL_9]], %[[VAL_10]] : i1
// FIROPT:           %[[VAL_14:.*]] = arith.andi %[[VAL_11]], %[[VAL_12]] : i1
// FIROPT:           %[[VAL_15:.*]] = arith.ori %[[VAL_13]], %[[VAL_14]] : i1
// FIROPT:           %[[VAL_16:.*]] = arith.andi %[[VAL_6]], %[[VAL_15]] : i1
// FIROPT:           cond_br %[[VAL_16]], ^bb2, ^bb3
// FIROPT:         ^bb2:
// FIROPT:           %[[VAL_17:.*]] = fir.call @f2() : () -> i1
// FIROPT:           %[[VAL_18:.*]] = arith.addi %[[VAL_5]], %[[VAL_2]] : index
// FIROPT:           br ^bb1(%[[VAL_18]], %[[VAL_6]], %[[VAL_17]] : index, i1, i1)
// FIROPT:         ^bb3:
// FIROPT:           %[[VAL_19:.*]] = arith.andi %[[VAL_6]], %[[VAL_7]] : i1
// FIROPT:           return %[[VAL_19]] : i1
// FIROPT:         }
// FIROPT:         func private @f4(i32) -> i1

// -----

// do_loop that returns the final value of the induction
// CHECK-LABEL: @x4
// CHECK-SAME: (i64 %[[lo:.*]],
func @x4(%lo : index, %up : index) -> index {
  %c1 = arith.constant 1 : index
  // CHECK: %[[top:.*]] = add i64
  // CHECK-DAG: %[[i:.*]] = phi i64 {{.*}}[ %[[lo]],
  // CHECK-DAG: %[[count:.*]] = phi i64 {{.*}}[ %[[top]],
  // CHECK: icmp sgt i64 %[[count]],
  %v = fir.do_loop %i = %lo to %up step %c1 -> index {
    // CHECK: trunc i64 %[[i]] to i32
    %i1 = fir.convert %i : (index) -> i32
    // CHECK: call i1 @f4
    %ok = fir.call @f4(%i1) : (i32) -> i1
    fir.result %i : index
  }
  // CHECK: ret i64 %[[i]]
  return %v : index
}

// FIROPT-LABEL:   func @x4(
// FIROPT-SAME:             %[[VAL_0:.*]]: index,
// FIROPT-SAME:             %[[VAL_1:.*]]: index) -> index {
// FIROPT:           %[[VAL_2:.*]] = arith.constant 1 : index
// FIROPT:           %[[VAL_3:.*]] = arith.subi %[[VAL_1]], %[[VAL_0]] : index
// FIROPT:           %[[VAL_4:.*]] = arith.addi %[[VAL_3]], %[[VAL_2]] : index
// FIROPT:           %[[VAL_5:.*]] = arith.divsi %[[VAL_4]], %[[VAL_2]] : index
// FIROPT:           br ^bb1(%[[VAL_0]], %[[VAL_5]] : index, index)
// FIROPT:         ^bb1(%[[VAL_6:.*]]: index, %[[VAL_7:.*]]: index):
// FIROPT:           %[[VAL_8:.*]] = arith.constant 0 : index
// FIROPT:           %[[VAL_9:.*]] = arith.cmpi sgt, %[[VAL_7]], %[[VAL_8]] : index
// FIROPT:           cond_br %[[VAL_9]], ^bb2, ^bb3
// FIROPT:         ^bb2:
// FIROPT:           %[[VAL_10:.*]] = fir.convert %[[VAL_6]] : (index) -> i32
// FIROPT:           %[[VAL_11:.*]] = fir.call @f4(%[[VAL_10]]) : (i32) -> i1
// FIROPT:           %[[VAL_12:.*]] = arith.addi %[[VAL_6]], %[[VAL_2]] : index
// FIROPT:           %[[VAL_13:.*]] = arith.constant 1 : index
// FIROPT:           %[[VAL_14:.*]] = arith.subi %[[VAL_7]], %[[VAL_13]] : index
// FIROPT:           br ^bb1(%[[VAL_12]], %[[VAL_14]] : index, index)
// FIROPT:         ^bb3:
// FIROPT:           return %[[VAL_6]] : index
// FIROPT:         }

// -----

// iterate_while that returns the final value of both inductions
// CHECK-LABEL: @y4
func @y4(%lo : index, %up : index) -> index {
  %c1 = arith.constant 1 : index
  %ok1 = arith.constant true
  // CHECK-DAG: %[[i:.*]] = phi i64 [
  // CHECK-DAG: %[[ok2:.*]] = phi i1 [
  // CHECK: icmp sle i64 %[[i]]
  // CHECK: and i1
  %v:2 = fir.iterate_while (%i = %lo to %up step %c1) and (%ok2 = %ok1) -> (index, i1) {
    %i1 = fir.convert %i : (index) -> i32
    // CHECK: call i1 @f4
    %ok = fir.call @f4(%i1) : (i32) -> i1
    fir.result %i, %ok : index, i1
  }
  // CHECK: ret i64 %[[i]]
  return %v#0 : index
}

// FIROPT-LABEL:   func @y4(
// FIROPT-SAME:             %[[VAL_0:.*]]: index,
// FIROPT-SAME:             %[[VAL_1:.*]]: index) -> index {
// FIROPT:           %[[VAL_2:.*]] = arith.constant 1 : index
// FIROPT:           %[[VAL_3:.*]] = arith.constant true
// FIROPT:           br ^bb1(%[[VAL_0]], %[[VAL_3]] : index, i1)
// FIROPT:         ^bb1(%[[VAL_4:.*]]: index, %[[VAL_5:.*]]: i1):
// FIROPT:           %[[VAL_6:.*]] = arith.constant 0 : index
// FIROPT:           %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_6]], %[[VAL_2]] : index
// FIROPT:           %[[VAL_8:.*]] = arith.cmpi sle, %[[VAL_4]], %[[VAL_1]] : index
// FIROPT:           %[[VAL_9:.*]] = arith.cmpi slt, %[[VAL_2]], %[[VAL_6]] : index
// FIROPT:           %[[VAL_10:.*]] = arith.cmpi sle, %[[VAL_1]], %[[VAL_4]] : index
// FIROPT:           %[[VAL_11:.*]] = arith.andi %[[VAL_7]], %[[VAL_8]] : i1
// FIROPT:           %[[VAL_12:.*]] = arith.andi %[[VAL_9]], %[[VAL_10]] : i1
// FIROPT:           %[[VAL_13:.*]] = arith.ori %[[VAL_11]], %[[VAL_12]] : i1
// FIROPT:           %[[VAL_14:.*]] = arith.andi %[[VAL_5]], %[[VAL_13]] : i1
// FIROPT:           cond_br %[[VAL_14]], ^bb2, ^bb3
// FIROPT:         ^bb2:
// FIROPT:           %[[VAL_15:.*]] = fir.convert %[[VAL_4]] : (index) -> i32
// FIROPT:           %[[VAL_16:.*]] = fir.call @f4(%[[VAL_15]]) : (i32) -> i1
// FIROPT:           %[[VAL_17:.*]] = arith.addi %[[VAL_4]], %[[VAL_2]] : index
// FIROPT:           br ^bb1(%[[VAL_17]], %[[VAL_16]] : index, i1)
// FIROPT:         ^bb3:
// FIROPT:           return %[[VAL_4]] : index
// FIROPT:         }


// -----

// do_loop that returns the final induction value
// and an extra loop-carried value
// CHECK-LABEL: @x5
// CHECK-SAME: (i64 %[[lo:.*]],
func @x5(%lo : index, %up : index) -> index {
  %c1 = arith.constant 1 : index
  // CHECK: %[[top:.*]] = add i64
  %s1 = arith.constant 42 : i16
  // CHECK-DAG: %[[i:.*]] = phi i64 {{.*}}[ %[[lo]],
  // CHECK-DAG: %[[count:.*]] = phi i64 {{.*}}[ %[[top]],
  // CHECK-DAG: %[[s:.*]] = phi i16
  // CHECK: icmp sgt i64 %[[count]]
  %v:2 = fir.do_loop %i = %lo to %up step %c1 iter_args(%s = %s1) -> (index, i16) {
    // CHECK: call i1 @f2
    %ok = fir.call @f2() : () -> i1
    %s2 = fir.convert %ok : (i1) -> i16
    fir.result %i, %s2 : index, i16
    // CHECK: add i64 %[[i]], 1
    // CHECK: sub i64 %[[count]], 1
  }
  // CHECK: call void @f3
  fir.call @f3(%v#1) : (i16) -> ()
  // CHECK: ret i64 %[[i]]
  return %v#0 : index
}

// FIROPT-LABEL:   func @x5(
// FIROPT-SAME:             %[[VAL_0:.*]]: index,
// FIROPT-SAME:             %[[VAL_1:.*]]: index) -> index {
// FIROPT:           %[[VAL_2:.*]] = arith.constant 1 : index
// FIROPT:           %[[VAL_3:.*]] = arith.constant 42 : i16
// FIROPT:           %[[VAL_4:.*]] = arith.subi %[[VAL_1]], %[[VAL_0]] : index
// FIROPT:           %[[VAL_5:.*]] = arith.addi %[[VAL_4]], %[[VAL_2]] : index
// FIROPT:           %[[VAL_6:.*]] = arith.divsi %[[VAL_5]], %[[VAL_2]] : index
// FIROPT:           br ^bb1(%[[VAL_0]], %[[VAL_3]], %[[VAL_6]] : index, i16, index)
// FIROPT:         ^bb1(%[[VAL_7:.*]]: index, %[[VAL_8:.*]]: i16, %[[VAL_9:.*]]: index):
// FIROPT:           %[[VAL_10:.*]] = arith.constant 0 : index
// FIROPT:           %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_9]], %[[VAL_10]] : index
// FIROPT:           cond_br %[[VAL_11]], ^bb2, ^bb3
// FIROPT:         ^bb2:
// FIROPT:           %[[VAL_12:.*]] = fir.call @f2() : () -> i1
// FIROPT:           %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i1) -> i16
// FIROPT:           %[[VAL_14:.*]] = arith.addi %[[VAL_7]], %[[VAL_2]] : index
// FIROPT:           %[[VAL_15:.*]] = arith.constant 1 : index
// FIROPT:           %[[VAL_16:.*]] = arith.subi %[[VAL_9]], %[[VAL_15]] : index
// FIROPT:           br ^bb1(%[[VAL_14]], %[[VAL_13]], %[[VAL_16]] : index, i16, index)
// FIROPT:         ^bb3:
// FIROPT:           fir.call @f3(%[[VAL_8]]) : (i16) -> ()
// FIROPT:           return %[[VAL_7]] : index
// FIROPT:         }

// -----

// iterate_while that returns the both induction values
// and an extra loop-carried value
// CHECK-LABEL: @y5
// CHECK-SAME: (i64 %[[lo:.*]],
func @y5(%lo : index, %up : index) -> index {
  %c1 = arith.constant 1 : index
  %s1 = arith.constant 42 : i16
  %ok1 = arith.constant true
  // CHECK-DAG: %[[i:.*]] = phi i64 {{.*}}[ %[[lo]],
  // CHECK-DAG: %[[ok2:.*]] = phi i1 {{.*}}[ true, 
  // CHECK-DAG: %[[s:.*]] = phi i16 {{.*}}[ 42,
  // CHECK: icmp sle i64 %[[i]]
  // CHECK: and i1 %[[ok2]]
  %v:3 = fir.iterate_while (%i = %lo to %up step %c1) and (%ok2 = %ok1) iter_args(%s = %s1) -> (index, i1, i16) {
    // CHECK: call i1 @f2
    %ok = fir.call @f2() : () -> i1
    %s2 = fir.convert %ok : (i1) -> i16
    fir.result %i, %ok, %s2 : index, i1, i16
    // CHECK: add i64 %[[i]], 1
  }
  // CHECK: br i1 %[[ok2]],
  fir.if %v#1 {
    %arg = arith.constant 0 : i32
    // CHECK: call i1 @f4
    %ok4 = fir.call @f4(%arg) : (i32) -> i1
  }
  // CHECK: call void @f3(i16 %[[s]])
  fir.call @f3(%v#2) : (i16) -> ()
  // CHECK: ret i64 %[[i]]
  return %v#0 : index
}

// FIROPT-LABEL:   func @y5(
// FIROPT-SAME:             %[[VAL_0:.*]]: index,
// FIROPT-SAME:             %[[VAL_1:.*]]: index) -> index {
// FIROPT:           %[[VAL_2:.*]] = arith.constant 1 : index
// FIROPT:           %[[VAL_3:.*]] = arith.constant 42 : i16
// FIROPT:           %[[VAL_4:.*]] = arith.constant true
// FIROPT:           br ^bb1(%[[VAL_0]], %[[VAL_4]], %[[VAL_3]] : index, i1, i16)
// FIROPT:         ^bb1(%[[VAL_5:.*]]: index, %[[VAL_6:.*]]: i1, %[[VAL_7:.*]]: i16):
// FIROPT:           %[[VAL_8:.*]] = arith.constant 0 : index
// FIROPT:           %[[VAL_9:.*]] = arith.cmpi slt, %[[VAL_8]], %[[VAL_2]] : index
// FIROPT:           %[[VAL_10:.*]] = arith.cmpi sle, %[[VAL_5]], %[[VAL_1]] : index
// FIROPT:           %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_2]], %[[VAL_8]] : index
// FIROPT:           %[[VAL_12:.*]] = arith.cmpi sle, %[[VAL_1]], %[[VAL_5]] : index
// FIROPT:           %[[VAL_13:.*]] = arith.andi %[[VAL_9]], %[[VAL_10]] : i1
// FIROPT:           %[[VAL_14:.*]] = arith.andi %[[VAL_11]], %[[VAL_12]] : i1
// FIROPT:           %[[VAL_15:.*]] = arith.ori %[[VAL_13]], %[[VAL_14]] : i1
// FIROPT:           %[[VAL_16:.*]] = arith.andi %[[VAL_6]], %[[VAL_15]] : i1
// FIROPT:           cond_br %[[VAL_16]], ^bb2, ^bb3
// FIROPT:         ^bb2:
// FIROPT:           %[[VAL_17:.*]] = fir.call @f2() : () -> i1
// FIROPT:           %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i1) -> i16
// FIROPT:           %[[VAL_19:.*]] = arith.addi %[[VAL_5]], %[[VAL_2]] : index
// FIROPT:           br ^bb1(%[[VAL_19]], %[[VAL_17]], %[[VAL_18]] : index, i1, i16)
// FIROPT:         ^bb3:
// FIROPT:           cond_br %[[VAL_6]], ^bb4, ^bb5
// FIROPT:         ^bb4:
// FIROPT:           %[[VAL_20:.*]] = arith.constant 0 : i32
// FIROPT:           %[[VAL_21:.*]] = fir.call @f4(%[[VAL_20]]) : (i32) -> i1
// FIROPT:           br ^bb5
// FIROPT:         ^bb5:
// FIROPT:           fir.call @f3(%[[VAL_7]]) : (i16) -> ()
// FIROPT:           return %[[VAL_5]] : index
// FIROPT:         }
