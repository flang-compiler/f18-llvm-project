! Test forall lowering

! RUN: bbc -emit-fir %s -o - | FileCheck %s

subroutine test_forall_with_slice(i1,i2)
  interface
     pure integer function f(i)
       integer i
       intent(in) i
     end function f
  end interface
  type t
     !integer :: arr(5:15)
     integer :: arr(11)
  end type t
  type(t) :: a(10,10)

  forall (i=1:5, j=1:10)
     a(i,j)%arr(i:i1:i2) = f(i)
  end forall
end subroutine test_forall_with_slice

! CHECK-LABEL: func @_QPtest_forall_with_slice(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<i32>, %[[VAL_1:.*]]: !fir.ref<i32>) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_6:.*]] = fir.alloca !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>> {bindc_name = "a", uniq_name = "_QFtest_forall_with_sliceEa"}
! CHECK:         %[[VAL_7:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
! CHECK:         %[[VAL_9:.*]] = arith.constant 5 : i32
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK:         %[[VAL_11:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_12:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i32) -> index
! CHECK:         %[[VAL_14:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i32) -> index
! CHECK:         %[[VAL_16:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_17:.*]] = fir.shape %[[VAL_4]], %[[VAL_5]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_18:.*]] = fir.array_load %[[VAL_6]](%[[VAL_17]]) : (!fir.ref<!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>>, !fir.shape<2>) -> !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
! CHECK:         %[[VAL_19:.*]] = fir.do_loop %[[VAL_20:.*]] = %[[VAL_8]] to %[[VAL_10]] step %[[VAL_11]] unordered iter_args(%[[VAL_21:.*]] = %[[VAL_18]]) -> (!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>) {
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_20]] : (index) -> i32
! CHECK:           fir.store %[[VAL_22]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_23:.*]] = fir.do_loop %[[VAL_24:.*]] = %[[VAL_13]] to %[[VAL_15]] step %[[VAL_16]] unordered iter_args(%[[VAL_25:.*]] = %[[VAL_21]]) -> (!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>) {
! CHECK:             %[[VAL_26:.*]] = fir.convert %[[VAL_24]] : (index) -> i32
! CHECK:             fir.store %[[VAL_26]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:             %[[VAL_27:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_28:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:             %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (i32) -> i64
! CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i64) -> index
! CHECK:             %[[VAL_31:.*]] = arith.subi %[[VAL_30]], %[[VAL_27]] : index
! CHECK:             %[[VAL_32:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:             %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (i32) -> i64
! CHECK:             %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i64) -> index
! CHECK:             %[[VAL_35:.*]] = arith.subi %[[VAL_34]], %[[VAL_27]] : index
! CHECK:             %[[VAL_36:.*]] = fir.field_index arr, !fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>
! CHECK:             %[[VAL_37:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:             %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (i32) -> i64
! CHECK:             %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (i64) -> index
! CHECK:             %[[VAL_40:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:             %[[VAL_41:.*]] = fir.convert %[[VAL_40]] : (i32) -> i64
! CHECK:             %[[VAL_42:.*]] = fir.convert %[[VAL_41]] : (i64) -> index
! CHECK:             %[[VAL_43:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:             %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (i32) -> i64
! CHECK:             %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (i64) -> index
! CHECK:             %[[VAL_46:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_47:.*]] = arith.subi %[[VAL_45]], %[[VAL_39]] : index
! CHECK:             %[[VAL_48:.*]] = arith.addi %[[VAL_47]], %[[VAL_42]] : index
! CHECK:             %[[VAL_49:.*]] = arith.divsi %[[VAL_48]], %[[VAL_42]] : index
! CHECK:             %[[VAL_50:.*]] = arith.cmpi sgt, %[[VAL_49]], %[[VAL_46]] : index
! CHECK:             %[[VAL_51:.*]] = select %[[VAL_50]], %[[VAL_49]], %[[VAL_46]] : index
! CHECK:             %[[VAL_52:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_53:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_54:.*]] = arith.subi %[[VAL_51]], %[[VAL_52]] : index
! CHECK:             %[[VAL_55:.*]] = fir.do_loop %[[VAL_56:.*]] = %[[VAL_53]] to %[[VAL_54]] step %[[VAL_52]] unordered iter_args(%[[VAL_57:.*]] = %[[VAL_25]]) -> (!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>) {
! CHECK:               %[[VAL_58:.*]] = fir.call @_QPf(%[[VAL_3]]) : (!fir.ref<i32>) -> i32
! CHECK:               %[[VAL_59:.*]] = arith.constant 0 : index
! CHECK:               %[[VAL_60:.*]] = arith.muli %[[VAL_56]], %[[VAL_42]] : index
! CHECK:               %[[VAL_61:.*]] = arith.addi %[[VAL_59]], %[[VAL_60]] : index
! CHECK:               %[[VAL_62:.*]] = fir.array_update %[[VAL_57]], %[[VAL_58]], %[[VAL_31]], %[[VAL_35]], %[[VAL_36]], %[[VAL_61]] : (!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>, i32, index, index, !fir.field, index) -> !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
! CHECK:               fir.result %[[VAL_62]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
! CHECK:             }
! CHECK:             fir.result %[[VAL_63:.*]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_64:.*]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_18]], %[[VAL_65:.*]] to %[[VAL_6]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>, !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>, !fir.ref<!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>>
! CHECK:         return
! CHECK:       }

