! RUN: bbc -emit-fir %s -o - | FileCheck %s

program nested_where
  integer :: a(3) = 0
  logical :: mask1(3) = (/ .true.,.false.,.true. /)
  logical :: mask2(3) = (/ .true.,.true.,.false. /)
  forall (i=1:3)
    where (mask1)
      where (mask2)
        a = 1
      end where
    endwhere
  end forall
end program nested_where

! CHECK-LABEL: func @_QQmain() {
! CHECK:         %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.heap<index>
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.heap<i8>
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.heap<index>
! CHECK:         %[[VAL_4:.*]] = fir.alloca !fir.heap<i8>
! CHECK:         %[[VAL_5:.*]] = fir.address_of(@_QEa) : !fir.ref<!fir.array<3xi32>>
! CHECK:         %[[VAL_6:.*]] = constant 3 : index
! CHECK:         %[[VAL_7:.*]] = fir.address_of(@_QEmask1) : !fir.ref<!fir.array<3x!fir.logical<4>>>
! CHECK:         %[[VAL_8:.*]] = constant 3 : index
! CHECK:         %[[VAL_9:.*]] = fir.address_of(@_QEmask2) : !fir.ref<!fir.array<3x!fir.logical<4>>>
! CHECK:         %[[VAL_10:.*]] = constant 3 : index
! CHECK:         %[[VAL_11:.*]] = fir.zero_bits !fir.heap<i8>
! CHECK:         fir.store %[[VAL_11]] to %[[VAL_4]] : !fir.ref<!fir.heap<i8>>
! CHECK:         %[[VAL_12:.*]] = fir.zero_bits !fir.heap<index>
! CHECK:         fir.store %[[VAL_12]] to %[[VAL_3]] : !fir.ref<!fir.heap<index>>
! CHECK:         %[[VAL_13:.*]] = fir.zero_bits !fir.heap<i8>
! CHECK:         fir.store %[[VAL_13]] to %[[VAL_2]] : !fir.ref<!fir.heap<i8>>
! CHECK:         %[[VAL_14:.*]] = fir.zero_bits !fir.heap<index>
! CHECK:         fir.store %[[VAL_14]] to %[[VAL_1]] : !fir.ref<!fir.heap<index>>
! CHECK:         %[[VAL_15:.*]] = constant 1 : i32
! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> index
! CHECK:         %[[VAL_17:.*]] = constant 3 : i32
! CHECK:         %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> index
! CHECK:         %[[VAL_19:.*]] = constant 1 : index
! CHECK:         %[[VAL_20:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_21:.*]] = fir.array_load %[[VAL_5]](%[[VAL_20]]) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.array<3xi32>
! CHECK:         %[[VAL_22:.*]] = fir.do_loop %[[VAL_23:.*]] = %[[VAL_16]] to %[[VAL_18]] step %[[VAL_19]] unordered iter_args(%[[VAL_24:.*]] = %[[VAL_21]]) -> (!fir.array<3xi32>) {
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_23]] : (index) -> i32
! CHECK:           fir.store %[[VAL_25]] to %[[VAL_0]] : !fir.ref<i32>
! CHECK:           %[[VAL_26:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_27:.*]] = fir.array_load %[[VAL_5]](%[[VAL_26]]) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.array<3xi32>
! CHECK:           %[[VAL_28:.*]] = constant 3 : i64
! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (i64) -> index
! CHECK:           %[[VAL_30:.*]] = constant 1 : i32
! CHECK:           %[[VAL_31:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_32:.*]] = fir.array_load %[[VAL_7]](%[[VAL_31]]) : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<3x!fir.logical<4>>
! CHECK:           %[[VAL_33:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.heap<i8>>
! CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
! CHECK:           %[[VAL_35:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_36:.*]] = fir.array_load %[[VAL_34]](%[[VAL_35]]) : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>) -> !fir.array<?xi8>
! CHECK:           %[[VAL_37:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.heap<i8>>
! CHECK:           %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
! CHECK:           %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (!fir.heap<!fir.array<?xi8>>) -> i64
! CHECK:           %[[VAL_40:.*]] = constant 0 : i64
! CHECK:           %[[VAL_41:.*]] = cmpi eq, %[[VAL_39]], %[[VAL_40]] : i64
! CHECK:           fir.if %[[VAL_41]] {
! CHECK:             %[[VAL_42:.*]] = fir.allocmem !fir.array<?xi8>, %[[VAL_8]] {uniq_name = ".lazy.mask"}
! CHECK:             %[[VAL_43:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.heap<i8>>) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             fir.store %[[VAL_42]] to %[[VAL_43]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             %[[VAL_44:.*]] = fir.allocmem !fir.array<1xindex> {uniq_name = ".lazy.mask.shape"}
! CHECK:             %[[VAL_45:.*]] = constant 0 : index
! CHECK:             %[[VAL_46:.*]] = fir.coordinate_of %[[VAL_44]], %[[VAL_45]] : (!fir.heap<!fir.array<1xindex>>, index) -> !fir.ref<index>
! CHECK:             fir.store %[[VAL_8]] to %[[VAL_46]] : !fir.ref<index>
! CHECK:             %[[VAL_47:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.heap<index>>) -> !fir.ref<!fir.heap<!fir.array<1xindex>>>
! CHECK:             fir.store %[[VAL_44]] to %[[VAL_47]] : !fir.ref<!fir.heap<!fir.array<1xindex>>>
! CHECK:           }
! CHECK:           %[[VAL_48:.*]] = constant 1 : index
! CHECK:           %[[VAL_49:.*]] = constant 0 : index
! CHECK:           %[[VAL_50:.*]] = subi %[[VAL_8]], %[[VAL_48]] : index
! CHECK:           %[[VAL_51:.*]] = fir.do_loop %[[VAL_52:.*]] = %[[VAL_49]] to %[[VAL_50]] step %[[VAL_48]] unordered iter_args(%[[VAL_53:.*]] = %[[VAL_36]]) -> (!fir.array<?xi8>) {
! CHECK:             %[[VAL_54:.*]] = fir.array_fetch %[[VAL_32]], %[[VAL_52]] : (!fir.array<3x!fir.logical<4>>, index) -> !fir.logical<4>
! CHECK:             %[[VAL_55:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.heap<i8>>
! CHECK:             %[[VAL_56:.*]] = fir.convert %[[VAL_55]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
! CHECK:             %[[VAL_57:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_58:.*]] = constant 1 : index
! CHECK:             %[[VAL_59:.*]] = addi %[[VAL_52]], %[[VAL_58]] : index
! CHECK:             %[[VAL_60:.*]] = fir.array_coor %[[VAL_56]](%[[VAL_57]]) %[[VAL_59]] : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
! CHECK:             %[[VAL_61:.*]] = fir.convert %[[VAL_54]] : (!fir.logical<4>) -> i8
! CHECK:             fir.store %[[VAL_61]] to %[[VAL_60]] : !fir.ref<i8>
! CHECK:             fir.result %[[VAL_53]] : !fir.array<?xi8>
! CHECK:           }
! CHECK:           %[[VAL_62:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.heap<i8>>
! CHECK:           %[[VAL_63:.*]] = fir.convert %[[VAL_62]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
! CHECK:           fir.array_merge_store %[[VAL_36]], %[[VAL_64:.*]] to %[[VAL_63]] : !fir.array<?xi8>, !fir.array<?xi8>, !fir.heap<!fir.array<?xi8>>
! CHECK:           %[[VAL_65:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_66:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_67:.*]] = fir.array_load %[[VAL_9]](%[[VAL_66]]) : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<3x!fir.logical<4>>
! CHECK:           %[[VAL_68:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<i8>>
! CHECK:           %[[VAL_69:.*]] = fir.convert %[[VAL_68]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
! CHECK:           %[[VAL_70:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_71:.*]] = fir.array_load %[[VAL_69]](%[[VAL_70]]) : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>) -> !fir.array<?xi8>
! CHECK:           %[[VAL_72:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<i8>>
! CHECK:           %[[VAL_73:.*]] = fir.convert %[[VAL_72]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
! CHECK:           %[[VAL_74:.*]] = fir.convert %[[VAL_73]] : (!fir.heap<!fir.array<?xi8>>) -> i64
! CHECK:           %[[VAL_75:.*]] = constant 0 : i64
! CHECK:           %[[VAL_76:.*]] = cmpi eq, %[[VAL_74]], %[[VAL_75]] : i64
! CHECK:           fir.if %[[VAL_76]] {
! CHECK:             %[[VAL_77:.*]] = fir.allocmem !fir.array<?xi8>, %[[VAL_10]] {uniq_name = ".lazy.mask"}
! CHECK:             %[[VAL_78:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.heap<i8>>) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             fir.store %[[VAL_77]] to %[[VAL_78]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             %[[VAL_79:.*]] = fir.allocmem !fir.array<1xindex> {uniq_name = ".lazy.mask.shape"}
! CHECK:             %[[VAL_80:.*]] = constant 0 : index
! CHECK:             %[[VAL_81:.*]] = fir.coordinate_of %[[VAL_79]], %[[VAL_80]] : (!fir.heap<!fir.array<1xindex>>, index) -> !fir.ref<index>
! CHECK:             fir.store %[[VAL_10]] to %[[VAL_81]] : !fir.ref<index>
! CHECK:             %[[VAL_82:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.heap<index>>) -> !fir.ref<!fir.heap<!fir.array<1xindex>>>
! CHECK:             fir.store %[[VAL_79]] to %[[VAL_82]] : !fir.ref<!fir.heap<!fir.array<1xindex>>>
! CHECK:           }
! CHECK:           %[[VAL_83:.*]] = constant 1 : index
! CHECK:           %[[VAL_84:.*]] = constant 0 : index
! CHECK:           %[[VAL_85:.*]] = subi %[[VAL_10]], %[[VAL_83]] : index
! CHECK:           %[[VAL_86:.*]] = fir.do_loop %[[VAL_87:.*]] = %[[VAL_84]] to %[[VAL_85]] step %[[VAL_83]] unordered iter_args(%[[VAL_88:.*]] = %[[VAL_71]]) -> (!fir.array<?xi8>) {
! CHECK:             %[[VAL_89:.*]] = fir.array_fetch %[[VAL_67]], %[[VAL_87]] : (!fir.array<3x!fir.logical<4>>, index) -> !fir.logical<4>
! CHECK:             %[[VAL_90:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<i8>>
! CHECK:             %[[VAL_91:.*]] = fir.convert %[[VAL_90]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
! CHECK:             %[[VAL_92:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_93:.*]] = constant 1 : index
! CHECK:             %[[VAL_94:.*]] = addi %[[VAL_87]], %[[VAL_93]] : index
! CHECK:             %[[VAL_95:.*]] = fir.array_coor %[[VAL_91]](%[[VAL_92]]) %[[VAL_94]] : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
! CHECK:             %[[VAL_96:.*]] = fir.convert %[[VAL_89]] : (!fir.logical<4>) -> i8
! CHECK:             fir.store %[[VAL_96]] to %[[VAL_95]] : !fir.ref<i8>
! CHECK:             fir.result %[[VAL_88]] : !fir.array<?xi8>
! CHECK:           }
! CHECK:           %[[VAL_97:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<i8>>
! CHECK:           %[[VAL_98:.*]] = fir.convert %[[VAL_97]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
! CHECK:           fir.array_merge_store %[[VAL_71]], %[[VAL_99:.*]] to %[[VAL_98]] : !fir.array<?xi8>, !fir.array<?xi8>, !fir.heap<!fir.array<?xi8>>
! CHECK:           %[[VAL_100:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_101:.*]] = constant 1 : index
! CHECK:           %[[VAL_102:.*]] = constant 0 : index
! CHECK:           %[[VAL_103:.*]] = subi %[[VAL_29]], %[[VAL_101]] : index
! CHECK:           %[[VAL_104:.*]] = fir.do_loop %[[VAL_105:.*]] = %[[VAL_102]] to %[[VAL_103]] step %[[VAL_101]] unordered iter_args(%[[VAL_106:.*]] = %[[VAL_27]]) -> (!fir.array<3xi32>) {
! CHECK:             %[[VAL_107:.*]] = constant 1 : index
! CHECK:             %[[VAL_108:.*]] = addi %[[VAL_105]], %[[VAL_107]] : index
! CHECK:             %[[VAL_109:.*]] = fir.array_coor %[[VAL_63]](%[[VAL_65]]) %[[VAL_108]] : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
! CHECK:             %[[VAL_110:.*]] = fir.load %[[VAL_109]] : !fir.ref<i8>
! CHECK:             %[[VAL_111:.*]] = fir.convert %[[VAL_110]] : (i8) -> i1
! CHECK:             %[[VAL_112:.*]] = fir.if %[[VAL_111]] -> (!fir.array<3xi32>) {
! CHECK:               %[[VAL_113:.*]] = constant 1 : index
! CHECK:               %[[VAL_114:.*]] = addi %[[VAL_105]], %[[VAL_113]] : index
! CHECK:               %[[VAL_115:.*]] = fir.array_coor %[[VAL_98]](%[[VAL_100]]) %[[VAL_114]] : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
! CHECK:               %[[VAL_116:.*]] = fir.load %[[VAL_115]] : !fir.ref<i8>
! CHECK:               %[[VAL_117:.*]] = fir.convert %[[VAL_116]] : (i8) -> i1
! CHECK:               %[[VAL_118:.*]] = fir.if %[[VAL_117]] -> (!fir.array<3xi32>) {
! CHECK:                 %[[VAL_119:.*]] = fir.array_update %[[VAL_106]], %[[VAL_30]], %[[VAL_105]] : (!fir.array<3xi32>, i32, index) -> !fir.array<3xi32>
! CHECK:                 fir.result %[[VAL_119]] : !fir.array<3xi32>
! CHECK:               } else {
! CHECK:                 fir.result %[[VAL_106]] : !fir.array<3xi32>
! CHECK:               }
! CHECK:               fir.result %[[VAL_120:.*]] : !fir.array<3xi32>
! CHECK:             } else {
! CHECK:               fir.result %[[VAL_106]] : !fir.array<3xi32>
! CHECK:             }
! CHECK:             fir.result %[[VAL_121:.*]] : !fir.array<3xi32>
! CHECK:           }
! CHECK:           fir.array_merge_store %[[VAL_27]], %[[VAL_122:.*]] to %[[VAL_5]] : !fir.array<3xi32>, !fir.array<3xi32>, !fir.ref<!fir.array<3xi32>>
! CHECK:           fir.result %[[VAL_24]] : !fir.array<3xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_21]], %[[VAL_123:.*]] to %[[VAL_5]] : !fir.array<3xi32>, !fir.array<3xi32>, !fir.ref<!fir.array<3xi32>>
! CHECK:         %[[VAL_124:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<i8>>
! CHECK:         %[[VAL_125:.*]] = fir.convert %[[VAL_124]] : (!fir.heap<i8>) -> i64
! CHECK:         %[[VAL_126:.*]] = constant 0 : i64
! CHECK:         %[[VAL_127:.*]] = cmpi ne, %[[VAL_125]], %[[VAL_126]] : i64
! CHECK:         fir.if %[[VAL_127]] {
! CHECK:           fir.freemem %[[VAL_124]] : !fir.heap<i8>
! CHECK:         }
! CHECK:         %[[VAL_128:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.heap<i8>>
! CHECK:         %[[VAL_129:.*]] = fir.convert %[[VAL_128]] : (!fir.heap<i8>) -> i64
! CHECK:         %[[VAL_130:.*]] = constant 0 : i64
! CHECK:         %[[VAL_131:.*]] = cmpi ne, %[[VAL_129]], %[[VAL_130]] : i64
! CHECK:         fir.if %[[VAL_131]] {
! CHECK:           fir.freemem %[[VAL_128]] : !fir.heap<i8>
! CHECK:         }
! CHECK:         return
! CHECK:       }
