! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: @_QQmain
! CHECK:   %[[V_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QEi"}
! CHECK:   %[[V_1:.*]] = constant 1 : i32
! CHECK:   %[[V_2:.*]] = fir.convert %[[V_1]] : (i32) -> index
! CHECK:   %[[V_3:.*]] = constant 5 : i32
! CHECK:   %[[V_4:.*]] = fir.convert %[[V_3]] : (i32) -> index
! CHECK:   %[[V_5:.*]] = constant 1 : index
! CHECK:   %[[V_6:.*]] = fir.do_loop %[[V_7:.*]] = %[[V_2]] to %[[V_4]] step %[[V_5]] -> index {
! CHECK:     %[[V_8:.*]] = fir.convert %[[V_7]] : (index) -> i32
! CHECK:     fir.store %[[V_8]] to %[[V_0]] : !fir.ref<i32>
! CHECK:     %[[V_9:.*]] = fir.load %[[V_0]] : !fir.ref<i32>
! CHECK:     %[[V_10:.*]] = constant 1 : i32
! CHECK:     %[[V_11:.*]] = cmpi sle, %[[V_9]], %[[V_10]] : i32
! CHECK:     %[[V_12:.*]] = fir.load %[[V_0]] : !fir.ref<i32>
! CHECK:     %[[V_13:.*]] = constant 5 : i32
! CHECK:     %[[V_14:.*]] = cmpi sge, %[[V_12]], %[[V_13]] : i32
! CHECK:     %[[V_15:.*]] = or %[[V_11]], %[[V_14]] : i1
! CHECK:     %[[V_16:.*]] = constant true
! CHECK:     %[[V_17:.*]] = xor %[[V_15]], %[[V_16]] : i1
! CHECK:     fir.if %[[V_17]] {
! CHECK:     }
! CHECK:     %[[V_27:.*]] = addi %[[V_7]], %[[V_5]] : index
! CHECK:     fir.result %[[V_27]] : index
! CHECK:   }
! CHECK:   %[[V_28:.*]] = fir.convert %[[V_6]] : (index) -> i32
! CHECK:   fir.store %[[V_28]] to %[[V_0]] : !fir.ref<i32>
! CHECK:   %[[V_29:.*]] = constant 1 : i32
! CHECK:   %[[V_30:.*]] = fir.convert %[[V_29]] : (i32) -> index
! CHECK:   %[[V_31:.*]] = constant 5 : i32
! CHECK:   %[[V_32:.*]] = fir.convert %[[V_31]] : (i32) -> index
! CHECK:   %[[V_33:.*]] = constant 1 : index
! CHECK:   %[[V_34:.*]] = fir.do_loop %[[V_35:.*]] = %[[V_30]] to %[[V_32]] step %[[V_33]] -> index {
! CHECK:     %[[V_36:.*]] = fir.convert %[[V_35]] : (index) -> i32
! CHECK:     fir.store %[[V_36]] to %[[V_0]] : !fir.ref<i32>
! CHECK:     %[[V_37:.*]] = fir.load %[[V_0]] : !fir.ref<i32>
! CHECK:     %[[V_38:.*]] = constant 1 : i32
! CHECK:     %[[V_39:.*]] = cmpi sle, %[[V_37]], %[[V_38]] : i32
! CHECK:     %[[V_40:.*]] = fir.load %[[V_0]] : !fir.ref<i32>
! CHECK:     %[[V_41:.*]] = constant 5 : i32
! CHECK:     %[[V_42:.*]] = cmpi sge, %[[V_40]], %[[V_41]] : i32
! CHECK:     %[[V_43:.*]] = or %[[V_39]], %[[V_42]] : i1
! CHECK:     %[[V_44:.*]] = constant true
! CHECK:     %[[V_45:.*]] = xor %[[V_43]], %[[V_44]] : i1
! CHECK:     fir.if %[[V_45]] {
! CHECK:     }
! CHECK:     %[[V_55:.*]] = addi %[[V_35]], %[[V_33]] : index
! CHECK:     fir.result %[[V_55]] : index
! CHECK:   }
! CHECK:   %[[V_56:.*]] = fir.convert %[[V_34]] : (index) -> i32
! CHECK:   fir.store %[[V_56]] to %[[V_0]] : !fir.ref<i32>
! CHECK:   %[[V_57:.*]] = constant 1 : i32
! CHECK:   %[[V_58:.*]] = fir.convert %[[V_57]] : (i32) -> index
! CHECK:   %[[V_59:.*]] = constant 5 : i32
! CHECK:   %[[V_60:.*]] = fir.convert %[[V_59]] : (i32) -> index
! CHECK:   %[[V_61:.*]] = constant 1 : index
! CHECK:   %[[V_62:.*]] = fir.do_loop %[[V_63:.*]] = %[[V_58]] to %[[V_60]] step %[[V_61]] -> index {
! CHECK:     %[[V_64:.*]] = fir.convert %[[V_63]] : (index) -> i32
! CHECK:     fir.store %[[V_64]] to %[[V_0]] : !fir.ref<i32>
! CHECK:     %[[V_65:.*]] = fir.load %[[V_0]] : !fir.ref<i32>
! CHECK:     %[[V_66:.*]] = constant 1 : i32
! CHECK:     %[[V_67:.*]] = cmpi sle, %[[V_65]], %[[V_66]] : i32
! CHECK:     %[[V_68:.*]] = fir.load %[[V_0]] : !fir.ref<i32>
! CHECK:     %[[V_69:.*]] = constant 5 : i32
! CHECK:     %[[V_70:.*]] = cmpi sge, %[[V_68]], %[[V_69]] : i32
! CHECK:     %[[V_71:.*]] = or %[[V_67]], %[[V_70]] : i1
! CHECK:     %[[V_72:.*]] = constant true
! CHECK:     %[[V_73:.*]] = xor %[[V_71]], %[[V_72]] : i1
! CHECK:     fir.if %[[V_73]] {
! CHECK:     }
! CHECK:     %[[V_83:.*]] = addi %[[V_63]], %[[V_61]] : index
! CHECK:     fir.result %[[V_83]] : index
! CHECK:   }
! CHECK:   %[[V_84:.*]] = fir.convert %[[V_62]] : (index) -> i32
! CHECK:   fir.store %[[V_84]] to %[[V_0]] : !fir.ref<i32>
! CHECK:   %[[V_85:.*]] = constant 1 : i32
! CHECK:   %[[V_86:.*]] = fir.convert %[[V_85]] : (i32) -> index
! CHECK:   %[[V_87:.*]] = constant 5 : i32
! CHECK:   %[[V_88:.*]] = fir.convert %[[V_87]] : (i32) -> index
! CHECK:   %[[V_89:.*]] = constant 1 : index
! CHECK:   %[[V_90:.*]] = fir.do_loop %[[V_91:.*]] = %[[V_86]] to %[[V_88]] step %[[V_89]] -> index {
! CHECK:     %[[V_92:.*]] = fir.convert %[[V_91]] : (index) -> i32
! CHECK:     fir.store %[[V_92]] to %[[V_0]] : !fir.ref<i32>
! CHECK:     %[[V_93:.*]] = fir.load %[[V_0]] : !fir.ref<i32>
! CHECK:     %[[V_94:.*]] = constant 1 : i32
! CHECK:     %[[V_95:.*]] = cmpi sle, %[[V_93]], %[[V_94]] : i32
! CHECK:     %[[V_96:.*]] = fir.load %[[V_0]] : !fir.ref<i32>
! CHECK:     %[[V_97:.*]] = constant 5 : i32
! CHECK:     %[[V_98:.*]] = cmpi sge, %[[V_96]], %[[V_97]] : i32
! CHECK:     %[[V_99:.*]] = or %[[V_95]], %[[V_98]] : i1
! CHECK:     %[[V_100:.*]] = constant true
! CHECK:     %[[V_101:.*]] = xor %[[V_99]], %[[V_100]] : i1
! CHECK:     fir.if %[[V_101]] {
! CHECK:       %[[V_102:.*]] = fir.load %[[V_0]] : !fir.ref<i32>
! CHECK:       %[[V_103:.*]] = constant 3 : i32
! CHECK:       %[[V_104:.*]] = cmpi eq, %[[V_102]], %[[V_103]] : i32
! CHECK:       %[[V_105:.*]] = constant true
! CHECK:       %[[V_106:.*]] = xor %[[V_104]], %[[V_105]] : i1
! CHECK:       fir.if %[[V_106]] {
! CHECK:       }
! CHECK:     } else {
! CHECK:     }
! CHECK:     %[[V_116:.*]] = addi %[[V_91]], %[[V_89]] : index
! CHECK:     fir.result %[[V_116]] : index
! CHECK:   }
! CHECK:   %[[V_117:.*]] = fir.convert %[[V_90]] : (index) -> i32
! CHECK:   fir.store %[[V_117]] to %[[V_0]] : !fir.ref<i32>
! CHECK:   %[[V_118:.*]] = constant 1 : i32
! CHECK:   %[[V_119:.*]] = fir.convert %[[V_118]] : (i32) -> index
! CHECK:   %[[V_120:.*]] = constant 5 : i32
! CHECK:   %[[V_121:.*]] = fir.convert %[[V_120]] : (i32) -> index
! CHECK:   %[[V_122:.*]] = constant 1 : index
! CHECK:   %[[V_123:.*]] = fir.do_loop %[[V_124:.*]] = %[[V_119]] to %[[V_121]] step %[[V_122]] -> index {
! CHECK:     %[[V_125:.*]] = fir.convert %[[V_124]] : (index) -> i32
! CHECK:     fir.store %[[V_125]] to %[[V_0]] : !fir.ref<i32>
! CHECK:     %[[V_126:.*]] = fir.load %[[V_0]] : !fir.ref<i32>
! CHECK:     %[[V_127:.*]] = constant 3 : i32
! CHECK:     %[[V_128:.*]] = cmpi eq, %[[V_126]], %[[V_127]] : i32
! CHECK:     %[[V_129:.*]] = constant true
! CHECK:     %[[V_130:.*]] = xor %[[V_128]], %[[V_129]] : i1
! CHECK:     fir.if %[[V_130]] {
! CHECK:       %[[V_131:.*]] = fir.load %[[V_0]] : !fir.ref<i32>
! CHECK:       %[[V_132:.*]] = constant 1 : i32
! CHECK:       %[[V_133:.*]] = cmpi sle, %[[V_131]], %[[V_132]] : i32
! CHECK:       %[[V_134:.*]] = fir.load %[[V_0]] : !fir.ref<i32>
! CHECK:       %[[V_135:.*]] = constant 5 : i32
! CHECK:       %[[V_136:.*]] = cmpi sge, %[[V_134]], %[[V_135]] : i32
! CHECK:       %[[V_137:.*]] = or %[[V_133]], %[[V_136]] : i1
! CHECK:       %[[V_138:.*]] = constant true
! CHECK:       %[[V_139:.*]] = xor %[[V_137]], %[[V_138]] : i1
! CHECK:       fir.if %[[V_139]] {
! CHECK:       }
! CHECK:     } else {
! CHECK:     }
! CHECK:     %[[V_149:.*]] = addi %[[V_124]], %[[V_122]] : index
! CHECK:     fir.result %[[V_149]] : index
! CHECK:   }
! CHECK:   %[[V_150:.*]] = fir.convert %[[V_123]] : (index) -> i32
! CHECK:   fir.store %[[V_150]] to %[[V_0]] : !fir.ref<i32>
! CHECK:   return
! CHECK: }
program pp
  integer :: i
  do i = 1, 5
     if (i <= 1 .or. i >= 5) goto 1
     print*, i
1 end do

! print*
  do i = 1, 5
     if (i <= 1 .or. i >= 5) cycle
     print*, i
  end do

! print*
  do i = 1, 5
     if (i <= 1 .or. i >= 5) cycle
     print*, i
2 end do

! print*
  abc: do i = 1, 5
     if (i <= 1 .or. i >= 5) cycle abc
     if (i == 3) goto 3
     print*, i
3 end do abc

! print*
  do i = 1, 5
     if (i == 3) goto 4
     if (i <= 1 .or. i >= 5) cycle
     print*, i
4 end do
end
