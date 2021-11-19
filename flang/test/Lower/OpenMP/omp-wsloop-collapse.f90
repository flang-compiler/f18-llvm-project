! This test checks lowering of OpenMP DO Directive(Worksharing) with collapse.

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   tco --disable-llvm --print-ir-after=fir-to-llvm-ir 2>&1 | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   tco | FileCheck %s --check-prefix=LLVMIR

program wsloop_collapse
! LLVMIRDialect:   llvm.func @_QQmain() {
  integer :: i, j, k
  integer :: a, b, c
  integer :: x
! FIRDialect:         %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "a", uniq_name = "_QEa"}
! FIRDialect:         %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "b", uniq_name = "_QEb"}
! FIRDialect:         %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "c", uniq_name = "_QEc"}
! FIRDialect:         %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QEi"}
! FIRDialect:         %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QEj"}
! FIRDialect:         %[[VAL_5:.*]] = fir.alloca i32 {bindc_name = "k", uniq_name = "_QEk"}
! FIRDialect:         %[[VAL_6:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QEx"}
! LLVMIRDialect-DAG:       %[[VAL_4:.*]] = llvm.mlir.constant(1 : i32) : i32
! LLVMIRDialect-DAG:       %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : i32
! LLVMIRDialect-DAG:       %[[VAL_2:.*]] = llvm.mlir.constant(5 : i32) : i32
! LLVMIRDialect-DAG:       %[[VAL_1:.*]] = llvm.mlir.constant(2 : i32) : i32
! LLVMIRDialect-DAG:       %[[VAL_0:.*]] = llvm.mlir.constant(3 : i32) : i32
! LLVMIRDialect-DAG:       %[[VAL_7:.*]] = llvm.mlir.constant(1 : i64) : i64
  a=3
! FIRDialect:         %[[VAL_7:.*]] = arith.constant 3 : i32
! FIRDialect:         fir.store %[[VAL_7]] to %[[VAL_0]] : !fir.ref<i32>
  b=2
! FIRDialect:         %[[VAL_8:.*]] = arith.constant 2 : i32
! FIRDialect:         fir.store %[[VAL_8]] to %[[VAL_1]] : !fir.ref<i32>
  c=5
! FIRDialect:         %[[VAL_9:.*]] = arith.constant 5 : i32
! FIRDialect:         fir.store %[[VAL_9]] to %[[VAL_2]] : !fir.ref<i32>
  x=0
! FIRDialect:         %[[VAL_10:.*]] = arith.constant 0 : i32
! FIRDialect:         fir.store %[[VAL_10]] to %[[VAL_6]] : !fir.ref<i32>
! LLVMIRDialect:           %[[VAL_8:.*]] = llvm.alloca %[[VAL_7]] x i32 {{{.*}} uniq_name = "_QEa"} : (i64) -> !llvm.ptr<i32>
! LLVMIRDialect:           %[[VAL_9:.*]] = llvm.mlir.constant(1 : i64) : i64
! LLVMIRDialect:           %[[VAL_10:.*]] = llvm.alloca %[[VAL_9]] x i32 {{{.*}} uniq_name = "_QEb"} : (i64) -> !llvm.ptr<i32>
! LLVMIRDialect:           %[[VAL_11:.*]] = llvm.mlir.constant(1 : i64) : i64
! LLVMIRDialect:           %[[VAL_12:.*]] = llvm.alloca %[[VAL_11]] x i32 {{{.*}} uniq_name = "_QEc"} : (i64) -> !llvm.ptr<i32>
! LLVMIRDialect:           %[[VAL_13:.*]] = llvm.mlir.constant(1 : i64) : i64
! LLVMIRDialect:           %[[VAL_14:.*]] = llvm.alloca %[[VAL_13]] x i32 {{{.*}} uniq_name = "_QEi"} : (i64) -> !llvm.ptr<i32>
! LLVMIRDialect:           %[[VAL_15:.*]] = llvm.mlir.constant(1 : i64) : i64
! LLVMIRDialect:           %[[VAL_16:.*]] = llvm.alloca %[[VAL_15]] x i32 {{{.*}} uniq_name = "_QEj"} : (i64) -> !llvm.ptr<i32>
! LLVMIRDialect:           %[[VAL_17:.*]] = llvm.mlir.constant(1 : i64) : i64
! LLVMIRDialect:           %[[VAL_18:.*]] = llvm.alloca %[[VAL_17]] x i32 {{{.*}} uniq_name = "_QEk"} : (i64) -> !llvm.ptr<i32>
! LLVMIRDialect:           %[[VAL_19:.*]] = llvm.mlir.constant(1 : i64) : i64
! LLVMIRDialect:           %[[VAL_20:.*]] = llvm.alloca %[[VAL_19]] x i32 {{{.*}} uniq_name = "_QEx"} : (i64) -> !llvm.ptr<i32>
! LLVMIRDialect:           llvm.store %[[VAL_0]], %[[VAL_8]] : !llvm.ptr<i32>
! LLVMIRDialect:           llvm.store %[[VAL_1]], %[[VAL_10]] : !llvm.ptr<i32>
! LLVMIRDialect:           llvm.store %[[VAL_2]], %[[VAL_12]] : !llvm.ptr<i32>
! LLVMIRDialect:           llvm.store %[[VAL_3]], %[[VAL_20]] : !llvm.ptr<i32>
! LLVMIR:         %[[VAL_0:.*]] = alloca i32, i64 1, align 4, !dbg !7
! LLVMIR:         %[[VAL_1:.*]] = alloca i32, i64 1, align 4, !dbg !9
! LLVMIR:         %[[VAL_2:.*]] = alloca i32, i64 1, align 4, !dbg !10
! LLVMIR:         %[[VAL_3:.*]] = alloca i32, i64 1, align 4, !dbg !11
! LLVMIR:         %[[VAL_4:.*]] = alloca i32, i64 1, align 4, !dbg !12
! LLVMIR:         %[[VAL_5:.*]] = alloca i32, i64 1, align 4, !dbg !13
! LLVMIR:         %[[VAL_6:.*]] = alloca i32, i64 1, align 4, !dbg !14
! LLVMIR:         store i32 3, i32* %[[VAL_0]], align 4, !dbg !15
! LLVMIR:         store i32 2, i32* %[[VAL_1]], align 4, !dbg !16
! LLVMIR:         store i32 5, i32* %[[VAL_2]], align 4, !dbg !17
! LLVMIR:         store i32 0, i32* %[[VAL_6]], align 4, !dbg !18
! LLVMIR:         %[[VAL_7:.*]] = call i32 @__kmpc_global_thread_num(%[[VAL_8:.*]]* @1), !dbg !19
! LLVMIR:         br label %[[VAL_9:.*]]

  !$omp parallel do collapse(3)
! FIRDialect:   omp.parallel {
! LLVMIRDialect:           omp.parallel {
! LLVMIRDialect:             %[[VAL_21:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr<i32>
! LLVMIRDialect:             %[[VAL_22:.*]] = llvm.load %[[VAL_10]] : !llvm.ptr<i32>
! LLVMIRDialect:             %[[VAL_23:.*]] = llvm.load %[[VAL_12]] : !llvm.ptr<i32>
! FIRDialect:           %[[VAL_20:.*]] = arith.constant 1 : i32
! FIRDialect:           %[[VAL_21:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! FIRDialect:           %[[VAL_22:.*]] = arith.constant 1 : i32
! FIRDialect:           %[[VAL_23:.*]] = arith.constant 1 : i32
! FIRDialect:           %[[VAL_24:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! FIRDialect:           %[[VAL_25:.*]] = arith.constant 1 : i32
! FIRDialect:           %[[VAL_26:.*]] = arith.constant 1 : i32
! FIRDialect:           %[[VAL_27:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! FIRDialect:           %[[VAL_28:.*]] = arith.constant 1 : i32
! LLVMIR:       omp_parallel:                                     ; preds = %[[VAL_10:.*]]
! LLVMIR:         call void (%[[VAL_8]]*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%[[VAL_8]]* @1, i32 4, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i32*, i32*, i32*, i32*)* @_QQmain..omp_par to void (i32*, i32*, ...)*), i32* %[[VAL_0]], i32* %[[VAL_1]], i32* %[[VAL_2]], i32* %[[VAL_6]]), !dbg !20
! LLVMIR:         br label %[[VAL_11:.*]]
! LLVMIR:       omp.par.outlined.exit:                            ; preds = %[[VAL_9]]
! LLVMIR:         br label %[[VAL_12:.*]]
! LLVMIR:       omp.par.exit.split:                               ; preds = %[[VAL_11]]
! LLVMIR:         ret void, !dbg !21
! LLVMIR:       omp.par.entry:
! LLVMIR:         %[[VAL_13:.*]] = alloca i32, align 4
! LLVMIR:         %[[VAL_14:.*]] = load i32, i32* %[[VAL_15:.*]], align 4
! LLVMIR:         store i32 %[[VAL_14]], i32* %[[VAL_13]], align 4
! LLVMIR:         %[[VAL_16:.*]] = load i32, i32* %[[VAL_13]], align 4
! LLVMIR:         %[[VAL_17:.*]] = alloca i32, align 4
! LLVMIR:         %[[VAL_18:.*]] = alloca i32, align 4
! LLVMIR:         %[[VAL_19:.*]] = alloca i32, align 4
! LLVMIR:         %[[VAL_20:.*]] = alloca i32, align 4
! LLVMIR:         br label %[[VAL_21:.*]]
! LLVMIR:       omp.par.region:                                   ; preds = %[[VAL_23:.*]]
! LLVMIR:         br label %[[VAL_24:.*]]
! LLVMIR:       omp.par.region1:                                  ; preds = %[[VAL_21]]
! LLVMIR:         %[[VAL_25:.*]] = load i32, i32* %[[VAL_26:.*]], align 4, !dbg !23
! LLVMIR:         %[[VAL_27:.*]] = load i32, i32* %[[VAL_28:.*]], align 4, !dbg !24
! LLVMIR:         %[[VAL_29:.*]] = load i32, i32* %[[VAL_30:.*]], align 4, !dbg !25
! LLVMIR:         %[[VAL_31:.*]] = select i1 false, i32 %[[VAL_25]], i32 1, !dbg !26
! LLVMIR:         %[[VAL_32:.*]] = select i1 false, i32 1, i32 %[[VAL_25]], !dbg !26
! LLVMIR:         %[[VAL_33:.*]] = sub nsw i32 %[[VAL_32]], %[[VAL_31]], !dbg !26
! LLVMIR:         %[[VAL_34:.*]] = icmp slt i32 %[[VAL_32]], %[[VAL_31]], !dbg !26
! LLVMIR:         %[[VAL_35:.*]] = udiv i32 %[[VAL_33]], 1, !dbg !26
! LLVMIR:         %[[VAL_36:.*]] = add i32 %[[VAL_35]], 1, !dbg !26
! LLVMIR:         %[[VAL_37:.*]] = select i1 %[[VAL_34]], i32 0, i32 %[[VAL_36]], !dbg !26
! LLVMIR:         br label %[[VAL_38:.*]]
! LLVMIR:       omp_loop.preheader:                               ; preds = %[[VAL_24]]
! LLVMIR:         %[[VAL_39:.*]] = select i1 false, i32 %[[VAL_27]], i32 1, !dbg !26
! LLVMIR:         %[[VAL_40:.*]] = select i1 false, i32 1, i32 %[[VAL_27]], !dbg !26
! LLVMIR:         %[[VAL_41:.*]] = sub nsw i32 %[[VAL_40]], %[[VAL_39]], !dbg !26
! LLVMIR:         %[[VAL_42:.*]] = icmp slt i32 %[[VAL_40]], %[[VAL_39]], !dbg !26
! LLVMIR:         %[[VAL_43:.*]] = udiv i32 %[[VAL_41]], 1, !dbg !26
! LLVMIR:         %[[VAL_44:.*]] = add i32 %[[VAL_43]], 1, !dbg !26
! LLVMIR:         %[[VAL_45:.*]] = select i1 %[[VAL_42]], i32 0, i32 %[[VAL_44]], !dbg !26
! LLVMIR:         %[[VAL_46:.*]] = select i1 false, i32 %[[VAL_29]], i32 1, !dbg !26
! LLVMIR:         %[[VAL_47:.*]] = select i1 false, i32 1, i32 %[[VAL_29]], !dbg !26
! LLVMIR:         %[[VAL_48:.*]] = sub nsw i32 %[[VAL_47]], %[[VAL_46]], !dbg !26
! LLVMIR:         %[[VAL_49:.*]] = icmp slt i32 %[[VAL_47]], %[[VAL_46]], !dbg !26
! LLVMIR:         %[[VAL_50:.*]] = udiv i32 %[[VAL_48]], 1, !dbg !26
! LLVMIR:         %[[VAL_51:.*]] = add i32 %[[VAL_50]], 1, !dbg !26
! LLVMIR:         %[[VAL_52:.*]] = select i1 %[[VAL_49]], i32 0, i32 %[[VAL_51]], !dbg !26
! LLVMIR:         %[[VAL_53:.*]] = mul nuw i32 %[[VAL_37]], %[[VAL_45]]
! LLVMIR:         %[[VAL_54:.*]] = mul nuw i32 %[[VAL_53]], %[[VAL_52]]
! LLVMIR:         br label %[[VAL_55:.*]]
! LLVMIR:       omp_collapsed.preheader:                          ; preds = %[[VAL_38]]
! LLVMIR:         store i32 0, i32* %[[VAL_18]], align 4, !dbg !26
! LLVMIR:         %[[VAL_56:.*]] = sub i32 %[[VAL_54]], 1, !dbg !26
! LLVMIR:         store i32 %[[VAL_56]], i32* %[[VAL_19]], align 4, !dbg !26
! LLVMIR:         store i32 1, i32* %[[VAL_20]], align 4, !dbg !26
! LLVMIR:         %[[VAL_57:.*]] = call i32 @__kmpc_global_thread_num(%[[VAL_58:.*]]* @3), !dbg !26
! LLVMIR:         call void @__kmpc_for_static_init_4u(%[[VAL_58]]* @3, i32 %[[VAL_57]], i32 34, i32* %[[VAL_17]], i32* %[[VAL_18]], i32* %[[VAL_19]], i32* %[[VAL_20]], i32 1, i32 1), !dbg !26
! LLVMIR:         %[[VAL_59:.*]] = load i32, i32* %[[VAL_18]], align 4, !dbg !26
! LLVMIR:         %[[VAL_60:.*]] = load i32, i32* %[[VAL_19]], align 4, !dbg !26
! LLVMIR:         %[[VAL_61:.*]] = sub i32 %[[VAL_60]], %[[VAL_59]], !dbg !26
! LLVMIR:         %[[VAL_62:.*]] = add i32 %[[VAL_61]], 1, !dbg !26
! LLVMIR:         br label %[[VAL_63:.*]], !dbg !26
! LLVMIR:       omp_collapsed.header:                             ; preds = %[[VAL_64:.*]], %[[VAL_55]]
! LLVMIR:         %[[VAL_65:.*]] = phi i32 [ 0, %[[VAL_55]] ], [ %[[VAL_66:.*]], %[[VAL_64]] ], !dbg !26
! LLVMIR:         br label %[[VAL_67:.*]], !dbg !26
! LLVMIR:       omp_collapsed.cond:                               ; preds = %[[VAL_63]]
! LLVMIR:         %[[VAL_68:.*]] = icmp ult i32 %[[VAL_65]], %[[VAL_62]], !dbg !26
! LLVMIR:         br i1 %[[VAL_68]], label %[[VAL_69:.*]], label %[[VAL_70:.*]], !dbg !26
! LLVMIR:       omp_collapsed.exit:                               ; preds = %[[VAL_67]]
! LLVMIR:         call void @__kmpc_for_static_fini(%[[VAL_58]]* @3, i32 %[[VAL_57]]), !dbg !26
! LLVMIR:         %[[VAL_71:.*]] = call i32 @__kmpc_global_thread_num(%[[VAL_58]]* @3), !dbg !26
! LLVMIR:         call void @__kmpc_barrier(%[[VAL_58]]* @4, i32 %[[VAL_71]]), !dbg !26
! LLVMIR:         br label %[[VAL_72:.*]], !dbg !26
! LLVMIR:       omp_collapsed.after:                              ; preds = %[[VAL_70]]
! LLVMIR:         br label %[[VAL_73:.*]], !dbg !26
! LLVMIR:       omp_loop.after:                                   ; preds = %[[VAL_72]]
! LLVMIR:       omp.par.pre_finalize:                             ; preds = %[[VAL_73]]
! LLVMIR:         br label %[[VAL_74:.*]]
  do i = 1, a
     do j= 1, b
        do k = 1, c
! FIRDialect:           omp.wsloop (%[[VAL_9:.*]], %[[VAL_10:.*]], %[[VAL_11:.*]]) : i32 = (%[[VAL_20]], %[[VAL_23]], %[[VAL_26]]) to (%[[VAL_21]], %[[VAL_24]], %[[VAL_27]]) inclusive step (%[[VAL_22]], %[[VAL_25]], %[[VAL_28]]) collapse(3) {
! FIRDialect:             %[[VAL_12:.*]] = fir.load %[[VAL_6]] : !fir.ref<i32>
! FIRDialect:             %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_9]] : i32
! FIRDialect:             %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_10]] : i32
! FIRDialect:             %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_11]] : i32
! FIRDialect:             fir.store %[[VAL_15]] to %[[VAL_6]] : !fir.ref<i32>
! FIRDialect:             omp.yield
! FIRDialect:           }
! LLVMIRDialect:             omp.wsloop (%[[VAL_24:.*]], %[[VAL_25:.*]], %[[VAL_26:.*]]) : i32 = (%[[VAL_4]], %[[VAL_4]], %[[VAL_4]]) to (%[[VAL_21]], %[[VAL_22]], %[[VAL_23]]) inclusive step (%[[VAL_4]], %[[VAL_4]], %[[VAL_4]]) collapse(3) {
! LLVMIRDialect:               %[[VAL_27:.*]] = llvm.load %[[VAL_20]] : !llvm.ptr<i32>
! LLVMIRDialect:               %[[VAL_28:.*]] = llvm.add %[[VAL_27]], %[[VAL_24]]  : i32
! LLVMIRDialect:               %[[VAL_29:.*]] = llvm.add %[[VAL_28]], %[[VAL_25]]  : i32
! LLVMIRDialect:               %[[VAL_30:.*]] = llvm.add %[[VAL_29]], %[[VAL_26]]  : i32
! LLVMIRDialect:               llvm.store %[[VAL_30]], %[[VAL_20]] : !llvm.ptr<i32>
! LLVMIRDialect:               omp.yield
! LLVMIRDialect:             }
! LLVMIR:       omp_collapsed.body:                               ; preds = %[[VAL_67]]
! LLVMIR:         %[[VAL_75:.*]] = add i32 %[[VAL_65]], %[[VAL_59]], !dbg !26
! LLVMIR:         %[[VAL_76:.*]] = urem i32 %[[VAL_75]], %[[VAL_52]], !dbg !26
! LLVMIR:         %[[VAL_77:.*]] = udiv i32 %[[VAL_75]], %[[VAL_52]], !dbg !26
! LLVMIR:         %[[VAL_78:.*]] = urem i32 %[[VAL_77]], %[[VAL_45]], !dbg !26
! LLVMIR:         %[[VAL_79:.*]] = udiv i32 %[[VAL_77]], %[[VAL_45]], !dbg !26
! LLVMIR:         br label %[[VAL_80:.*]], !dbg !26
! LLVMIR:       omp_loop.body:                                    ; preds = %[[VAL_69]]
! LLVMIR:         %[[VAL_81:.*]] = mul i32 %[[VAL_79]], 1
! LLVMIR:         %[[VAL_82:.*]] = add i32 %[[VAL_81]], 1
! LLVMIR:         br label %[[VAL_83:.*]]
! LLVMIR:       omp_loop.preheader3:                              ; preds = %[[VAL_80]]
! LLVMIR:         br label %[[VAL_84:.*]]
! LLVMIR:       omp_loop.body6:                                   ; preds = %[[VAL_83]]
! LLVMIR:         %[[VAL_85:.*]] = mul i32 %[[VAL_78]], 1
! LLVMIR:         %[[VAL_86:.*]] = add i32 %[[VAL_85]], 1
! LLVMIR:         br label %[[VAL_87:.*]]
! LLVMIR:       omp_loop.preheader14:                             ; preds = %[[VAL_84]]
! LLVMIR:         br label %[[VAL_88:.*]]
! LLVMIR:       omp_loop.body17:                                  ; preds = %[[VAL_87]]
! LLVMIR:         %[[VAL_89:.*]] = mul i32 %[[VAL_76]], 1
! LLVMIR:         %[[VAL_90:.*]] = add i32 %[[VAL_89]], 1
! LLVMIR:         br label %[[VAL_91:.*]]
! LLVMIR:       omp.wsloop.region:                                ; preds = %[[VAL_88]]
! LLVMIR:         %[[VAL_92:.*]] = load i32, i32* %[[VAL_93:.*]], align 4, !dbg !28
! LLVMIR:         %[[VAL_94:.*]] = add i32 %[[VAL_92]], %[[VAL_82]], !dbg !29
! LLVMIR:         %[[VAL_95:.*]] = add i32 %[[VAL_94]], %[[VAL_86]], !dbg !30
! LLVMIR:         %[[VAL_96:.*]] = add i32 %[[VAL_95]], %[[VAL_90]], !dbg !31
! LLVMIR:         store i32 %[[VAL_96]], i32* %[[VAL_93]], align 4, !dbg !32
! LLVMIR:         br label %[[VAL_97:.*]], !dbg !33
! LLVMIR:       omp.wsloop.exit:                                  ; preds = %[[VAL_91]]
! LLVMIR:         br label %[[VAL_98:.*]]
! LLVMIR:       omp_loop.after20:                                 ; preds = %[[VAL_97]]
! LLVMIR:         br label %[[VAL_99:.*]]
! LLVMIR:       omp_loop.after9:                                  ; preds = %[[VAL_98]]
! LLVMIR:         br label %[[VAL_64]]
! LLVMIR:       omp_collapsed.inc:                                ; preds = %[[VAL_99]]
! LLVMIR:         %[[VAL_66]] = add nuw i32 %[[VAL_65]], 1, !dbg !26
! LLVMIR:         br label %[[VAL_63]], !dbg !26
           x = x + i + j + k
        enddo
     enddo
  enddo
  !$omp end parallel do
! FIRDialect:           omp.terminator
! FIRDialect:         }
! FIRDialect:         return
! FIRDialect:       }
! LLVMIRDialect:             omp.terminator
! LLVMIRDialect:           }
! LLVMIRDialect:           llvm.return
! LLVMIRDialect:         }
end program wsloop_collapse


