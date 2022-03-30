! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN: FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp %s -o - | \
! RUN: tco --disable-llvm --print-ir-after=fir-to-llvm-ir 2>&1 | \
! RUN: FileCheck %s --check-prefix=LLVMIRDialect

! This test checks the lowering of atomic write

!FIRDialect: func @_QQmain() {
!FIRDialect: %[[VAR_X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!FIRDialect: %[[VAR_Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
!FIRDialect: %[[VAR_Z:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFEz"}
!FIRDialect: %[[CONST_44:.*]] = arith.constant 44 : i32
!FIRDialect: omp.atomic.write %[[VAR_X]], %[[CONST_44]] memory_order(seq_cst) hint(uncontended) : !fir.ref<i32>, i32
!FIRDialect: %[[CONST_7:.*]] = arith.constant 7 : i32
!FIRDialect: {{.*}} = fir.load %[[VAR_Y]] : !fir.ref<i32>
!FIRDialect: %[[VAR_7y:.*]] = arith.muli %c7_i32, %3 : i32
!FIRDialect: omp.atomic.write %[[VAR_X]], %[[VAR_7y]] memory_order(relaxed) : !fir.ref<i32>, i32
!FIRDialect: %[[CONST_10:.*]] = arith.constant 10 : i32
!FIRDialect: {{.*}} = fir.load %[[VAR_X]] : !fir.ref<i32>
!FIRDialect: {{.*}} = arith.muli %[[CONST_10]], {{.*}} : i32
!FIRDialect: {{.*}} = fir.load %[[VAR_Z]] : !fir.ref<i32>
!FIRDialect: %[[CONST_2:.*]] = arith.constant 2 : i32
!FIRDialect: {{.*}} = arith.divsi {{.*}}, %[[CONST_2]] : i32
!FIRDialect: {{.*}} = arith.addi {{.*}}, {{.*}} : i32
!FIRDialect: omp.atomic.write %[[VAR_Y]], {{.*}} memory_order(release) hint(speculative) : !fir.ref<i32>, i32
!FIRDialect: return
!FIRDialect: }

!LLVMIRDialect: llvm.func @_QQmain() {
!LLVMIRDialect: %[[LLVM_VAR_44:.*]] = llvm.mlir.constant(44 : i32) : i32
!LLVMIRDialect: %[[LLVM_VAR_7:.*]] = llvm.mlir.constant(7 : i32) : i32
!LLVMIRDialect: %[[LLVM_VAR_10:.*]] = llvm.mlir.constant(10 : i32) : i32
!LLVMIRDialect: %[[LLVM_VAR_2:.*]] = llvm.mlir.constant(2 : i32) : i32
!LLVMIRDialect: %[[LLVM_VAR_1:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect: %[[LLVM_VAR_X:.*]] = llvm.alloca {{.*}} x i32 {bindc_name = "x", in_type = i32, operand_segment_sizes = dense<0> : vector<2xi32>, uniq_name = "_QFEx"} : (i64) -> !llvm.ptr<i32>
!LLVMIRDialect: %[[LLVM_VAR_1_SECOND:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect: %[[LLVM_VAR_Y:.*]] = llvm.alloca {{.*}} x i32 {bindc_name = "y", in_type = i32, operand_segment_sizes = dense<0> : vector<2xi32>, uniq_name = "_QFEy"} : (i64) -> !llvm.ptr<i32>
!LLVMIRDialect: %[[LLVM_VAR_1_THIRD:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect: %[[LLVM_VAR_Z:.*]] = llvm.alloca {{.*}} x i32 {bindc_name = "z", in_type = i32, operand_segment_sizes = dense<0> : vector<2xi32>, uniq_name = "_QFEz"} : (i64) -> !llvm.ptr<i32>
!LLVMIRDialect: omp.atomic.write %[[LLVM_VAR_X]], %[[LLVM_VAR_44]] memory_order(seq_cst) hint(uncontended) : !llvm.ptr<i32>, i32
!LLVMIRDialect: {{.*}} = llvm.load %[[LLVM_VAR_Y]] : !llvm.ptr<i32>
!LLVMIRDialect: %[[LLVM_VAR_MUL_RESULT:.*]] = llvm.mul {{.*}}, %[[LLVM_VAR_7]] : i32
!LLVMIRDialect: omp.atomic.write %[[LLVM_VAR_X]], %[[LLVM_VAR_MUL_RESULT]] memory_order(relaxed) : !llvm.ptr<i32>, i32
!LLVMIRDialect: {{.*}} = llvm.load %[[LLVM_VAR_X]] : !llvm.ptr<i32>
!LLVMIRDialect: {{.*}} = llvm.mul {{.*}}, %[[LLVM_VAR_10]] : i32
!LLVMIRDialect: {{.*}} = llvm.load %[[LLVM_VAR_Z]] : !llvm.ptr<i32>
!LLVMIRDialect: {{.*}} = llvm.sdiv {{.*}}, %[[LLVM_VAR_2]] : i32
!LLVMIRDialect: {{.*}} = llvm.add {{.*}} : i32
!LLVMIRDialect: omp.atomic.write %[[LLVM_VAR_Y]], {{.*}} memory_order(release) hint(speculative) : !llvm.ptr<i32>, i32
!LLVMIRDialect: llvm.return
!LLVMIRDialect: }

program OmpAtomicWrite
    use omp_lib
    integer :: x, y, z 
    !$omp atomic seq_cst write hint(omp_sync_hint_uncontended)
        x = 8*4 + 12

    !$omp atomic write relaxed
        x = 7 * y

    !$omp atomic write release hint(omp_sync_hint_speculative)
        y = 10*x + z/2
end program OmpAtomicWrite
