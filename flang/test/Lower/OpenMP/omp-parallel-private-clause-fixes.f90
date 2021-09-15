! This test checks a few bug fixes in the PRIVATE clause lowering

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s 

! CHECK-LABEL: multiple_private_fix
! CHECK:         %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFmultiple_private_fixEi"}
! CHECK:         %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFmultiple_private_fixEj"}
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFmultiple_private_fixEx"}
! CHECK:         omp.parallel {
! CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "j", pinned}
! CHECK:           %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFmultiple_private_fixEx"}
! CHECK:           %[[VAL_2:.*]] = constant 1 : i32
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_4:.*]] : !fir.ref<i32>
! CHECK:           %[[VAL_5:.*]] = constant 1 : i32
! CHECK:           omp.wsloop (%[[VAL_6:.*]]) : i32 = (%[[VAL_2]]) to (%[[VAL_3]]) step (%[[VAL_5]]) inclusive {
! CHECK:             %[[VAL_7:.*]] = constant 1 : i32
! CHECK:             %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
! CHECK:             %[[VAL_9:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:             %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK:             %[[VAL_11:.*]] = constant 1 : index
! CHECK:             %[[VAL_12:.*]] = fir.do_loop %[[VAL_13:.*]] = %[[VAL_8]] to %[[VAL_10]] step %[[VAL_11]] -> index {
! CHECK:               %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (index) -> i32
! CHECK:               fir.store %[[VAL_14]] to %[[VAL_0]] : !fir.ref<i32>
! CHECK:               %[[VAL_15:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:               %[[VAL_16:.*]] = addi %[[VAL_6]], %[[VAL_15]] : i32
! CHECK:               fir.store %[[VAL_16]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:               %[[VAL_17:.*]] = addi %[[VAL_13]], %[[VAL_11]] : index
! CHECK:               fir.result %[[VAL_17]] : index
! CHECK:             }
! CHECK:             %[[VAL_18:.*]] = fir.convert %[[VAL_19:.*]] : (index) -> i32
! CHECK:             fir.store %[[VAL_18]] to %[[VAL_0]] : !fir.ref<i32>
! CHECK:             omp.yield
! CHECK:           }
! CHECK:           omp.terminator
! CHECK:         }
! CHECK:         return
subroutine multiple_private_fix(gama)
        integer :: i, j, x, gama
!$OMP PARALLEL DO PRIVATE(j,x)
        do i = 1, gama
          do j = 1, gama
            x = i + j
          end do
        end do
!$OMP END PARALLEL DO
end subroutine
