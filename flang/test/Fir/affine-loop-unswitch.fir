
!arr_d1 = type !fir.ref<!fir.array<?xf32>>
#arr_len = affine_map<()[j1,k1] -> (k1 - j1 + 1)>

func @calc(%a: !arr_d1, %v: f32) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %len = constant 100 : index
  %dims = fir.gendims %c1, %len, %c1
       : (index, index, index) -> !fir.dims<1>

  fir.do_loop %i = %c1 to %len step %c1 {
    fir.do_loop %j = %c1 to %len step %c1 {
      fir.do_loop %k = %c1 to %len step %c1 {
        %im2 = subi %i, %c2 : index
        %cond = cmpi "sgt", %im2, %c0 : index
        fir.if %cond {
          %a_idx = fir.array_coor %a(%dims) %i
            : (!arr_d1, !fir.dims<1>, index) -> !fir.ref<f32>
          fir.store %v to %a_idx : !fir.ref<f32>
        }
      }
    }
  }
  return
}
