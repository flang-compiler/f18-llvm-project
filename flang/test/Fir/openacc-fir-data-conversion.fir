// RUN: fir-opt --openacc-data-conversion %s | FileCheck %s

func @_QQmain() {
  %0 = fir.address_of(@_QEa) : !fir.ref<!fir.array<10xf32>>
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QEi"}
  acc.enter_data create(%0, %1 : !fir.ref<!fir.array<10xf32>>, !fir.ref<i32>)
  return
}
fir.global internal @_QEa : !fir.array<10xf32> {
  %0 = fir.undefined !fir.array<10xf32>
  fir.has_value %0 : !fir.array<10xf32>
}

// CHECK: %{{.*}} = fir.cast %{{.*}} : !fir.ref<!fir.array<10xf32>> to !llvm.ptr<array<10 x f32>>
// CHECK: %{{.*}} = fir.cast %{{.*}} : !fir.ref<i32> to !llvm.ptr<i32>
// CHECK: acc.enter_data create(%{{.*}}, %{{.*}} : !llvm.ptr<array<10 x f32>>, !llvm.ptr<i32>)
