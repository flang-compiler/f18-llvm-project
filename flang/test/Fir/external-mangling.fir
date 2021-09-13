// RUN: fir-opt --external-name-interop %s | FileCheck %s
// RUN: tco --external-name-interop %s | FileCheck %s
// RUN: tco --external-name-interop %s | tco | FileCheck %s --check-prefix=LLVMIR

func @_QPfoo() {  
  %c0 = constant 0 : index
  %0 = fir.address_of(@_QBa) : !fir.ref<!fir.array<4xi8>>
  %1 = fir.convert %0 : (!fir.ref<!fir.array<4xi8>>) -> !fir.ref<!fir.array<?xi8>>
  %2 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ref<i32>
  %4 = fir.address_of(@_QB) : !fir.ref<!fir.array<4xi8>>
  %5 = fir.convert %4 : (!fir.ref<!fir.array<4xi8>>) -> !fir.ref<!fir.array<?xi8>>
  %6 = fir.coordinate_of %5, %c0 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  %7 = fir.convert %6 : (!fir.ref<i8>) -> !fir.ref<f32>
  fir.call @_QPbar(%3) : (!fir.ref<i32>) -> ()
  fir.call @_QPbar2(%7) : (!fir.ref<f32>) -> ()
  %e6 = fir.alloca tuple<i32,f64>
  %8 = fir.emboxproc @_QPfoo_impl, %e6 : ((!fir.box<!fir.type<derived3{f:f32}>>) -> (), !fir.ref<tuple<i32,f64>>) -> !fir.boxproc<(!fir.box<!fir.type<derived3{f:f32}>>) -> ()>
  return
}
fir.global common @_QBa(dense<0> : vector<4xi8>) : !fir.array<4xi8>
fir.global common @_QB(dense<0> : vector<4xi8>) : !fir.array<4xi8>
func private @_QPbar(!fir.ref<i32>)
func private @_QPbar2(!fir.ref<f32>)

// CHECK: func @foo_
// CHECK: %{{.*}} = fir.address_of(@a_) : !fir.ref<!fir.array<4xi8>>
// CHECK: %{{.*}} = fir.address_of(@__BLNK__) : !fir.ref<!fir.array<4xi8>>
// CHECK: fir.call @bar_
// CHECK: fir.call @bar2_
// CHECK: %{{.*}}= fir.emboxproc @foo_impl_
// CHECK: fir.global common @a_(dense<0> : vector<4xi8>) : !fir.array<4xi8>
// CHECK: fir.global common @__BLNK__(dense<0> : vector<4xi8>) : !fir.array<4xi8>
// CHECK: func private @bar_(!fir.ref<i32>)

// LLVMIR: @a_ = common global [4 x i8] zeroinitializer
// LLVMIR: @__BLNK__ = common global [4 x i8] zeroinitializer
// LLVMIR: define void @foo_()
// LLVMIR: call void @bar_(
// LLVMIR: call void @bar2_(
// LLVMIR: declare void @bar_(
// LLVMIR: declare void @bar2_(
