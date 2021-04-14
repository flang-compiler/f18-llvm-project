! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL @_QQmain

   real :: a(10), b(10)

   ! CHECK-DAG: %[[a:.*]] = fir.address_of(@_QEa) : !fir.ref<!fir.array<10xf32>>
   ! CHECK-DAG: %[[b:.*]] = fir.address_of(@_QEb) : !fir.ref<!fir.array<10xf32>>
   ! CHECK-DAG: %[[bv:.*]] = fir.array_load %[[b]](%{{.*}}) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
   ! CHECK-DAG: %[[av1:.*]] = fir.array_load %[[a]](%{{.*}}) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
   ! CHECK-DAG: %[[av2:.*]] = fir.array_load %[[a]](%{{.*}}) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
   ! CHECK-DAG: %[[four:.*]] = constant 4.0{{.*}} : f32
   ! CHECK: fir.do_loop
   ! CHECK: %[[fet:.*]] = fir.array_fetch %[[av2]]
   ! CHECK: %[[cmp:.*]] = fir.cmpf "ogt", %[[fet]], %[[four]]
   ! CHECK: fir.if %[[cmp]]
   ! CHECK: fir.array_fetch
   ! CHECK: fir.negf
   ! CHECK: fir.array_update
   ! CHECK: fir.array_merge_store
   where (a > 4.0) b = -a

   ! Test that the basic structure is correct
   where (a > 100.0)
   ! loop with an if
     ! CHECK-DAG: constant 2
     ! CHECK-DAG: %[[cst:.*]] = constant 1.0{{.*}}2 : f32
     ! CHECK: fir.do_loop
     ! CHECK: fir.array_fetch
     ! CHECK: %[[tst:.*]] = fir.cmpf "ogt", %{{.*}}, %[[cst]]
     ! CHECK: fir.if %[[tst]]
     ! CHECK: fir.array_fetch
     ! CHECK: mulf
     ! CHECK: fir.array_update
     ! CHECK: fir.array_merge_store
     b = 2.0 * a
   elsewhere (a > 50.0)
   ! loop with else if
     ! CHECK-DAG: constant 3.0
     ! CHECK-DAG: %[[cst50:.*]] = constant 5.0{{.*}}1 : f32
     ! CHECK-DAG: %[[cst:.*]] = constant 1.0{{.*}}2 : f32
     ! CHECK: fir.do_loop
     ! CHECK: fir.array_fetch
     ! CHECK: %[[tst:.*]] = fir.cmpf "ogt", %{{.*}}, %[[cst]]
     ! CHECK: fir.if %[[tst]]
     ! CHECK: } else {
     ! CHECK: fir.array_fetch
     ! CHECK: %[[tst:.*]] = fir.cmpf "ogt", %{{.*}}, %[[cst50]]
     ! CHECK: fir.if %[[tst]]
     ! CHECK: fir.array_fetch
     ! CHECK: addf
     ! CHECK: fir.array_update
     ! CHECK: fir.array_merge_store
     b = 3.0 + a
   ! Identical structure again
     ! CHECK-DAG: constant 1.0{{.*}}0 : f32
     ! CHECK-DAG: %[[cst50:.*]] = constant 5.0{{.*}}1 : f32
     ! CHECK-DAG: %[[cst:.*]] = constant 1.0{{.*}}2 : f32
     ! CHECK: fir.do_loop
     ! CHECK: fir.array_fetch
     ! CHECK: %[[tst:.*]] = fir.cmpf "ogt", %{{.*}}, %[[cst]]
     ! CHECK: fir.if %[[tst]]
     ! CHECK: } else {
     ! CHECK: fir.array_fetch
     ! CHECK: %[[tst:.*]] = fir.cmpf "ogt", %{{.*}}, %[[cst50]]
     ! CHECK: fir.if %[[tst]]
     ! CHECK: fir.array_fetch
     ! CHECK: subf
     ! CHECK: fir.array_update
     ! CHECK: fir.array_merge_store
     a = a - 1.0
   elsewhere
   ! loop with if..else if..else if true
     ! CHECK-DAG: constant 2.
     ! CHECK-DAG: %[[cst50:.*]] = constant 5.0{{.*}}1 : f32
     ! CHECK-DAG: %[[cst:.*]] = constant 1.0{{.*}}2 : f32
     ! CHECK: fir.do_loop
     ! CHECK: fir.array_fetch
     ! CHECK: %[[tst:.*]] = fir.cmpf "ogt", %{{.*}}, %[[cst]]
     ! CHECK: fir.if %[[tst]]
     ! CHECK: } else {
     ! CHECK: fir.array_fetch
     ! CHECK: %[[tst:.*]] = fir.cmpf "ogt", %{{.*}}, %[[cst50]]
     ! CHECK: fir.if %[[tst]]
     ! CHECK: } else {
     ! CHECK: %[[t:.*]] = constant true
     ! CHECK: fir.if %[[t]]
     ! CHECK: fir.array_fetch
     ! CHECK: divf
     ! CHECK: fir.array_update
     ! CHECK: fir.array_merge_store
     a = a / 2.0
   end where
   ! CHECK: return
end
