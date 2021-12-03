<!--===- docs/FIRArrayOperations.md 
  
   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
-->

# Design: FIR Array operations

```eval_rst
.. contents::
   :local:
```

## General

The array operations in FIR model the copy-in/copy-out semantics over Fortran
statements.

They are currently 6 array operations:
- `fir.array_load`
- `fir.array_merge_store`
- `fir.array_fetch`
- `fir.array_update`
- `fir.array_access`
- `fir.array_amend`

`array_load`(s) and `array_merge_store` are a pairing that brackets the lifetime
of the array copies.

`array_fetch` and `array_update` are defined to work as getter/setter pairs on 
values of elements from loaded array copies. These have "GEP-like" syntax and
semantics.

`array_access` and `array_amend` are defined to work as getter/setter pairs on
references to elements in loaded array copies. `array_access` has "GEP-like"
syntax. `array_amend` annotates which loaded array copy is being written to.
It is invalid to update an array copy without array_amend; doing so will result
in undefined behavior.

## array_load

This operation taken with `array_merge_store` captures Fortran's
copy-in/copy-out semantics. One way to think of this is that array_load
creates a snapshot copy of the entire array. This copy can then be used
as the "original value" of the array while the array's new value is
computed. The `array_merge_store` operation is the copy-out semantics, which
merge the updates with the original array value to produce the final array
result. This abstracts the copy operations as opposed to always creating
copies or requiring dependence analysis be performed on the syntax trees
and before lowering to the IR.

Load an entire array as a single SSA value.

```fortran
  real :: a(o:n,p:m)
  ...
  ... = ... a ...
```

One can use `fir.array_load` to produce an ssa-value that captures an
immutable value of the entire array `a`, as in the Fortran array expression
shown above. Subsequent changes to the memory containing the array do not
alter its composite value. This operation let's one load an array as a
value while applying a runtime shape, shift, or slice to the memory
reference, and its semantics guarantee immutability.

```mlir
%s = fir.shape_shift %o, %n, %p, %m : (index, index, index, index) -> !fir.shape<2>
// load the entire array 'a'
%v = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
// a fir.store here into array %a does not change %v
```

# array_merge_store

The `array_merge_store` operation store a merged array value to memory. 


```fortran
  real :: a(n,m)
  ...
  a = ...
```

One can use `fir.array_merge_store` to merge/copy the value of `a` in an
array expression as shown above.

```mlir
  %v = fir.array_load %a(%shape) : ...
  %r = fir.array_update %v, %f, %i, %j : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
  fir.array_merge_store %v, %r to %a : !fir.ref<!fir.array<?x?xf32>>
```

This operation merges the original loaded array value, `%v`, with the
chained updates, `%r`, and stores the result to the array at address, `%a`.

## array_fetch

The `array_fetch` operation fetches the value of an element in an array value.

```fortran
  real :: a(n,m)
  ...
  ... a ...
  ... a(r,s+1) ...
```

One can use `fir.array_fetch` to fetch the (implied) value of `a(i,j)` in
an array expression as shown above. It can also be used to extract the
element `a(r,s+1)` in the second expression.

```mlir
  %s = fir.shape %n, %m : (index, index) -> !fir.shape<2>
  // load the entire array 'a'
  %v = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
  // fetch the value of one of the array value's elements
  %1 = fir.array_fetch %v, %i, %j : (!fir.array<?x?xf32>, index, index) -> f32
```

It is only possible to use `array_fetch` on an `array_load` result value.

## array_update

The `array_update` operation is used to update the value of an element in an
array value. A new array value is returned where all element values of the input
array are identical except for the selected element which is the value passed in
the update.

```fortran
  real :: a(n,m)
  ...
  a = ...
```

One can use `fir.array_update` to update the (implied) value of `a(i,j)`
in an array expression as shown above.

```mlir
  %s = fir.shape %n, %m : (index, index) -> !fir.shape<2>
  // load the entire array 'a'
  %v = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
  // update the value of one of the array value's elements
  // %r_{ij} = %f  if (i,j) = (%i,%j),   %v_{ij} otherwise
  %r = fir.array_update %v, %f, %i, %j : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
  fir.array_merge_store %v, %r to %a : !fir.ref<!fir.array<?x?xf32>>
```

An array value update behaves as if a mapping function from the indices
to the new value has been added, replacing the previous mapping. These
mappings can be added to the ssa-value, but will not be materialized in
memory until the `fir.array_merge_store` is performed.

## array_access

The `array_access` operationis used to fetch the memory reference of an element
in an array value.

```fortran
  real :: a(n,m)
  ...
  ... a ...
  ... a(r,s+1) ...
```

One can use `fir.array_access` to recover the implied memory reference to
the element `a(i,j)` in an array expression `a` as shown above. It can also
be used to recover the reference element `a(r,s+1)` in the second
expression.

```mlir
  %s = fir.shape %n, %m : (index, index) -> !fir.shape<2>
  // load the entire array 'a'
  %v = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
  // fetch the value of one of the array value's elements
  %1 = fir.array_access %v, %i, %j : (!fir.array<?x?xf32>, index, index) -> !fir.ref<f32>
```

It is only possible to use `array_access` on an `array_load` result value.

## array_amend

The `array_amend` operation marks an array value as having been changed via its
reference. The reference into the array value is obtained via a
`fir.array_access` op.

```mlir
  // fetch the value of one of the array value's elements
  %1 = fir.array_access %v, %i, %j : (!fir.array<?x?xT>, index, index) -> !fir.ref<T>
  // modify the element by storing data using %1 as a reference
  %2 = ... %1 ...
  // mark the array value
  %new_v = fir.array_amend %v, %2 : (!fir.array<?x?xT>, !fir.ref<T>) -> !fir.array<?x?xT>
```

## Array value copy pass

One of the main purpose of the array operations present in FIR is to be able to
perform the dependence analysis and elide copies where possible with a MLIR
pass. This pass is called the `array-value-copy` pass.
The analysis detects if there are any conflicts. A conflicts is when one of the
following cases occurs:

1. There is an `array_update`/`array_amend` to an array value/reference, a_j,
   such that a_j was loaded from the same array memory reference (array_j) but
   with a different shape as the other array values a_i, where i != j.
   [Possible overlapping arrays.]
2. There is either an `array_fetch`/`array_access` or
   `array_update`/`array_amend` of a_j with a different set of index values.
   [Possible loop-carried dependence.]

`array_update` writes an entire element in the loaded array value. So an 
`array_update` that does not change any of the arrays fetched from or updates 
the exact same element that was read on the current iteration does not
introduce a dependence.

`array_amend` may be a partial update to an element, such as a substring. In
that case, there is no dependence if all the other `array_access` ops are
referencing other arrays. We conservatively assume there may be an
overlap like in s(:)(1:4) = s(:)(3:6) where s is an array of characters.

If none of the array values overlap in storage and the accesses are not
loop-carried, then the arrays are conflict-free and no copies are required.

Below is an example of the FIR/MLIR code before and after the `array-value-copy`
pass.

```fortran
subroutine s(a,l,u)
  type t
    integer m
  end type t
  type(t) :: a(:)
  integer :: l, u
  forall (i=l:u)
    a(i) = a(u-i+1)
  end forall
end
```

```
func @_QPs(%arg0: !fir.box<!fir.array<?x!fir.type<_QFsTt{m:i32}>>>, %arg1: !fir.ref<i32>, %arg2: !fir.ref<i32>) {
  %0 = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  %1 = fir.load %arg1 : !fir.ref<i32>
  %2 = fir.convert %1 : (i32) -> index
  %3 = fir.load %arg2 : !fir.ref<i32>
  %4 = fir.convert %3 : (i32) -> index
  %c1 = arith.constant 1 : index
  %5 = fir.array_load %arg0 : (!fir.box<!fir.array<?x!fir.type<_QFsTt{m:i32}>>>) -> !fir.array<?x!fir.type<_QFsTt{m:i32}>>
  %6 = fir.array_load %arg0 : (!fir.box<!fir.array<?x!fir.type<_QFsTt{m:i32}>>>) -> !fir.array<?x!fir.type<_QFsTt{m:i32}>>
  %7 = fir.do_loop %arg3 = %2 to %4 step %c1 unordered iter_args(%arg4 = %5) -> (!fir.array<?x!fir.type<_QFsTt{m:i32}>>) {
    %8 = fir.convert %arg3 : (index) -> i32
    fir.store %8 to %0 : !fir.ref<i32>
    %c1_i32 = arith.constant 1 : i32
    %9 = fir.load %arg2 : !fir.ref<i32>
    %10 = fir.load %0 : !fir.ref<i32>
    %11 = arith.subi %9, %10 : i32
    %12 = arith.addi %11, %c1_i32 : i32
    %13 = fir.convert %12 : (i32) -> i64
    %14 = fir.convert %13 : (i64) -> index
    %15 = fir.array_access %6, %14 {Fortran.offsets} : (!fir.array<?x!fir.type<_QFsTt{m:i32}>>, index) -> !fir.ref<!fir.type<_QFsTt{m:i32}>>
    %16 = fir.load %0 : !fir.ref<i32>
    %17 = fir.convert %16 : (i32) -> i64
    %18 = fir.convert %17 : (i64) -> index
    %19 = fir.array_access %arg4, %18 {Fortran.offsets} : (!fir.array<?x!fir.type<_QFsTt{m:i32}>>, index) -> !fir.ref<!fir.type<_QFsTt{m:i32}>>
    %20 = fir.load %15 : !fir.ref<!fir.type<_QFsTt{m:i32}>>
    fir.store %20 to %19 : !fir.ref<!fir.type<_QFsTt{m:i32}>>
    %21 = fir.array_amend %arg4, %19 : (!fir.array<?x!fir.type<_QFsTt{m:i32}>>, !fir.ref<!fir.type<_QFsTt{m:i32}>>) -> !fir.array<?x!fir.type<_QFsTt{m:i32}>>
    fir.result %21 : !fir.array<?x!fir.type<_QFsTt{m:i32}>>
  }
  fir.array_merge_store %5, %7 to %arg0 : !fir.array<?x!fir.type<_QFsTt{m:i32}>>, !fir.array<?x!fir.type<_QFsTt{m:i32}>>, !fir.box<!fir.array<?x!fir.type<_QFsTt{m:i32}>>>
  return
}
```

```
func @_QPs(%arg0: !fir.box<!fir.array<?x!fir.type<_QFsTt{m:i32}>>>, %arg1: !fir.ref<i32>, %arg2: !fir.ref<i32>) {
    %0 = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
    %1 = fir.load %arg1 : !fir.ref<i32>
    %2 = fir.convert %1 : (i32) -> index
    %3 = fir.load %arg2 : !fir.ref<i32>
    %4 = fir.convert %3 : (i32) -> index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %5:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.array<?x!fir.type<_QFsTt{m:i32}>>>, index) -> (index, index, index)
    %6 = fir.shape %5#1 : (index) -> !fir.shape<1>
    // Allocate copy
    %7 = fir.allocmem !fir.array<?x!fir.type<_QFsTt{m:i32}>>, %5#1
    %8 = fir.convert %5#1 : (index) -> index
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %9 = arith.subi %8, %c1_1 : index
    // Initialize copy
    fir.do_loop %arg3 = %c0_0 to %9 step %c1_1 {
      %c1_4 = arith.constant 1 : index
      %15 = arith.addi %arg3, %c1_4 : index
      %16 = fir.array_coor %arg0(%6) %15 : (!fir.box<!fir.array<?x!fir.type<_QFsTt{m:i32}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QFsTt{m:i32}>>
      %c1_5 = arith.constant 1 : index
      %17 = arith.addi %arg3, %c1_5 : index
      %18 = fir.array_coor %7(%6) %17 : (!fir.heap<!fir.array<?x!fir.type<_QFsTt{m:i32}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QFsTt{m:i32}>>
      %19 = fir.field_index m, !fir.type<_QFsTt{m:i32}>
      %20 = fir.coordinate_of %16, %19 : (!fir.ref<!fir.type<_QFsTt{m:i32}>>, !fir.field) -> !fir.ref<i32>
      %21 = fir.coordinate_of %18, %19 : (!fir.ref<!fir.type<_QFsTt{m:i32}>>, !fir.field) -> !fir.ref<i32>
      %22 = fir.load %20 : !fir.ref<i32>
      fir.store %22 to %21 : !fir.ref<i32>
    }
    %10 = fir.undefined !fir.array<?x!fir.type<_QFsTt{m:i32}>>
    %11 = fir.undefined !fir.array<?x!fir.type<_QFsTt{m:i32}>>
    // Perform the actual work
    %12 = fir.do_loop %arg3 = %2 to %4 step %c1 unordered iter_args(%arg4 = %10) -> (!fir.array<?x!fir.type<_QFsTt{m:i32}>>) {
      %15 = fir.convert %arg3 : (index) -> i32
      fir.store %15 to %0 : !fir.ref<i32>
      %c1_i32 = arith.constant 1 : i32
      %16 = fir.load %arg2 : !fir.ref<i32>
      %17 = fir.load %0 : !fir.ref<i32>
      %18 = arith.subi %16, %17 : i32
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = fir.convert %19 : (i32) -> i64
      %21 = fir.convert %20 : (i64) -> index
      %22 = fir.array_coor %arg0 %21 : (!fir.box<!fir.array<?x!fir.type<_QFsTt{m:i32}>>>, index) -> !fir.ref<!fir.type<_QFsTt{m:i32}>>
      %23 = fir.load %0 : !fir.ref<i32>
      %24 = fir.convert %23 : (i32) -> i64
      %25 = fir.convert %24 : (i64) -> index
      %26 = fir.array_coor %7(%6) %25 : (!fir.heap<!fir.array<?x!fir.type<_QFsTt{m:i32}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QFsTt{m:i32}>>
      %27 = fir.load %22 : !fir.ref<!fir.type<_QFsTt{m:i32}>>
      fir.store %27 to %26 : !fir.ref<!fir.type<_QFsTt{m:i32}>>
      %28 = fir.undefined !fir.array<?x!fir.type<_QFsTt{m:i32}>>
      fir.result %28 : !fir.array<?x!fir.type<_QFsTt{m:i32}>>
    }
    %13 = fir.convert %5#1 : (index) -> index
    %c0_2 = arith.constant 0 : index
    %c1_3 = arith.constant 1 : index
    %14 = arith.subi %13, %c1_3 : index
    // Move tha value back from the copy to the original array
    fir.do_loop %arg3 = %c0_2 to %14 step %c1_3 {
      %c1_4 = arith.constant 1 : index
      %15 = arith.addi %arg3, %c1_4 : index
      %16 = fir.array_coor %7(%6) %15 : (!fir.heap<!fir.array<?x!fir.type<_QFsTt{m:i32}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QFsTt{m:i32}>>
      %c1_5 = arith.constant 1 : index
      %17 = arith.addi %arg3, %c1_5 : index
      %18 = fir.array_coor %arg0(%6) %17 : (!fir.box<!fir.array<?x!fir.type<_QFsTt{m:i32}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QFsTt{m:i32}>>
      %19 = fir.field_index m, !fir.type<_QFsTt{m:i32}>
      %20 = fir.coordinate_of %16, %19 : (!fir.ref<!fir.type<_QFsTt{m:i32}>>, !fir.field) -> !fir.ref<i32>
      %21 = fir.coordinate_of %18, %19 : (!fir.ref<!fir.type<_QFsTt{m:i32}>>, !fir.field) -> !fir.ref<i32>
      %22 = fir.load %20 : !fir.ref<i32>
      fir.store %22 to %21 : !fir.ref<i32>
    }
    fir.freemem %7 : !fir.heap<!fir.array<?x!fir.type<_QFsTt{m:i32}>>>
    return
  }
```
