<!--===- docs/LLVMCodingStyle.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Brief Note on Flang's LLVM style.

In the directories that deal extensively with MLIR and LLVM, the MLIR
coding style, based on LLVM's coding style, is used. These directories include

- Lowering, which converts front-end functional data structures to the FIR
dialect of MLIR
- Optimizer, which are passes over FIR
- Code generation, which is converting FIR to LLVM IR

[The MLIR coding style can be found
here.](https://mlir.llvm.org/getting_started/DeveloperGuide/).

The flang convention in these directories also follows a more liberal use
of `auto` and type inferencing than [what is documented
here](https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable). In
Flang, `auto` is used for nearly all local variable declarations that have
initialization expressions. Obviously, variables that make use of default
initialization or that use a constructor call must use declared types and
not `auto`. Function signatures have specified types unless they are
lambdas where `auto` is used as a template type variable.

For the flang project, use of `auto` in this way allows the compiler to
type inference variables, resulting in the following benefits:

- Extremely long and complicated type names are not used, which is less
  error-prone and greatly improves understandability.
- The clutter from repetitive namespace tags is removed.
- Explicitly declared types can and do introduce hard-to-find bugs from
  unexpected or incorrect truncations, extensions, and other conversions.
- Using `auto` instead of declared types simplifies refactoring when
  interfaces are rapidly evolving.
- `auto` can be required when writing some templatized code.
