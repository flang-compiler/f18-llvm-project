<!--===- docs/LLVMCodingStyle.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Brief Note on Flang's LLVM style.

In the directories that deal extensively with MLIR and LLVM, the
decision was made to adopt the MLIR style and coding conventions. The
lowering directories convert front-end functional data structures to
MLIR and make very heavy use of MLIR interfaces and data
structures. The FIR dialect is itself an extension of MLIR.

Furthermore, the optimizer (passes over FIR) and code generation
(converting to LLVM IR) also make very heavy use of MLIR and LLVM
interfaces and data structures.

For consistency's sake, lowering, codegen, and the optimizer use the
[MLIR style](https://mlir.llvm.org/getting_started/DeveloperGuide/).

One additional clarification to the style used within these flang
directories is with respect to [the use of
auto](https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable). While
LLVM does not disallow `auto`, it does come short of suggesting `auto` be
used rather liberally. In flang, we take a more open stance and use `auto`
for most local variable declarations. Function signatures, on the other
hand, should have specified types.

For the flang project, use of `auto` in this way, which allows the
compiler to type inference local variables, is generally seen as a win
for the following reasons.

- Some of the types are extremely large and typing them out is
  error-prone and greatly diminishes understandability.
- Repetitive token clutter from all the namespace tags that are used
  is removed.
- Explicitly declared types can and do introduce hard-to-find bugs from
  unexpected or incorrect truncations, extensions, and other conversions.
- Explicitly declared types increase refactoring costs when
  interfaces are rapidly evolving.
