<!--===- README.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# FIR

Yes, we've moved. This is now the working branch for FIR development.

## Monorepo now contains Flang!

### In-tree build

1. Get the stuff.

```
  git clone git@github.com:flang-compiler/f18-llvm-project.git
```

2. Get "on" the right branches.

```
  (cd f18-llvm-project ; git checkout fir-dev)
```

3. (not needed!)
             
4. Create a build space for cmake and make (or ninja)

```
  mkdir build
  cd build
  cmake ../f18-llvm-project/llvm -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_PROJECTS="flang;mlir" -DCMAKE_CXX_STANDARD=17 -DLLVM_BUILD_TOOLS=On -DLLVM_INSTALL_UTILS=On <other-arguments>
```

5. Build everything

```
  make
  make check-flang
  make install
```

### Out-of-tree build

Assuming someone was nice enough to build MLIR and LLVM libraries and
install them in a convenient place for you, then you may want to do a
standalone build.

1. Get the stuff is the same as above. Get the code from the same repos.

2. Get on the right branches. Again, same as above.

3. Create a build space for cmake and make (or ninja)

```
  mkdir build
  cd build
  export CC=<my-favorite-C-compiler>
  export CXX=<my-favorite-C++-compiler>
  cmake ../llvm/flang -DCMAKE_PREFIX_PATH=<llvm-instal-root> -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=X86 -DCMAKE_CXX_STANDARD=17 <other-arguments>
  make
  make check-flang
```

# How to Generate FIR Documentation

It  is possible  to generate  FIR language  documentation by  running `make
flang-doc`.   This  will   create  `docs/Dialect/FIRLangRef.md` in  build
directory.
