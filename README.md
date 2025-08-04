# CIRCT plugin pass example

## Preparation

### Dependencies

Build CIRCT at the same level as this example:

```sh
cd ..
git clone https://github.com/llvm/circt
cd circt
git checkout f1213ba8bdbcaba2a4903c5e078fb8c223a2ab39
git submodule update --init
cd llvm
mkdir build && cd build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_USE_SPLIT_DWARF=ON \
    -DLLVM_ENABLE_LLD=ON
ninja
cd ../..
mkdir build && cd build
cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_USE_SPLIT_DWARF=ON \
    -DLLVM_ENABLE_LLD=ON \
    -DCIRCT_SLANG_FRONTEND_ENABLED=ON
ninja
```
## Part 1: Pass plugin

### Build

Build the pass as a shared library:

```sh
mkdir build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
ninja
cd ..
```

### Run `circt-opt` with custom pass

```sh
../circt/build/bin/circt-opt -load-pass-plugin=build/CombAddOptimize.so -pass-pipeline='builtin.module(reduce-comb-add-width)' test/basic.mlir
```

### Exercise

In this part, we want to optimize `AddOp`.
This optimization targets to reduce the width of the operation.
To make this happen, we look at the inputs of the operation and check the value
ranges they have. If the output width has a size bigger than the actual value
range, we can replace this operation with a new, smaller, `AddOp`.

An example why this works:

```verilog
logic [2:0] a;        \\ Value range: 0-7
logic [1:0] b, c;     \\ Value range: 0-3
logic [3:0] ab;       \\ Value range: 0-15
logic [4:0] abc;      \\ Value range: 0-31

assign ab = a + b;    \\ ab range: 0 - 10 (7+3)
assign abc = ab + c;  \\ abc range: 0 - 13 (10+3)
```

We are writing our own pass plugin, which will optimize the size of the last
addition.
The code for this is in `CombAddOptimize.cpp`.

The pass is defined in `include/CombAddOptimize/CombAddOptimize.td`.
This is a tablegen file, which is used in LLVM to serve as a code generator
framework. By defining our pass in this format, tablegen will create the
boilerplate code for us. This can be found in the directory
`build/include/CombAddOptimize/`, after the build was started.

## Part 2: Chisel generator

The previous pass only makes sense if we have a lot of addition, which we can
not all check by hand.
We are now using Chisel, a hardware construction language, to create a larger
test case for our pass.
