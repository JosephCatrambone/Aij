# Aij
A small, simple Java AI library with minimal requirements.

By default, this library depends only on OpenCL.  If you remove the GPUGraph.java class and strip the OpenCL requirement from the build.gradle file, it should run just fine.  Methods are first implemented in the CPUGraph class before being ported to the GPUGraph, so they should be approximately at parity.

## TODO:

1. JSON Serialization of Graphs
1. Operator broadcasting of inputs
1. In-memory variables so we don't have to keep pushing stuff via inputs
1. Constants (depends on Variables)
1. Convolution
1. Input and Variable aren't supported as right-hand-side operators to power or exp functions.

## Operators and Basic Nodes

- [x] ABS
- [x] ADD
- [x] EXP
- [x] INPUT
- [x] INVERT
- [x] LOG
- [x] MATMUL
- [x] MULTIPLY
- [x] NEGATE
- [x] POWER
- [x] SIGMOID
- [x] SUBTRACT
- [x] TANH
- [ ] TRACE
- [ ] TRANSPOSE
- [ ] VARIABLE
