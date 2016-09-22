# Aij
A small, simple Java AI library with minimal requirements.

By default, this library depends only on OpenCL.  If you remove the GPUGraph.java class and strip the OpenCL requirement from the build.gradle file, it should run just fine.  Methods are first implemented in the CPUGraph class before being ported to the GPUGraph, so they should be approximately at parity.

## TODO:

1. JSON Serialization of Graphs
1. Subtract Node
1. Finish Inverse Node
1. Operator broadcasting of inputs
1. In-memory variables so we don't have to keep pushing stuff via inputs
1. Constants
1. Convolution

## Operators

- [x] ABS
- [x] ADD
- [x] INPUT
- [ ] INVERT
- [ ] LOG
- [x] MATMUL
- [x] MULTIPLY
- [x] NEGATE
- [x] POWER
- [x] SIGMOID
- [ ] SUBTRACT
- [x] TANH
- [ ] TRACE
- [ ] TRANSPOSE
- [ ] VARIABLE
