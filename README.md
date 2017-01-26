# Aij
A small, simple Java AI library with minimal requirements.

By default, this library depends only on OpenCL.  If you remove the GPUGraph.java class and strip the OpenCL requirement from the build.gradle file, it should run just fine.  Methods are first implemented in the CPUGraph class before being ported to the GPUGraph, so they should be approximately at parity.

## TODO:

1. Input and Variable aren't supported as right-hand-side operators to power or exp functions
1. Operator broadcasting of inputs
1. Examples and better documentation.

## Done:

1. [ 238b71294fbd6743ab8440daa5fd4b2ace38479e ] JSON Serialization of Graphs (?) Text serialization, but whatever.
1. [ 6fc2a6c1835359d714028237a9827f029e90a537 ] Convolution
1. [ 2cfdf17cf802fb46271ff17ade147c007b952322 ] Refactor how nodes are handled in the graph so we don't have a single huge file
1. [ 2cfdf17cf802fb46271ff17ade147c007b952322 ] In-memory variables so we don't have to keep pushing stuff via inputs
1. [ 2cfdf17cf802fb46271ff17ade147c007b952322 ] Constants (depends on Variables)


## Operators and Basic Nodes

- [x] ABS
- [x] ADD
- [x] CONVOLVE3
- [X] DECONVOLVE3
- [x] EXP
- [x] INPUT
- [x] INVERT
- [x] LOG
- [x] MATMUL
- [x] MULTIPLY
- [x] NEGATE
- [ ] POOL
- [x] POWER
- [x] SIGMOID
- [x] SUBTRACT
- [x] TANH
- [ ] TRACE

## Known Issues

### Backprop in Softmax

I need to figure out the calculus behind softmax in the reverse direction.  

### Backprop in Convolution

I think there's a bug in Conv2D.  One of my applications' losses isn't decreasing when Conv layers are present.  Need to investigate.


