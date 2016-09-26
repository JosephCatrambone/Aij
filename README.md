# Aij
A small, simple Java AI library with minimal requirements.

By default, this library depends only on OpenCL.  If you remove the GPUGraph.java class and strip the OpenCL requirement from the build.gradle file, it should run just fine.  Methods are first implemented in the CPUGraph class before being ported to the GPUGraph, so they should be approximately at parity.

## TODO:

1. Input and Variable aren't supported as right-hand-side operators to power or exp functions
1. Refactor how nodes are handled in the graph so we don't have a single huge file
1. JSON Serialization of Graphs (?)
1. Operator broadcasting of inputs
1. In-memory variables so we don't have to keep pushing stuff via inputs
1. Constants (depends on Variables)
1. Convolution

## Operators and Basic Nodes

- [x] ABS
- [x] ADD
- [ ] CONVOLVE3
- [ ] DECONVOLVE3
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
- [ ] TRANSPOSE

## Considerations

### Broadcasting.

Right now I have no idea how I want to implement broadcasting.  It's an open problem.

One idea might be to have a 'RESIZE' command in the graph, so when we're doing training, we can resize elements to specific dimension, then when running set them back to other sizes.

### Variable + Update nodes.

When using the library, I found myself getting sick of doing inputs.replace(arg, float[]{}) when new values came in.  More so, I got sick of reading into arrays and then assigning them back to the inputs, ESPECIALLY for constants like exponents.

If I include a 'variable' node, I'm going to need an 'update' node which assigns a value to a variable.  That should save a memory copy in the GPU case and make things run much faster.

### Training Word2Vec.

I'm hitting a small issue with word2vec.  I don't want to have to build a whole extra graph to do the decoding, so it would be nice to have a 'set internal state' method, but that's also a hassle and makes things messy.  

Currently, a Wikipedia sample with 22149 words and 10000 sentences (window size 5, batch size 1) takes about 10053ms to train per batch on the CPU.

