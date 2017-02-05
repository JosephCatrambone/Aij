# Aij
A small, simple Java AI library with no external requirements.

## What's New

v1.1: Lots of speed improvements!  Parallel matrix operations!  Breaking changes include moving to Double from Float for better Java 8 parallel/stream support.

v1.0: New high-level model interface.  Graph serialization.  LOTS of new node types like ReLU, Softmax, HStack, and Broadcast.

## Samples:

### Training XOR with the High-Level Wrapper:

```java
Model m = new Model(1, 2);
m.addDenseLayer(10, Model.Activation.TANH);
m.addDenseLayer(1, Model.Activation.SIGMOID);

double[][] x = new double[][] {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
};

double[][] y = new double[][] { {0}, {1}, {1}, {0} };

for(int i=0; i < 10000; i++) {
    m.fit(x, y, 0.5f, Model.Loss.SQUARED);
    System.out.println(m.predict(x[0])[0]);
    System.out.println(m.predict(x[1])[0]);
    System.out.println(m.predict(x[2])[0]);
    System.out.println(m.predict(x[3])[0]);
    System.out.println();
}

```

### Directly Utilizing Compute Graph:

```java
Graph g = new Graph();
InputNode x = new InputNode(1, 1);
VariableNode m = new VariableNode(1, 1);
VariableNode b = new VariableNode(1, 1);
InputNode y = new InputNode(1, 1); // Target.

Node out = new AddNode(b, new MultiplyNode(m, x));
Node error = new SubtractNode(y, out);
Node loss = new PowerNode(error, 2.0f);

g.addNode(loss);

// Try and approximate some linear function.
Random random = new Random();
double target_m = (float)random.nextGaussian()*100f;
double target_b = (float)random.nextGaussian()*100f;
m.setVariable(new Matrix(1, 1, new float[]{random.nextFloat()}));

// Do a few iterations.
final float LEARNING_RATE = 0.1f;
HashMap<Node, Matrix> inputFeed = new HashMap<>();
for(int i=0; i < 1000; i++) {
    double xData = random.nextDouble();
    inputFeed.put(x, new Matrix(1, 1, new double[]{xData}));
    inputFeed.put(y, new Matrix(1, 1, new double[]{xData*target_m + target_b}));
    // Minimize loss wrt error:
    Matrix[] grad = g.getGradient(inputFeed, null, loss);
    m.setVariable(m.getVariable().elementOp(d -> d-grad[m.id].data[0]*LEARNING_RATE));
    b.setVariable(b.getVariable().elementOp(d -> d-grad[b.id].data[0]*LEARNING_RATE));
}

System.out.println(" Expected: y = " + target_m + " * x + " + target_b);
System.out.println(" Got: y = " + m.getVariable().data[0] + " * x + " + b.getVariable().data[0]);

```

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


