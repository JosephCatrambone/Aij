# Aij
A small, simple Java AI library with minimal requirements.

# Features and Example Uses
## Approximating sin(x)

```
// Make our data
Matrix x = new Matrix(10, 1);
x.set(0, 0, 0);
x.set(1, 0, 0.1);
// ...
Matrix y = x.elementOp(v -> Math.sin(v)); // Run Math.sin(v) on every element of x to make a new matrix.

// Make our network
NeuralNetwork nn = new NeuralNetwork(new int[]{1, 3, 1}, new String[]{"linear", "tanh", "linear"});

// Make our trainer
BackpropTrainer trainer = new BackpropTrainer();

// Train
trainer.train(nn, x, y);
```

## Classic XOR

```
Matrix x = new Matrix(4, 2, 0.0);
x.setRow(0, new double[]{0, 0});
x.setRow(1, new double[]{0, 1});
x.setRow(2, new double[]{1, 0});
x.setRow(3, new double[]{1, 1});

Matrix y = new Matrix(4, 1, 0.0);
y.set(0, 0, 0.0);
y.set(1, 0, 1.0);
y.set(2, 0, 1.0);
y.set(3, 0, 0.0);

NeuralNetwork nn = new NeuralNetwork(new int[]{2, 3, 1}, new String[]{"tanh", "tanh", "tanh"});

BackpropTrainer trainer = new BackpropTrainer();
trainer.train(nn, x, y, null);

Matrix predictions = nn.predict(x);
```

## Word2Vec in less than 100 lines.

```
NeuralNetwork nn = new NeuralNetwork(new int[]{indexToWord.size(), HIDDEN_LAYER_SIZE, indexToWord.size()}, new String[]{"tanh", "tanh", "tanh"});
trainer.train(nn, examples, examples, null);

// How similar are "cat" and "dog"?

input.set(0, wordToIndex.get("cat"), 1.0); // Input is a single row with one column for each word.
Matrix catActivation = nn.forwardPropagate(input)[1]; // Go from a very sparse vector of words to our dense representation.

// Repeat for "dog" and "computer".

Matrix catDogDiff = catActivation.subtract(dogActivation);
double catDogDistance = catDogDiff.elementMultiply(catDogDiff).sum(); // Squared distance.
Matrix dogComputerDiff = dogActivation.subtract(computerActivation);
double dogComputerDistance = dogComputerDiff.elementMultiply(dogComputerDiff).sum();

System.out.println("cat vec: " + catActivation);
System.out.println("dog vec: " + dogActivation);
System.out.println("comp vec: " + computerActivation);
System.out.println("catDog dist: " + catDogDistance);
System.out.println("dogComp dist: " + dogComputerDistance);
```

Results:

```
cat vec: [0.139042, 0.111121, -0.291885, 0.147191, 0.348583]
dog vec: [0.079685, 0.118307, -0.309900, 0.124663, 0.381532]
comp vec: [0.160329, 0.048527, -0.211264, 0.231911, -0.086544]
catDog dist: 0.005492550166584046
dogComp dist: 0.25169884693665084
```

See TestNeuralNetworkTrainer.java for full source.

# Project Structure & Object Hierarchy
## Network

A network is a bunch of layers (or other networks) connected by weights.

## Trainer

A trainer is used with a network to impart weights.
Given some input examples and a networks (and, often, some labels), a trainer will adjust the weights of a network
and the biases of the layers to fit the data.  Some trainers only work on specific network types.

RBM Trainer -> RestrictedBoltzmannMachine Network

Backprop Trainer -> Neural Network

Convolutional Trainer -> Convnet

# Function conventions

## Network

predict() -> Returns the activated result of pushing data through the network.

reconstruct() -> Given the activated output, returns the input activities. (Not all networks can reconstruct.)
Note: For RBM, we return the activation of the input rather than the activity.  We do this because getActivities 
doesn't add the bias to the input, and it makes our reconstruction better.  Perhaps in the future, we'll make the  
default behavior return the activation instead of activity.

## Matrix

Functions ending with "_i" are mutating and will modify the host matrix.

## TODO

* LSTM Recurrent Network + Trainer
* Gated Feedback Network + Trainer
* Nice tool to display the RBM activations.
* Add a derivative function to networks so it's possible to backprop across multiple networks for fine-tuning.

# Why did you?

## Why make trainer a separate type?  Why not have a train method on each of the networks.

By making the network trainers separate from the networks themselves, it allows us to hot swap the trainers.  
You may have a backprop trainer that you use most of the time, and an Adagrad trainer that you use to impress 
undergraduates.

## Why make a matrix type instead of reusing the jBlas DoubleMatrix.

The matrix class is a very thin wrapper which was done because we might want to swap out jblas in the future.

## Why make the 'FunctionNetwork'?  What is it?

Sometimes it's useful to be able to define a network with a deterministic function.  A function network is one which takes an explicit function which maps the input to the output.  It also has a function to 'listen in' as the predict and reconstruct operators are called.  It's helpful (for debugging and convolution, among others) to be able to see what is being passed through a network when predict and reconstruct are called.
