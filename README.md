# Aij
A small, simple Java AI library with minimal requirements.

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

## Layer

You probably don't need to worry about layers, since the networks will take care of them most of the time.
A layer is like a double[], insofar as it maintains activity.  It's different, however,
because it maintains multiple activations, activities, and the derivatives of a propagation at the same time.
A layer also contains (perhaps incorrectly) the bias term.

# Example Use

```
// Make our data
Matrix x = new Matrix(10, 1);
x.set(0, 0, 0);
x.set(1, 0, 0.1);
// ...
// There are a bunch of ways to set values.
Matrix y = x.elementOp(v -> Math.sin(v)); // Run Math.sin(v) on every element of x to make a new matrix.

// Make our network
NeuralNetwork nn = new NeuralNetwork(new int[]{1, 3, 1}, new String[]{"linear", "tanh", "linear"});

// Make our trainer
BackpropTrainer trainer = new BackpropTrainer();

// Train
trainer.train(nn, x, y);
```

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

* Softmax/Downsampling Network (Maybe I'll subclass 1:1 for this)
* LSTM Recurrent Network + Trainer
* Gated Feedback Network + Trainer
* Nice tool to display the RBM activations.

# Why did you?

## Why make trainer a separate type?  Why not have a train method on each of the networks.

By making the network trainers separate from the networks themselves, it allows us to hot swap the trainers.  
You may have a backprop trainer that you use most of the time, and an Adagrad trainer that you use to impress 
undergraduates.

## Why make layer a separate type?

Originally, networks were just maintaining their own activations, preactivations, and baises for each layer, 
but there was so much overlap it made sense to split out the layer type.

## Why make a matrix type instead of reusing the jBlas DoubleMatrix.

The matrix class is a very thin wrapper which was done because we might want to swap out jblas in the future.

## Why make the 'OneToOne' network?  What is it?

A one to one network is a network with uses a function to calculate the output value instead of any weights.  It also has a function to 'listen in' as the predict and reconstruct operators are called.  That might seem silly, but it's helpful for me to be able to see what is being passed through a network when predict and reconstruct are called.  It also helps when making the convolution trainer, since I can write convolve once and then use this method to grab training data from the network.  I may, in a future release, rename this FunctionNetwork.
