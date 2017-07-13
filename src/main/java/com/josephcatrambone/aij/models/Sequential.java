package com.josephcatrambone.aij;

import com.josephcatrambone.aij.nodes.*;
import com.josephcatrambone.aij.optimizers.Momentum;
import com.josephcatrambone.aij.optimizers.Optimizer;
import com.josephcatrambone.aij.optimizers.SGD;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * Created by josephcatrambone on 1/17/17.
 * Model is a high-level wrapper for graph which has some common uses, like making dense layers and conv layers.
 */
public class Model extends Graph {
	//public enum Optimizer { SGD, MOMENTUM, ADAGRAD };
	public enum Activation { NONE, TANH, SIGMOID, RELU, SOFTMAX };
	public enum Loss { ABS, SQUARED, BINARY_CROSS_ENTROPY, SOFTMAX_CROSS_ENTROPY };

	private Random random; // Used for weight init.

	private Node inputNode; // Only one input to our compute graph.
	private Node outputNode; // Keeps track of the last later.

	// Used for training.
	private Node targetNode;
	private Node lossNode;
	private List<VariableNode> trainableVariables;

	public Model(int inputRows, int inputColumns) {
		super();
		random = new Random();
		inputNode = new InputNode(inputRows, inputColumns); // Gotta' resize.
		outputNode = inputNode;
		trainableVariables = new ArrayList<>();
	}

	private VariableNode randomWeight(int rows, int columns) {
		VariableNode w = new VariableNode(rows, columns);
		w.setVariable(new Matrix(rows, columns, (i,j) -> (2.0f*random.nextDouble())-1.0f));
		w.name = "variable";
		trainableVariables.add(w);
		return w;
	}

	private VariableNode xavierWeight(int rows, int columns) {
		VariableNode w = new VariableNode(rows, columns);
		// Xavier says 2 + (n_in + n_out).
		double scaling = 2.0f / (double)rows; // Based on a recent paper by He, Rang, Zhen, and Sun.
		w.setVariable(new Matrix(rows, columns, (i,j) -> scaling*(double)random.nextGaussian()));
		w.name = "variable";
		trainableVariables.add(w);
		return w;
	}

	@Override
	public void restoreFromString(String s) {
		super.restoreFromString(s);
		// We populate the names with some special tags when we start training.
		// Then we restore and use the names to put back the values.
		for(Node n : this.nodes) {
			if(n.name.startsWith("variable")) {
				this.trainableVariables.add((VariableNode)n);
			} else if(n.name.startsWith("input")) {
				this.inputNode = n;
			} else if(n.name.startsWith("output")) {
				this.outputNode = n;
			} else if(n.name.startsWith("loss")) {
				this.lossNode = n;
			} else if(n.name.startsWith("target")) {
				this.targetNode = n;
			}
		}
	}

	public Node getOutputNode() {
		return outputNode;
	}

	private void finalizeNetwork(Loss loss) {
		// If this is the first time we've run fit, we'll need to make our loss node.
		if(targetNode == null || lossNode == null) {
			// TODO: Sanity check if target node size changed.
			targetNode = new InputNode(outputNode.rows, outputNode.columns);
			Node diff = new SubtractNode(outputNode, targetNode);
			switch(loss) {
				case ABS:
					lossNode = new AbsNode(diff);
					break;
				case SQUARED:
					lossNode = new PowerNode(diff, 2.0f);
					break;
				case BINARY_CROSS_ENTROPY:
					// Binary XENT = target*log(pred) + (1-target)*log(1-pred)
					//Node flatTarget = new ReshapeNode(targetNode, 1, -1);
					//Node flatOutput = new ReshapeNode(outputNode, 1, -1);
					// TODO: This has issues with AOOB exceptions. I think perhaps RowSumNode is screwing with gradients.
					lossNode = new AddNode(
						new MultiplyNode(targetNode, new LogNode(outputNode)), // target*log(pred)
						new MultiplyNode(new SubtractNode(new ConstantNode(1.0, targetNode), targetNode), new LogNode(new SubtractNode(new ConstantNode(1.0, outputNode), outputNode)))
					);
					break;
			}
			lossNode = new CollapseSumNode(lossNode); // Roll up into a single value.
			addNode(lossNode);
			// Need these for save/restore.
			inputNode.name = "input";
			outputNode.name = "output";
			targetNode.name = "target";
			lossNode.name = "loss";
		}
	}

	public void fit(double[] x, double[] y, double learningRate, Loss loss) {
		assert(x.length == this.inputNode.rows*this.inputNode.columns);
		finalizeNetwork(loss);

		// Calculate the difference and apply the gradient.
		HashMap<Node, Matrix> inputFeed = new HashMap<>();
		inputFeed.put(inputNode, new Matrix(inputNode.rows, inputNode.columns, x));
		inputFeed.put(targetNode, new Matrix(targetNode.rows, targetNode.columns, y));

		// Minimize loss
		Optimizer optimizer = new Momentum(this, this.trainableVariables.toArray(new VariableNode[0]), learningRate, 0.5);
		optimizer.minimize(lossNode, inputFeed);
	}

	public void fit(double[][] x, double [][] y, double learningRate, Loss loss) {
		finalizeNetwork(loss);
		assert(x[0].length == this.inputNode.rows*this.inputNode.columns);

		Optimizer optimizer = new SGD(this, this.trainableVariables.toArray(new VariableNode[0]), learningRate);

		for(int i=0; i < x.length; i++) {
			// Calculate the difference and apply the gradient.
			HashMap<Node, Matrix> inputFeed = new HashMap<>();
			inputFeed.put(inputNode, new Matrix(inputNode.rows, inputNode.columns, x[i]));
			inputFeed.put(targetNode, new Matrix(targetNode.rows, targetNode.columns, y[i]));

			// Minimize loss
			optimizer.accumulateGradients(lossNode, inputFeed);
		}
		optimizer.applyGradients();
		optimizer.clearGradients();
	}

	/***
	 * fitBatch, unlike fit, calculates the gradient for all examples in parallel.
	 * @param x
	 * @param y
	 * @param learningRate
	 * @param loss
	 */
	public void fitBatch(double[][] x, double[][] y, double learningRate, Loss loss) {
		finalizeNetwork(loss);

		// This will accumulate our gradients below.
		Matrix[][] grads = new Matrix[x.length][nodes.size()];

		// Calculate all the gradients in parallel.
		IntStream.range(0, x.length).parallel().forEach(i -> {
			HashMap<Node, Matrix> inputFeed = new HashMap<>();
			inputFeed.put(inputNode, new Matrix(inputNode.rows, inputNode.columns, x[i]));
			inputFeed.put(targetNode, new Matrix(targetNode.rows, targetNode.columns, y[i]));
			grads[i] = getGradient(inputFeed, null, lossNode);
		});

		// Apply the gradients, scaled, to each of the learning variables.
		// Dividing by x.length will average our gradients.
		for(VariableNode n : trainableVariables) {
			Matrix accumulator = grads[0][n.id];
			for(int j=1; j < x.length; j++) {
				accumulator.elementOp_i(grads[j][n.id], (a,b) -> a+b);
			}
			n.getVariable().elementOp_i(accumulator, (w, dw) -> w - learningRate*dw/((float)x.length));
		}
	}

	public double[] predict(double[] x) {
		HashMap<Node, double[]> inputMap = new HashMap<>();
		inputMap.put(inputNode, x);
		return getOutput(inputMap, outputNode);
	}

	public double[][] predict(double[][] x) {
		double[][] result = new double[x.length][outputNode.rows*outputNode.columns];
		for(int i=0; i < x.length; i++) {
			result[i] = predict(x[i]);
		}
		return result;
	}

	private Node makeActivationNode(Node n, Activation act) {
		switch(act) {
			case NONE:
				return n;
			case RELU:
				return new ReLUNode(n);
			case SIGMOID:
				return new SigmoidNode(n);
			case TANH:
				return new TanhNode(n);
			case SOFTMAX:
				return new SoftmaxRowNode(n);
		}
		return null;
	}

	public void addDenseLayer(int hiddenSize, Activation act) {
		assert(outputNode.rows == 1); // TODO: Throw error.
		VariableNode w = xavierWeight(outputNode.columns, hiddenSize);
		VariableNode b = randomWeight(1, hiddenSize);
		Node prod = new AddNode(new MatrixMultiplyNode(outputNode, w), b);

		outputNode = makeActivationNode(prod, act);
		addNode(outputNode);
	}

	public void addConvLayer(int numFilters, int kernelHeight, int kernelWidth, int yStride, int xStride, Activation act) {
		VariableNode[] kernels = new VariableNode[numFilters];
		for(int i=0; i < numFilters; i++) {
			VariableNode kernel = xavierWeight(kernelHeight, kernelWidth);
			kernels[i] = kernel;
		}
		Node conv = new Convolution2DNode(outputNode, kernels, yStride, xStride);
		VariableNode bias = randomWeight(conv.rows, conv.columns);
		Node prod = new AddNode(conv, bias);
		outputNode = makeActivationNode(prod, act);
		addNode(outputNode);
	}

	public void addFlattenLayer() {
		outputNode = new ReshapeNode(outputNode, 1, -1);
		addNode(outputNode);
	}

	public void addReshapeLayer(int height, int width) {
		outputNode = new ReshapeNode(outputNode, height, width);
		addNode(outputNode);
	}

	public void addDeconvLayer(int numFilters, int kernelHeight, int kernelWidth, int yStride, int xStride, Activation act) {
		VariableNode[] kernels = new VariableNode[numFilters];
		for(int i=0; i < numFilters; i++) {
			VariableNode kernel = xavierWeight(kernelHeight, kernelWidth);
			kernels[i] = kernel;
		}
		Node deconv = new Deconvolution2DNode(outputNode, kernels, yStride, xStride);
		VariableNode bias = randomWeight(deconv.rows, deconv.columns); // TODO: Verify these dimensions are correct.
		Node prod = new AddNode(deconv, bias);
		outputNode = makeActivationNode(prod, act);
		addNode(outputNode);
	}

	/*** addLayer
	 * Assumes the user has already fetched the current output node.
	 * Pushes the new output node onto the stack of operations.
	 * @param node A node which takes the (former) output node and transforms it.  Will become the new output node.
	 */
	public void addLayer(Node node) {
		// TODO: Should we override ADD NODE?  It may cause confusion.
		outputNode = node;
		addNode(outputNode);
	}
}
