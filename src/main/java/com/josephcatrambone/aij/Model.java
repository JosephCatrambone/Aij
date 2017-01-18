package com.josephcatrambone.aij;

import com.josephcatrambone.aij.nodes.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

/**
 * Created by josephcatrambone on 1/17/17.
 * Model is a high-level wrapper for graph which has some common uses, like making dense layers and conv layers.
 */
public class Model extends Graph {
	public enum Optimizer { SGD, MOMENTUM, ADAGRAD };
	public enum Activation { NONE, TANH, SIGMOID, RELU };
	public enum Loss { ABS, SQUARED };

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
		w.setVariable(new Matrix(rows, columns, (i,j) -> (2.0f*random.nextFloat())-1.0f));
		trainableVariables.add(w);
		return w;
	}

	private VariableNode xavierWeight(int rows, int columns) {
		VariableNode w = new VariableNode(rows, columns);
		// Xavier says 2 + (n_in + n_out).
		float scaling = 2.0f / (float)rows; // Based on a recent paper by He, Rang, Zhen, and Sun.
		w.setVariable(new Matrix(rows, columns, (i,j) -> scaling*(float)random.nextGaussian()));
		trainableVariables.add(w);
		return w;
	}

	public void fit(float[] x, float[] y, float learningRate, Loss loss) {
		// If this is the first time we've run fit, we'll need to make our loss node.
		if(targetNode == null || lossNode == null) {
			// TODO: Sanity check if target node size changed.
			targetNode = new InputNode(1, y.length);
			Node diff = new SubtractNode(outputNode, targetNode);
			switch(loss) {
				case ABS:
					lossNode = new AbsNode(diff);
					break;
				case SQUARED:
					lossNode = new PowerNode(diff, 2.0f);
					break;
			}
			addNode(lossNode);
		}

		// Calculate the difference and apply the gradient.
		HashMap<Node, Matrix> inputFeed = new HashMap<>();
		inputFeed.put(inputNode, new Matrix(inputNode.rows, inputNode.columns, x));
		inputFeed.put(targetNode, new Matrix(targetNode.rows, targetNode.columns, y));
		Matrix[] grads = getGradient(inputFeed, null, lossNode);

		// Apply the gradients, scaled, to each of the learning variables.
		for(VariableNode n : trainableVariables) {
			n.getVariable().elementOp_i(grads[n.id], (w, dw) -> w - learningRate*dw);
		}
	}

	public void fit(float[][] x, float [][] y, float learningRate, Loss loss) {
		for(int i=0; i < x.length; i++) {
			fit(x[i], y[i], learningRate, loss);
		}
	}

	public float[] predict(float[] x) {
		HashMap<Node, float[]> inputMap = new HashMap<>();
		inputMap.put(inputNode, x);
		return getOutput(inputMap, outputNode);
	}

	public float[][] predict(float[][] x) {
		float[][] result = new float[x.length][outputNode.rows*outputNode.columns];
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
				throw new RuntimeException("Not implemented.");
			case SIGMOID:
				return new SigmoidNode(n);
			case TANH:
				return new TanhNode(n);
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

	public void addConvLayer(int kernelWidth, int kernelHeight, int padding, int stride, Activation act) {
		VariableNode kernel = xavierWeight(kernelHeight, kernelWidth);
		Node conv = new Convolution2DNode(outputNode, kernel, stride, padding);
		VariableNode bias = randomWeight(conv.rows, conv.columns);
		Node prod = new AddNode(conv, bias);
		outputNode = makeActivationNode(prod, act);
		addNode(outputNode);
	}

	public void addFlattenLayer() {
		outputNode = new ReshapeNode(outputNode, 1, -1);
		addNode(outputNode);
	}
}
