package com.josephcatrambone.aij.networks;

import com.josephcatrambone.aij.Matrix;

import java.io.Serializable;
import java.util.function.UnaryOperator;

/**
 * Created by jcatrambone on 5/28/15.
 */
public class NeuralNetwork implements Network, Serializable {
	public static double WEIGHT_SCALE = 0.1;

	private Matrix[] weights;
	private UnaryOperator<Double>[] activationFunctions;
	private UnaryOperator<Double>[] derivativeFromActivationFunctions; // Given c, where c = f(x), compute f^-1(x).
	// For sigmoid = 1/(1 + e^(-x)) = f(x), dsigmoid(w) = w*(1-w) for w=f(x)
	// This isn't the derivative from X, it's the derivative from the activation of X.
	// DON'T SCREW THIS UP.  AGAIN.

	public NeuralNetwork(int[] layerSizes, String[] activationFunctions) {
		weights = new Matrix[layerSizes.length-1];
		this.activationFunctions = new UnaryOperator[layerSizes.length];
		this.derivativeFromActivationFunctions = new UnaryOperator[layerSizes.length];

		for(int i=0; i < weights.length; i++) {
			weights[i] = Matrix.random(layerSizes[i], layerSizes[i + 1]);
			weights[i].elementMultiply_i(WEIGHT_SCALE);
			switch(activationFunctions[i]) { // Java 8 and higher support switch/case.
				case "tanh":
					this.activationFunctions[i] = v -> Math.tanh(v);
					this.derivativeFromActivationFunctions[i] = v -> 1.0 - (v*v); // 1 - tanh^2. v = tanh()
					break;
				case "logistic":
				case "sigmoid":
					this.activationFunctions[i] = v -> 1.0/(1.0+Math.exp(-v));
					this.derivativeFromActivationFunctions[i] = v -> v*(1.0-v);
					break;
				case "linear":
					this.activationFunctions[i] = v -> v;
					this.derivativeFromActivationFunctions[i] = v -> 1.0;
					break;
				default:
					throw new IllegalArgumentException("Unrecognized activation function in NN: " + activationFunctions[i]);
			}
		}
	}

	/*** forwardPropagate
	 * Given an input, return an array of the activations from every level.
	 * @param input
	 * @return
	 */
	public Matrix[] forwardPropagate(Matrix input) {
		Matrix[] results = new Matrix[weights.length+1];
		results[0] = input.elementOp(activationFunctions[0]);
		for(int i=0; i < weights.length; i++) {
			results[i+1] = results[i].multiply(weights[i]).elementOp_i(activationFunctions[i+1]);
		}
		return results;
	}

	public Matrix[] backPropagate(Matrix output) {
		Matrix[] results = new Matrix[weights.length+1];
		results[weights.length+1] = output;
		for(int i=weights.length+1; i > 0; i--) {
			results[i-1] = results[i].multiply(weights[i - 1].transpose());
		}
		return results;
	}

	@Override
	public Matrix predict(Matrix input) {
		Matrix[] results = forwardPropagate(input);
		return results[results.length-1];
	}

	@Override
	public Matrix reconstruct(Matrix output) {
		return null;
	}

	@Override
	public int getNumInputs() {
		return weights[0].numRows();
	}

	@Override
	public int getNumOutputs() {
		return weights[weights.length].numColumns();
	}

	@Override
	public Matrix getWeights(int layer) {
		return weights[layer];
	}

	@Override
	public void setWeights(int layer, Matrix weights) {
		this.weights[layer] = weights;
	}

	@Override
	public int getNumLayers() {
		return weights.length+1;
	}

	public UnaryOperator getActivationFunction(int i) {
		return activationFunctions[i];
	}

	public UnaryOperator getDerivativeFunction(int i) {
		return derivativeFromActivationFunctions[i];
	}

}
