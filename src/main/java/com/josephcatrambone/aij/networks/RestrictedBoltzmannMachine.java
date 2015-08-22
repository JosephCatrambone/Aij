package com.josephcatrambone.aij.networks;

import com.josephcatrambone.aij.Matrix;

import java.io.Serializable;
import java.util.Random;

/**
 * Created by jcatrambone on 5/28/15.
 */
public class RestrictedBoltzmannMachine implements Network, Serializable {
	final double ACTIVE_STATE = 1.0;
	final double INACTIVE_STATE = 0.0;

	Random random;
	Matrix weights;

	public RestrictedBoltzmannMachine(int numVisible, int numHidden) {
		random = new Random();
		weights = Matrix.random(numVisible, numHidden);
	}

	public RestrictedBoltzmannMachine(int numVisible, int numHidden, String layerType) {
		weights = Matrix.random(numVisible, numHidden);
	}

	@Override
	public Matrix predict(Matrix input) {
		return input.multiply(weights).elementOp_i(v -> v > random.nextDouble() ? ACTIVE_STATE : INACTIVE_STATE);
	}

	@Override
	public Matrix reconstruct(Matrix output) {
		return reconstruct(output, true);
	}

	public Matrix reconstruct(Matrix output, boolean stochastic) {
		Matrix result = output.multiply(weights.transpose());
		if(stochastic) {
			return result.elementOp_i(v -> v > random.nextDouble() ? ACTIVE_STATE : INACTIVE_STATE);
		} else {
			return result;
		}
	}

	@Override
	public int getNumInputs() {
		return weights.numRows();
	}

	@Override
	public int getNumOutputs() {
		return weights.numColumns();
	}

	@Override
	public int getNumLayers() {
		return 1;
	}

	@Override
	public Matrix getWeights(int i) {
		return weights;
	}

	@Override
	public void setWeights(int i, Matrix weights) {
		this.weights = weights;
	}

	public Matrix daydream(int numSamples, int numCycles) {
		return daydream(numSamples, numCycles, true, true);
	}

	public Matrix daydream(int numSamples, int numCycles, boolean binaryIntermediate, boolean binaryTerminal) {
		// Do numCycles gibbs samples to produce numSample sampels.
		Matrix input = Matrix.random(numSamples, getNumInputs());
		for(int i=0; i < numCycles; i++) {
			input = reconstruct(predict(input), binaryIntermediate);
		}

		return reconstruct(predict(input), binaryTerminal);
	}
}
