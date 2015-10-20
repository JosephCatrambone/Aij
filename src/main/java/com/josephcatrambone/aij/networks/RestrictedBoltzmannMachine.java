package com.josephcatrambone.aij.networks;

import com.josephcatrambone.aij.Matrix;

import java.io.Serializable;
import java.util.Random;

/**
 * Created by jcatrambone on 5/28/15.
 */
public class RestrictedBoltzmannMachine implements Network, Serializable {
	public static final double ACTIVE_STATE = 1.0;
	public static final double INACTIVE_STATE = 0.0;
	public static double INITIAL_WEIGHT_SCALE = 0.1;

	Random random;
	Matrix weights;
	Matrix visibleBias;
	Matrix hiddenBias;

	public RestrictedBoltzmannMachine(int numVisible, int numHidden) {
		random = new Random();
		weights = Matrix.random(numVisible, numHidden);
		weights.elementMultiply_i(INITIAL_WEIGHT_SCALE);
		visibleBias = Matrix.zeros(1, numVisible);
		hiddenBias = Matrix.zeros(1, numHidden);
	}

	@Override
	public Matrix predict(Matrix input) {
		final Matrix activities = input.multiply(weights);
		final Matrix activations = hiddenBias.repmat(input.numRows(), 1).add(activities).sigmoid();
		activations.elementOp_i(v -> v > random.nextDouble() ? ACTIVE_STATE : INACTIVE_STATE);
		return activations;
	}

	@Override
	public Matrix reconstruct(Matrix output) {
		return reconstruct(output, true, false);
	}

	public Matrix reconstruct(Matrix output, boolean activate, boolean stochastic) {
		// Push through weights.
		Matrix result = output.multiply(weights.transpose());
		// Bias before sigmoid.
		result.add_i(visibleBias.repmat(output.numRows(), 1));
		// If we want stochastic/binary reconstruction, we have to activate.
		if(activate || stochastic) {
			result.sigmoid_i();
		}
		if(stochastic) {
			result.elementOp_i(v -> v > random.nextDouble() ? ACTIVE_STATE : INACTIVE_STATE);
		}
		return result;
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

	public Matrix getVisibleBias() {
		return visibleBias;
	}

	public void setVisibleBias(Matrix newBias) {
		this.visibleBias = newBias;
	}

	public Matrix getHiddenBias() {
		return hiddenBias;
	}

	public void setHiddenBias(Matrix newBias) {
		this.hiddenBias = newBias;
	}

	public Matrix daydream(int numSamples, int numCycles) {
		Matrix input = Matrix.random(numSamples, getNumInputs());
		return daydream(input, numCycles);
	}

	public Matrix daydream(Matrix input, int numCycles) {
		// Do numCycles gibbs samples to produce numSample sampels.
		for(int i=0; i < numCycles; i++) {
			input = reconstruct(predict(input), true, true);
		}
		//input = reconstruct(predict(input), true, true);

		return input;
	}

	public double getFreeEnergy() {
		double accumulator = 0;
		for(int i=0; i < visibleBias.numColumns(); i++) {
			for(int j=0; j < hiddenBias.numColumns(); j++) {
				accumulator += visibleBias.get(0, i) * hiddenBias.get(0, j) * weights.get(i, j);
			}
		}
		return -accumulator;
	}
}
