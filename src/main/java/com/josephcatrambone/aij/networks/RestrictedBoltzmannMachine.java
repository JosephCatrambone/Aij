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
		final Matrix biasedInput = visibleBias.repmat(input.numRows(), 1).add(input);
		final Matrix activities = biasedInput.multiply(weights);
		activities.elementOp_i(v -> v > random.nextDouble() ? ACTIVE_STATE : INACTIVE_STATE);
		return hiddenBias.repmat(input.numRows(), 1).add(activities);
	}

	@Override
	public Matrix reconstruct(Matrix output) {
		return reconstruct(output, false);
	}

	public Matrix reconstruct(Matrix output, boolean stochastic) {
		final Matrix biasedOutput = output.subtract(hiddenBias.repmat(output.numRows(), 1));
		final Matrix result = (biasedOutput.multiply(weights.transpose())).subtract(visibleBias.repmat(output.numRows(), 1));
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
