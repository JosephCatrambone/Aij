package com.josephcatrambone.aij.trainers;

import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.networks.Network;
import com.josephcatrambone.aij.networks.RestrictedBoltzmannMachine;

import java.util.Random;

/**
 * Created by jcatrambone on 5/29/15.
 */
public class ContrastiveDivergenceTrainer implements Trainer {
	Random random = new Random();
	public int notificationIncrement = 0; // If > 0, when iter%notificationIncrement == 0, run notification func.
	public int maxIterations = 10000; // If we do more iterations than this, stop.
	public double earlyStopError = 0.0; // If error is less than this, stop early.
	public int batchSize = 1;
	public double learningRate = 0.1;
	public double regularization = 0.001; // L2
	public double dropout = 0.0;
	public double momentum = 0.0;
	public double lastError = Double.MAX_VALUE;
	public int gibbsSamples = 1; // 'k' in literature.  1 is almost always ideal.

	private void log(String msg) {
		System.out.println(msg);
	}

	public void train(Network net, Matrix inputs, Matrix labels, Runnable notification) {
		RestrictedBoltzmannMachine rbm = (RestrictedBoltzmannMachine)net; // We only train neural networks.
		final Matrix weights = rbm.getWeights(0);
		final Matrix visibleBias = rbm.getVisibleBias();
		final Matrix hiddenBias = rbm.getHiddenBias();
		int[] sampleIndices = new int[batchSize];

		for(int i=0; i < maxIterations && lastError > earlyStopError; i++) {
			// Randomly sample input matrix and examples.
			//Arrays.parallelSetAll(sampleIndices, x -> random.nextInt(inputs.numRows()));
			for(int j=0; j < batchSize; j++) {
				sampleIndices[j] = random.nextInt(inputs.numRows());
			}

			final Matrix batch = inputs.getRows(sampleIndices);
			Matrix positiveHiddenActivations = null;
			Matrix positiveHiddenProbabilities = null;
			Matrix positiveHiddenStates = null;
			Matrix positiveProduct = null;
			Matrix negativeVisibleActivities = null;
			Matrix negativeVisibleProbabilities = null;
			Matrix negativeHiddenActivities = null;
			Matrix negativeHiddenProbabilities = null;
			Matrix negativeProduct = null;
			Matrix x = batch;

			final Matrix vBiasBlock = visibleBias.repmat(batchSize, 1);
			final Matrix hBiasBlock = hiddenBias.repmat(batchSize, 1);

			for(int k=0; k < gibbsSamples; k++) {
				// Positive CD phase.
				positiveHiddenActivations = x.multiply(weights);
				positiveHiddenProbabilities = hBiasBlock.add(positiveHiddenActivations).sigmoid();
				positiveHiddenStates = positiveHiddenProbabilities.elementOp(
						v -> v > random.nextDouble() ? RestrictedBoltzmannMachine.ACTIVE_STATE : RestrictedBoltzmannMachine.INACTIVE_STATE);

				if(k == 0) {
					positiveProduct = x.transpose().multiply(positiveHiddenStates);
				}

				// Negative CD phase.
				// Reconstruct the visible units and sample again from the hidden units.
				// Hinton: "When the hidden units are being driven by data, always use stochastic binary states.
				// When they are being driven by reconstructions, always use probabilities without sampling."
				negativeVisibleActivities = positiveHiddenStates.multiply(weights.transpose());
				negativeVisibleProbabilities = vBiasBlock.add(negativeVisibleActivities).sigmoid();
				negativeHiddenActivities = negativeVisibleProbabilities.multiply(weights);
				negativeHiddenProbabilities = hBiasBlock.add(negativeHiddenActivities).sigmoid();

				negativeProduct = negativeVisibleProbabilities.transpose().multiply(negativeHiddenProbabilities);

				x = negativeVisibleProbabilities;
			}

			// Update weights.
			weights.add_i(positiveProduct.subtract(negativeProduct).elementMultiply(learningRate / (float) batchSize));
			visibleBias.add_i(batch.subtract(negativeVisibleProbabilities).meanRow().elementMultiply(learningRate));
			hiddenBias.add_i(positiveHiddenProbabilities.subtract(negativeHiddenProbabilities).meanRow().elementMultiply(learningRate));
			lastError = batch.subtract(negativeVisibleProbabilities).elementOp_i(v -> v*v).sum()/(float)batchSize;

			if(notification != null && notificationIncrement > 0 && (i+1)%notificationIncrement == 0) {
				notification.run();
			}
		}

		final Matrix l1Delta = weights.elementMultiply(regularization);
		weights.subtract_i(l1Delta);

		rbm.setWeights(0, weights);
		rbm.setVisibleBias(visibleBias);
		rbm.setHiddenBias(hiddenBias);
	}
}
