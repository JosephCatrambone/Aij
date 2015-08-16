package com.josephcatrambone.aij.trainers;

import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.layers.Layer;
import com.josephcatrambone.aij.networks.Network;
import com.josephcatrambone.aij.networks.NeuralNetwork;
import com.josephcatrambone.aij.networks.RestrictedBoltzmannMachine;

import java.util.Arrays;
import java.util.Random;

/**
 * Created by jcatrambone on 5/29/15.
 */
public class RBMTrainer implements Trainer {
	Random random = new Random();
	public int notificationIncrement = 0; // If > 0, when iter%notificationIncrement == 0, run notification func.
	public int maxIterations = 10000; // If we do more iterations than this, stop.
	public double earlyStopError = 0.0; // If error is less than this, stop early.
	public int batchSize = 1;
	public double learningRate = 0.1;
	public double momentum = 0.0;
	public double lastError = Double.MAX_VALUE;
	//public int gibbsSamples = 1; // 'k' in literature.  1 is almost always ideal.

	private void log(String msg) {
		System.out.println(msg);
	}

	public void train(Network net, Matrix inputs, Matrix labels, Runnable notification) {
		RestrictedBoltzmannMachine rbm = (RestrictedBoltzmannMachine)net; // We only train neural networks.
		final Matrix weights = rbm.getWeights(0);
		int[] sampleIndices = new int[batchSize];

		for(int i=0; i < maxIterations && lastError > earlyStopError; i++) {
			// Randomly sample input matrix and examples.
			Arrays.parallelSetAll(sampleIndices, x -> random.nextInt(inputs.numRows()));

			final Matrix x = inputs.getRows(sampleIndices);

			// Positive CD phase.
			final Matrix positive_hidden_activations = x.multiply(weights);
			final Matrix positive_hidden_probabilities = positive_hidden_activations.sigmoid();
			final Matrix positive_hidden_states = positive_hidden_probabilities.elementOp(v -> v > random.nextDouble() ? 1.0 : 0.0);

			final Matrix positive_product = x.transpose().multiply(positive_hidden_probabilities); // pos_associations = np.dot(data.T, pos_hidden_probs)

			// Negative CD phase.
			// Reconstruct the visible units and sample again from the hidden units.
			final Matrix negative_visible_activities = positive_hidden_states.multiply(weights.transpose()); //neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
			final Matrix negative_visible_probabilities = negative_visible_activities.sigmoid(); //neg_visible_probs = self._logistic(neg_visible_activations)
			final Matrix negative_hidden_activities = negative_visible_probabilities.multiply(weights); //neg_hidden_probs = self._logistic(neg_hidden_activations)
			final Matrix negative_hidden_probabilities = negative_hidden_activities.sigmoid();

			final Matrix negative_product = negative_visible_probabilities.transpose().multiply(negative_hidden_probabilities);//neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

			// Update weights.
			weights.add(positive_product.subtract(negative_product).elementMultiply_i(learningRate/(float)batchSize));
			// TODO: Recheck these bias updates.  I think they're wrong.
			//rbm.getVisible().setBias(rbm.getVisible().getBias().add(x.subtract(negative_visible_probabilities).meanRow().elementMultiply_i(learningRate/(float)batchSize)));
			//rbm.getHidden().setBias(rbm.getHidden().getBias().add(positive_hidden_probabilities.subtract(negative_hidden_probabilities).meanRow().elementMultiply_i(learningRate/(float)batchSize)));
			lastError = x.subtract(negative_visible_probabilities).elementOp_i(v -> v*v).sum();

			if(notification != null && notificationIncrement > 0 && (i+1)%notificationIncrement == 0) {
				notification.run();
			}
		}

		rbm.setWeights(0, weights);
	}
}
