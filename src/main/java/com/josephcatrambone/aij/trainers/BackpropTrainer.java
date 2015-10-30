package com.josephcatrambone.aij.trainers;

import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.networks.Network;
import com.josephcatrambone.aij.networks.NeuralNetwork;

import java.util.Arrays;
import java.util.Random;

/**
 * Created by jcatrambone on 5/29/15.
 */
public class BackpropTrainer implements Trainer {
	Random random = new Random();
	public int notificationIncrement = 0; // If > 0, when iter%notificationIncrement == 0, run notification func.
	public int maxIterations = 10000; // If we do more iterations than this, stop.
	public double earlyStopError = 0.0; // If error is less than this, stop early.
	public int batchSize = 1;
	public double learningRate = 0.1;
	public double momentum = 0.0;

	Matrix[] weightBlame;
	Matrix[] biasBlame;
	Matrix[] deltaWeights;

	private void log(String msg) {
		System.out.println(msg);
	}

	public void train(Network net, Matrix inputs, Matrix labels, Runnable notification) {
		NeuralNetwork nn = (NeuralNetwork)net; // We only train neural networks.
		weightBlame = new Matrix[nn.getNumLayers()];
		biasBlame = new Matrix[nn.getNumLayers()];
		deltaWeights = new Matrix[nn.getNumLayers()-1];
		double sumError = Double.MAX_VALUE;
		int[] sampleIndices = new int[batchSize];
		Matrix[] layerActivation = null;
		Matrix[] layerGradient = new Matrix[nn.getNumLayers()];

		// Init delta weights to zero
		for(int i=0; i < deltaWeights.length; i++) {
			deltaWeights[i] = Matrix.zeros(nn.getWeights(i).numRows(), nn.getWeights(i).numColumns());
		}

		for(int i=0; i < maxIterations; i++) {
			// Randomly sample input matrix and examples.
			Arrays.parallelSetAll(sampleIndices, x -> random.nextInt(inputs.numRows()));

			Matrix x = inputs.getRows(sampleIndices);

			layerActivation = nn.forwardPropagate(x);

			for(int j=0; j < nn.getNumLayers(); j++) {
				layerGradient[j] = layerActivation[j].elementOp(nn.getDerivativeFunction(j));
			}

			// Delta(L) = (activation - truth) dot (activity)
			// delta(L) = (weight_L+1_Transpose * delta_L+1 dot deltaActivation(activity_L)

			// Using crossentropy loss dOut = label - prediction

			Matrix error = labels.getRows(sampleIndices).subtract(layerActivation[layerActivation.length-1]);
			sumError = error.sum();

			weightBlame[nn.getNumLayers()-1] = error.elementMultiply(layerGradient[layerActivation.length-1]); // Was activity
			biasBlame[nn.getNumLayers()-1] = error.sumColumns();

			for(int j=nn.getNumLayers()-2; j >= 0; j--) {
				Matrix wt = nn.getWeights(j).transpose();
				error = error.multiply(wt);
				weightBlame[j] = error.elementMultiply(layerGradient[j]);
				biasBlame[j] = error.sumColumns();
			}

			// Activation_L * delta_L+1
			for(int j=0; j < deltaWeights.length; j++) {
				deltaWeights[j].elementMultiply_i(momentum);
				deltaWeights[j].add_i(layerActivation[j].transpose().multiply(weightBlame[j+1]).elementMultiply_i(1.0 - momentum));
			}

			// Apply weight changes.
			for(int j=0; j < deltaWeights.length; j++) {
				nn.setWeights(j, nn.getWeights(j).add(deltaWeights[j].elementMultiply(learningRate/(float)batchSize)));
			}

			// Notify user
			if(notification != null && notificationIncrement > 0 && i % notificationIncrement == 0) {
				notification.run();
			}

			//sumError > earlyStopError
		}

	}
}
