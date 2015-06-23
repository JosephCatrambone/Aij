package com.josephcatrambone.aij.trainers;

import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.networks.ConvolutionalNetwork;
import com.josephcatrambone.aij.networks.Network;
import com.josephcatrambone.aij.networks.OneToOneNetwork;

import java.util.Arrays;
import java.util.Random;

/** ConvolutionalTrainer
 * Iterate across a dataset, subsampling according to a window.
 * If gatherSize > batchSize, will build up gatherSize examples from the set, then train on batchSize sized subsets.
 * Created by jcatrambone on 5/29/15.
 */
public class ConvolutionalTrainer implements Trainer {
	Random random = new Random();
	public Trainer operatorTrainer = null; // A convolutional network has a trainer. This is the trainer for said network.
	public int notificationIncrement = 0; // If > 0, when iter%notificationIncrement == 0, run notification func.
	public int maxIterations = 10000; // If we do more iterations than this, stop.
	public double earlyStopError = 0.0; // If error is less than this, stop early.
	public int batchSize = 1; // We will take (in total) this many examples from the training set.
	public int minibatchSize = 1; // We will take this many windows per example.
	public double learningRate = 0.1;
	public double momentum = 0.0;
	public double lastError = Double.MAX_VALUE;

	private void log(String msg) {
		System.out.println(msg);
	}

	public void train(Network net, Matrix inputs, Matrix labels, Runnable notification) {
		Random random = new Random();
		int numWindows = batchSize/minibatchSize;
		ConvolutionalNetwork cn = (ConvolutionalNetwork)net;

		// Remove the old operator (that we are training) from the network and replace with our example spy.
		Network op = cn.getOperator();

		OneToOneNetwork netSpy = new OneToOneNetwork(cn.getOperator().getNumInputs());
		Matrix windowData = new Matrix(minibatchSize, cn.getOperations());
		netSpy.predictionMonitor = new OneToOneNetwork.Monitor() {
			int i=0;
			@Override
			public void run(Matrix intermediate) {
				windowData.setRow(i, intermediate);
				i = i+1%windowData.numRows();
			}
		};

		cn.setOperator(netSpy);

		// The netspy will intercept all the training exampels that WOULD be predicted by the network,
		// and will add them to a list.  We then select random subsets and train the network on them.


		for(int i=0; i < maxIterations && lastError > earlyStopError; i++) {
			Matrix examples = new Matrix(batchSize, op.getNumInputs());
			Matrix labels2 = new Matrix(batchSize, op.getNumInputs());

			// Randomly select an example
			for(int j=0; j < numWindows; j++) {
				int exampleIndex = random.nextInt(inputs.numRows());

				// Now get all the windows in this example.
				cn.predict(inputs.getRow(exampleIndex));

				// After running predict, the windowData should have a bunch of data from the example.
				// Select a subset of the windows.
				for(int k=0; k < minibatchSize; k++) {
					examples.setRow(j*minibatchSize + k, windowData.getRow(random.nextInt(windowData.numRows())));
					labels2.setRow(j*minibatchSize + k, labels.getRow(exampleIndex));
				}
			}

			// Train on the data.
			operatorTrainer.train(op, examples, labels2, notification);

			// At the appropriate step, notify user.
			/*
			if(notification != null && notificationIncrement > 0 && (i+1)%notificationIncrement == 0) {
				notification.run();
			}
			*/
		}

		cn.setOperator(op);

	}
}
