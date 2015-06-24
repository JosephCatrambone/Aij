package com.josephcatrambone.aij.trainers;

import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.networks.ConvolutionalNetwork;
import com.josephcatrambone.aij.networks.Network;
import com.josephcatrambone.aij.networks.OneToOneNetwork;

import java.util.Random;
import java.util.function.Consumer;
import java.util.function.UnaryOperator;

/** ConvolutionalTrainer
 * Iterate across a dataset, subsampling according to a window.
 * If gatherSize > examplesPerBatch, will build up gatherSize examples from the set, then train on examplesPerBatch sized subsets.
 * Created by jcatrambone on 5/29/15.
 */
public class ConvolutionalTrainer implements Trainer {
	Random random = new Random();
	public Trainer operatorTrainer = null; // A convolutional network has a trainer. This is the trainer for said network.
	public int notificationIncrement = 0; // If > 0, when iter%notificationIncrement == 0, run notification func.
	public int maxIterations = 10000; // If we do more iterations than this, stop.
	public double earlyStopError = 0.0; // If error is less than this, stop early.
	public int examplesPerBatch = 1; // We will take this many examples from the training set.
	public int subwindowsPerExample = 1; // We will take this many windows per example.
	public double learningRate = 0.1;
	public double momentum = 0.0;
	public double lastError = Double.MAX_VALUE;

	private void log(String msg) {
		System.out.println(msg);
	}

	public void train(Network net, Matrix inputs, Matrix labels, Runnable notification) {
		Random random = new Random();
		ConvolutionalNetwork cn = (ConvolutionalNetwork)net;

		// Remove the old operator (that we are training) from the network and replace with our example spy.
		Network op = cn.getOperator();

		OneToOneNetwork netSpy = new OneToOneNetwork(op.getNumInputs(), op.getNumOutputs());
		Matrix windowData = new Matrix(subwindowsPerExample, op.getNumInputs());
		netSpy.predictionMonitor = new Consumer<Matrix>() {
			int i=0;
			@Override
			public void accept(Matrix intermediate) {
				windowData.setRow(i, intermediate);
				i = (i+1)%windowData.numRows();
			}
		};
		netSpy.predictionFunction = new UnaryOperator<Matrix>() {
			@Override
			public Matrix apply(Matrix matrix) {
				return new Matrix(1, op.getNumOutputs());
			}
		};

		cn.setOperator(netSpy);

		// The netspy will intercept all the training exampels that WOULD be predicted by the network,
		// and will add them to a list.  We then select random subsets and train the network on them.


		for(int i=0; i < maxIterations && lastError > earlyStopError; i++) {
			Matrix examples = new Matrix(subwindowsPerExample*examplesPerBatch, op.getNumInputs());
			Matrix labels2 = null;

			// Don't waste space if we have no labels.
			if(labels != null) {
				labels2 = new Matrix(examplesPerBatch, op.getNumInputs());
			}

			// Randomly select an example
			for(int j=0; j < examplesPerBatch; j++) {
				int exampleIndex = random.nextInt(inputs.numRows());

				// Now get all the windows in this example.
				// windowData will be full of subwindowsPerExample examples.
				cn.predict(inputs.getRow(exampleIndex));

				// After running predict, the windowData should have a bunch of data from the example.
				// Select a subset of the windows.
				for(int k=0; k < subwindowsPerExample; k++) {
					examples.setRow(j*subwindowsPerExample + k, windowData.getRow(random.nextInt(windowData.numRows())));
					if(labels != null) {
						labels2.setRow(j * subwindowsPerExample + k, labels.getRow(exampleIndex));
					}
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
