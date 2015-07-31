package com.josephcatrambone.aij.trainers;

import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.networks.MeanFilterNetwork;
import com.josephcatrambone.aij.networks.Network;

/**
 * Created by josephcatrambone on 7/31/15.
 */
public class MeanFilterTrainer implements Trainer {
	public Matrix accumulator;
	public int rowCount;

	/*** train
	 * Given a list of examples (one row = one example),
	 * accumulate the average of the examples and apply it to the mean network.
	 * Can call multiple times to update the average.
	 * Labels are ignored.
	 * @param network
	 * @param examples
	 * @param labels Ignored
	 * @param notification Ignored
	 */
	@Override
	public void train(Network network, Matrix examples, Matrix labels, Runnable notification) {
		if(accumulator == null) {
			accumulator = new Matrix(1, examples.numColumns());
		}

		MeanFilterNetwork net = (MeanFilterNetwork)network;
		accumulator.add_i(examples.sumColumns());
		rowCount += examples.numRows();
		net.mean = accumulator.getRow(0).elementMultiply_i(1.0/(float)rowCount);
	}
}
