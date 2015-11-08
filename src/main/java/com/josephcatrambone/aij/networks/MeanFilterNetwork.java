package com.josephcatrambone.aij.networks;

import com.josephcatrambone.aij.Matrix;

import java.io.Serializable;

/**
 * Created by josephcatrambone on 7/31/15.
 */
public class MeanFilterNetwork extends FunctionNetwork implements Serializable {

	public Matrix mean;

	public MeanFilterNetwork(int inputSize) {
		super(inputSize, inputSize);
		this.predictionFunction = (Matrix m) -> {
			return m.subtract(mean.repmat(m.numRows(), 1));
		};
		this.reconstructionFunction = (Matrix m) -> {
			return m.add(mean.repmat(m.numRows(), 1));
		};
	}
}
