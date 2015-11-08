package com.josephcatrambone.aij.networks;

import com.josephcatrambone.aij.Matrix;

import java.io.IOException;
import java.io.Serializable;

/**
 * Created by josephcatrambone on 7/31/15.
 */
public class MeanFilterNetwork extends FunctionNetwork implements Serializable {
	static final long serialVersionUID = 331112824943661084L;
	public Matrix mean;
	public int inputSize;

	public MeanFilterNetwork(int inputSize) {
		super(inputSize, inputSize);
		init();
	}

	private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
		in.defaultReadObject();
		init();
	}

	private final void init() {
		this.predictionFunction = (Matrix m) -> {
			return m.subtract(mean.repmat(m.numRows(), 1));
		};
		this.reconstructionFunction = (Matrix m) -> {
			return m.add(mean.repmat(m.numRows(), 1));
		};
	}
}
