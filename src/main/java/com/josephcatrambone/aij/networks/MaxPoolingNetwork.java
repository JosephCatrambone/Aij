package com.josephcatrambone.aij.networks;

import com.josephcatrambone.aij.Matrix;

import java.io.IOException;
import java.util.Random;
import java.util.function.UnaryOperator;

/**
 * Created by josephcatrambone on 6/26/15.
 */
public class MaxPoolingNetwork extends FunctionNetwork {
	static final long serialVersionUID = 132431828970254666L;
	private transient Random random;

	public MaxPoolingNetwork(int inputSize) {
		super(inputSize, 1);
		init();
	}

	private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
		in.defaultReadObject();
		init();
	}

	private final void init() {
		random = new Random();
		int inputSize = this.getNumInputs(); // See parent.

		// When predicting, go from num inputs to one output
		this.predictionFunction = (Matrix matrix) -> new Matrix(1, 1, matrix.max());

		// When reconstructing, randomly assign one of the units to the max value.
		this.reconstructionFunction = new UnaryOperator<Matrix>() {
			@Override
			public Matrix apply(Matrix matrix) {
				Matrix output = new Matrix(1, inputSize);
				output.set(0, random.nextInt(output.numRows()), matrix.get(0, 0));
				return output;
			}
		};
	}

}
