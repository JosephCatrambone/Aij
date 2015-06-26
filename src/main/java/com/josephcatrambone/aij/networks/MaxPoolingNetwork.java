package com.josephcatrambone.aij.networks;

import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.layers.Layer;

import java.io.Serializable;
import java.util.Random;
import java.util.function.UnaryOperator;

/**
 * Created by josephcatrambone on 6/26/15.
 */
public class MaxPoolingNetwork extends FunctionNetwork implements Serializable, Network {
	public MaxPoolingNetwork(int inputSize) {
		super(inputSize, 1);

		// When predicting, go from num inputs to one output
		this.predictionFunction = new UnaryOperator<Matrix>() {
			@Override
			public Matrix apply(Matrix matrix) {
				return new Matrix(1, 1, matrix.max());
			}
		};

		// When reconstructing, randomly assign one of the units to the max value.
		this.reconstructionFunction = new UnaryOperator<Matrix>() {
			Random random = new Random();
			@Override
			public Matrix apply(Matrix matrix) {
				Matrix output = new Matrix(1, inputSize);
				output.set(0, random.nextInt(output.numRows()), matrix.get(0, 0));
				return output;
			}
		};
	}
}
