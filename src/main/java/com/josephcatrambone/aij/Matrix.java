package com.josephcatrambone.aij;

import org.jblas.DoubleMatrix;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.UnaryOperator;

/**
 * Created by jcatrambone on 5/28/15.
 * This is a generic matrix wrapper tool with most of the methods needed to NN/ML activities.
 * There are some static methods provided to emulate numpy funcionality as required.
 * All in-place operations are suffixed with 'i'.
 * This is a VERY thin wrapper around JBlas.
 */
public class Matrix extends DoubleMatrix implements Serializable {

	public Matrix(int rows, int columns) {
		super(rows, columns);
	}

	// Some more specialized operations
	// We're doing these operations inside the matrix class because there's a chance we can optimize transparently.
	public Matrix tanh() {
		return this.elementOp(x -> Math.tanh(x));
	}

	public Matrix tanh_i() {
		//List<Double> dataCollection = m.elementsAsList();
		//dataCollection.parallelStream().forEachOrdered(x -> Math.tanh(x));
		//return new Matrix(new DoubleMatrix(dataCollection));
		this.elementOp_i(x -> Math.tanh(x));
		return this;
	}

	public Matrix dtanhFromActivation() {
		return this.elementOp(x -> (1 - x*x));
	}

	public Matrix sigmoid() {
		return this.elementOp(x -> 1.0 / (1.0 + Math.exp(-x)));
	}

	public Matrix sigmoid_i() {
		return this.elementOp_i(x -> 1.0 / (1.0 + Math.exp(-x)));
	}


	public Matrix dsigmoidFromActivation() {
		return this.elementOp(sigx -> sigx * (1 - sigx));
	}

	public Matrix softplus() {
		return this.elementOp(x -> Math.log(1 + Math.exp(x)));
	}

	public Matrix softplus_i() {
		return this.elementOp_i(x -> Math.log(1 + Math.exp(x)));
	}

	public Matrix dsoftplusFromActivation() {
		// Derivative of log(1 + e^x) = sigmoid.
		return null;
	}

	/*** elementOp_i
	 * Run an operator in place on each of the elements of this matrix.
	 * MUCH slower than the above ops, but general purpose.
	 * @param op
	 */
	public Matrix elementOp_i(UnaryOperator<Double> op) {
		for(int y=0; y < this.rows; y++) {
			for(int x=0; x < this.columns; x++) {
				this.put(y, x, op.apply(this.get(y, x)));
			}
		}
		return this;
	}

	/*** elementOp
	 * Run an operator on each of the elements of this matrix.
	 * MUCH slower than the above ops, but general purpose.
	 * @param op
	 */
	public Matrix elementOp(UnaryOperator<Double> op) {
		Matrix output = new Matrix(this.rows, this.columns);
		for(int y=0; y < this.rows; y++) {
			for(int x=0; x < this.columns; x++) {
				output.put(y, x, op.apply(this.get(y, x)));
			}
		}
		return output;
	}
}
