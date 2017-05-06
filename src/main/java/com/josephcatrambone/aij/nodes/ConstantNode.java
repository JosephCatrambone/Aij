package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

/**
 * Created by jcatrambone on 2/22/17.
 */
public class ConstantNode extends Node {

	private double value;

	public ConstantNode() { super(); }

	public ConstantNode(int rows, int columns, double value) {
		super(rows, columns);
		this.value = value;
	}

	public ConstantNode(double value, Node match) {
		super(match.rows, match.columns);
		this.value = value;
	}

	public Matrix forward(Matrix[] args) {
		Matrix m = new Matrix(this.rows, this.columns, (i, j) -> value );
		return m;
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		// Have to sum up the adjoints from all the rows, taking slices.
		return new Matrix[]{};
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""+value; };
	public void extraDataFromString(String s) {
		value = Double.parseDouble(s);
	}

}
