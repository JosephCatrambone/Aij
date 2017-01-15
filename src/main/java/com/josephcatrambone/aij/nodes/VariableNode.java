package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class VariableNode extends Node {
	Matrix variable;

	public VariableNode(int rows, int columns) {
		variable = new Matrix(rows, columns);
		this.rows = rows;
		this.columns = columns;
		this.inputs = new int[]{};
	}

	public Matrix forward(Matrix[] args) {
		return variable;
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		return new Matrix[]{};
	}

	// Used to augment serialization.
	public String extraDataToString() { return variable.toString(); };
	public void extraDataFromString(String s) {
		variable = Matrix.fromString(s);
	}
}