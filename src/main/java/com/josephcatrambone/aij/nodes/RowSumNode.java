package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class RowSumNode extends Node {

	public RowSumNode() { super(); }

	public RowSumNode(Node input) {
		this.rows = input.rows;
		this.columns = 1;
		this.inputs = new Node[]{input};
	}

	public Matrix forward(Matrix[] args) {
		Matrix m = new Matrix(this.rows, this.columns);
		for(int r = 0; r < this.rows; r++) {
			float accumulator = 0;
			for(int c = 0; c < args[0].columns; c++) {
				accumulator += args[0].get(r, c);
			}
			m.set(r, 0, accumulator);
		}
		return m;
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		// Have to sum up the adjoints from all the rows, taking slices.
		Matrix newAdjoint = new Matrix(forward[0].rows, forward[0].columns);
		for(int r = 0; r < this.rows; r++) {
			for(int c = 0; c < forward[0].columns; c++) {
				newAdjoint.set(r, c, newAdjoint.get(r, c) + adjoint.get(r, 0));
			}
		}
		return new Matrix[]{newAdjoint};
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}