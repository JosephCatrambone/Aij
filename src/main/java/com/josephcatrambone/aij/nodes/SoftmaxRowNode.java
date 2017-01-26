package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class SoftmaxRowNode extends Node {

	public SoftmaxRowNode() { super(); }

	public SoftmaxRowNode(Node input) {
		super(input);
	}

	public Matrix forward(Matrix[] args) {
		Matrix m = new Matrix(this.rows, this.columns);
		for(int r = 0; r < this.rows; r++) {
			float accumulator = 0;
			for(int c = 0; c < args[0].columns; c++) {
				accumulator += Math.exp(args[0].get(r, c));
			}
			for(int c = 0; c < args[0].columns; c++) {
				m.set(r, c, (float)Math.exp(args[0].get(r, c))/accumulator);
			}
		}
		return m;
	}

	public Matrix[] reverse(Matrix[] fwd, Matrix adjoint) {
		// Have to sum up the adjoints from all the rows, taking slices.
		Matrix newAdjoint = new Matrix(fwd[0].rows, fwd[0].columns);
		Matrix x = this.forward(fwd);
		for(int r = 0; r < this.rows; r++) {
			for(int c = 0; c < adjoint.columns; c++) {
				//newAdjoint.set(r, c, (float)Math.exp(forward[0].get(r, c)));
			}
		}
		return new Matrix[]{newAdjoint};
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}