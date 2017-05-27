package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class CollapseSumNode extends Node {

	public CollapseSumNode() { super(); }

	public CollapseSumNode(Node input) {
		this.rows = 1;
		this.columns = 1;
		this.inputs = new Node[]{input};
	}

	public Matrix forward(Matrix[] args) {
		double accumulator = 0;
		for(int i=0; i < args[0].data.length; i++) {
			accumulator += args[0].data[i];
		}
		return new Matrix(1, 1, new double[]{accumulator});
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		// Have to sum up the adjoints from all the rows, taking slices.
		Matrix newAdjoint = new Matrix(forward[0].rows, forward[0].columns, (i,j) -> adjoint.data[0]);
		return new Matrix[]{newAdjoint};
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}