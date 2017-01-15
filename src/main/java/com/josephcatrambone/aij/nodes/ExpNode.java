package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class ExpNode extends Node {
	public ExpNode(Node input) {
		this.rows = input.rows;
		this.columns = input.columns;
		this.inputs = new int[]{input.id};
	}

	public Matrix forward(Matrix[] args) {
		return args[0].elementOp(a -> (float)Math.exp(a));
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		throw new RuntimeException("Not yet implemented.");
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}