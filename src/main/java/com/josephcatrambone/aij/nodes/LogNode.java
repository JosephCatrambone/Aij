package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class LogNode extends Node {

	public LogNode() { super(); }

	public LogNode(Node input) {
		super(input);
	}

	// z := elem(x, op)
	// x_adj += z_adj (dot) elem(x, delta op)
	public Matrix forward(Matrix[] args) {
		return args[0].elementOp((a) -> (double)Math.log(a));
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		// elem x and delta op.
		return new Matrix[]{adjoint.elementOp(forward[0], (a, b) -> a/b)};
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}