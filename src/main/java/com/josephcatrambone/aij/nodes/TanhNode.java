package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class TanhNode extends Node {

	public TanhNode() { super(); }

	public TanhNode(Node input) {
		super(input);
	}

	public Matrix forward(Matrix[] args) {
		return args[0].elementOp(a -> (double)Math.tanh(a));
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		return new Matrix[] {
			adjoint.elementOp(forward[0], (adj, x) -> adj*(1.0f - (double)(Math.tanh(x)*Math.tanh(x))) )
		};
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}