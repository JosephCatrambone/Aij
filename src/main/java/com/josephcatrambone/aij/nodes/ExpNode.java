package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class ExpNode extends Node {
	public ExpNode() { super(); }

	public ExpNode(Node input) {
		super(input);
	}

	public Matrix forward(Matrix[] args) {
		return args[0].elementOp(a -> Math.exp(a));
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		return new Matrix[]{ adjoint.elementOp(forward[0], (a,b) -> a*Math.exp(b)) };
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}