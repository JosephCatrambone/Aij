package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class AbsNode extends Node {
	public AbsNode(Node inputNode) {
		super(inputNode);
	}

	public Matrix forward(Matrix[] args) {
		return args[0].elementOp(v -> Math.abs(v));
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		return new Matrix[]{adjoint.elementOp(forward[0], (a, b) -> a*Math.signum(b))};
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}