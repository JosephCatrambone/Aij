package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class InverseNode extends Node {
	public InverseNode() { super(); }

	public InverseNode(Node inputNode) {
		super(inputNode);
	}

	public Matrix forward(Matrix[] args) {
		return args[0].elementOp(v -> 1.0f/v);
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		return new Matrix[]{adjoint.elementOp(forward[0], (a, b) -> a*-1.0f/(b*b))};
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}