package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class SigmoidNode extends Node {
	public Matrix forward(Matrix[] args) {
		return null;
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		return null;
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}