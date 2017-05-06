package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class NegateNode extends Node {

	public NegateNode() { super(); }

	public NegateNode(Node left) {
		super(left);
	}

	public Matrix forward(Matrix[] args) {
		return args[0].elementOp(a -> -a);
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		return new Matrix[]{
			adjoint.elementOp(a -> -a)
		};
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}