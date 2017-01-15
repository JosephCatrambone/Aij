package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class MultiplyNode extends Node {

	public MultiplyNode(Node left, Node right) {
		this.rows = left.rows;
		this.columns = left.columns;
		this.inputs = new Node[]{left, right};
	}

	public Matrix forward(Matrix[] args) {
		return args[0].elementOp(args[1], (a,b) -> a*b);
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		return new Matrix[]{
			adjoint.elementOp(forward[1], (a, b) -> a*b),
			adjoint.elementOp(forward[0], (a, b) -> a*b)
		};
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}