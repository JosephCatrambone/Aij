package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class SubtractNode extends Node {
	public SubtractNode(Node left, Node right) {
		super(left.rows, left.columns, left, right);
	}

	@Override
	public Matrix forward(Matrix[] args) {
		return args[0].elementOp(args[1], (a,b) -> a-b);
	}

	@Override
	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		Matrix[] result = new Matrix[forward.length];
		result[0] = adjoint;
		result[1] = adjoint.elementOp(a -> -a);
		return result;
	}
}