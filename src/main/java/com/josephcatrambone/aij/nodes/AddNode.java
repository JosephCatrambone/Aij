package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

/**
 * Created by josephcatrambone on 1/13/17.
 */
public class AddNode extends Node {

	public AddNode() { super(); } // Need empty constructor.

	public AddNode(Node left, Node right) {
		super(left.rows, left.columns, left, right);
	}

	@Override
	public Matrix forward(Matrix[] args) {
		Matrix result = new Matrix(args[0].rows, args[0].columns);
		for(Matrix m : args) {
			result.elementOp_i(m, (a,b) -> a+b);
		}
		return result;
	}

	@Override
	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		Matrix[] result = new Matrix[forward.length];
		for(int i=0; i < forward.length; i++) {
			result[i] = adjoint;
		}
		return result;
	}
}
