package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class MatrixMultiplyNode extends Node {

	public MatrixMultiplyNode(Node left, Node right) {
		this.rows = left.rows;
		this.columns = right.columns;
		assert(left.columns == right.rows);
		this.inputs = new Node[]{left, right};
	}

	public Matrix forward(Matrix[] args) {
		return args[0].matmul(args[1]);
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {

		return new Matrix[]{
				// Left adjoint. If C=AB, adj(a) = adj(c)*bT
				adjoint.matmul(forward[1].transpose()),
				// Right adjoint.  adj(b) = aT*adj(c)
				forward[0].transpose().matmul(adjoint)
		};

	}

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}