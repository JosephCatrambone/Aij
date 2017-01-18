package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

/**
 * Created by josephcatrambone on 1/16/17.
 */
public class ReshapeNode extends Node {

	public ReshapeNode() { super(); }

	public ReshapeNode(Node n, int rows, int columns) {
		// If rows == -1 or columns == -1, calculate it from the other value.
		if(rows == -1 ^ columns == -1) {
			if(rows == -1) {
				rows = (n.rows*n.columns)/columns;
			} else {
				columns = (n.rows*n.columns)/rows;
			}
		}
		assert(rows*columns == n.rows*n.columns);
		this.rows = rows;
		this.columns = columns;
		this.inputs = new Node[]{n};
	}

	@Override
	public Matrix forward(Matrix[] args) {
		return new Matrix(this.rows, this.columns, (r,c) -> args[0].data[c+r*args[0].columns]);
	}

	@Override
	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		return new Matrix[]{new Matrix(forward[0].rows, forward[0].columns, (r,c) -> adjoint.data[c+r*adjoint.columns])};
	}
}
