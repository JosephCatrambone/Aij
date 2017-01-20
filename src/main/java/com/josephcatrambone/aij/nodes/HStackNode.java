package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class HStackNode extends Node {

	public HStackNode() { super(); }

	public HStackNode(Node left, Node right) {
		assert(left.rows == right.rows);
		this.rows = left.rows;
		this.columns = left.columns + right.columns;
		this.inputs = new Node[]{left, right};
	}

	public Matrix forward(Matrix[] args) {
		Matrix m = new Matrix(this.rows, this.columns);
		for(int r = 0; r < this.rows; r++) {
			for(int c = 0; c < args[0].columns; c++) {
				m.set(r, c, args[0].get(r, c));
			}
			for(int c = 0; c < args[1].columns; c++) {
				m.set(r, c+args[0].columns, args[1].get(r, c));
			}
		}
		return m;
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		// Have to sum up the adjoints from all the rows, taking slices.
		Matrix leftAdjoint = new Matrix(forward[0].rows, forward[0].columns);
		Matrix rightAdjoint = new Matrix(forward[1].rows, forward[1].columns);
		for(int r=0; r < adjoint.rows; r++) {
			for(int c=0; c < forward[0].columns; c++) {
				leftAdjoint.set(r, c, adjoint.get(r, c));
			}
			for(int c=0; c < forward[1].columns; c++) {
				rightAdjoint.set(r, c, adjoint.get(r, forward[0].columns+c));
			}
		}
		return new Matrix[]{leftAdjoint, rightAdjoint};
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""; }
	public void extraDataFromString(String s) {}
}
