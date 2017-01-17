package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

/**
 * Created by josephcatrambone on 1/13/17.
 */
public class InputNode extends Node {
	public InputNode() { super(); }

	public InputNode(int rows, int columns) {
		this.columns = columns;
		this.rows = rows;
		this.inputs = new Node[]{};
	}

	public Matrix forward(Matrix[] args) {
		throw new RuntimeException("Forward operation called on input node.  You should not see this.");
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		return new Matrix[]{};
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}

