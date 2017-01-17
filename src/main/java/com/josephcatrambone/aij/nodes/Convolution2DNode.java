package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

class Convolution2DNode extends Node {
	int padding = 0;
	int stride = 0;

	public Convolution2DNode() { super(); }

	public Convolution2DNode(Node input, Node kernel, int stride, int padding) {
		// Performs a shallow 2D convolution on the input node.
		// W1 H1 D1
		// K = num filters.
		// F = spatial extent.
		// S = stride.
		// P = padding.
		// W2 = (W1 - F + 2P)/S + 1
		// H2 = (H1 - F + 2P)/S + 1
		// D = k
		// For the 2D convolution, the number of filters is restricted to 1.

		// F = kernel.width
		int outputRows = (input.rows - kernel.rows + 2*padding)/stride + 1;
		int outputColumns = (input.columns - kernel.columns + 2*padding)/stride + 1;
		this.rows = outputRows;
		this.columns = outputColumns;
		this.inputs = new Node[]{input, kernel};
	}

	public Matrix forward(Matrix[] args) {
		Matrix output = new Matrix(this.rows, this.columns);
		// For each filter, sum the element-wise product with the input volume and assign it to the output.
		return null;
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		return null;
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}