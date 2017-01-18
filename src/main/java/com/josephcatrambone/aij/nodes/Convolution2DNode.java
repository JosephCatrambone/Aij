package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class Convolution2DNode extends Node {
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
		this.stride = stride;
		this.padding = padding;

		// F = kernel.width
		int outputRows = (input.rows - kernel.rows + 2*padding)/stride + 1;
		int outputColumns = (input.columns - kernel.columns + 2*padding)/stride + 1;
		this.rows = outputRows;
		this.columns = outputColumns;
		this.inputs = new Node[]{input, kernel};
	}

	public Matrix forward(Matrix[] args) {
		Matrix output = new Matrix(this.rows, this.columns);
		Matrix input = args[0];
		Matrix kernel = args[1];
		// For each filter, sum the element-wise product with the input volume and assign it to the output.
		for(int r=0; r < this.rows; r++) {
			for(int c=0; c < this.columns; c++) {

				int inR = padding+(r*stride);
				int inC = padding+(c*stride);
				float accumulator = 0;
				// Center kernel at r,c
				for(int rk = 0; rk < kernel.rows; rk++) {
					for(int ck=0; ck < kernel.columns; ck++) {
						// Delta position
						int drk = rk-kernel.rows/2;
						int dck = ck-kernel.columns/2;
						if(inR+drk >= 0 && inR+drk < this.rows && inC+dck >= 0 && inC+dck <= this.columns) {
							accumulator += input.get(inR+drk, inC+dck)*kernel.get(rk, ck);
						}
					}
				}
				output.set(r, c, accumulator);
			}
		}
		return output;
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		// If this were c = i*k, then i_adj = c_adj*k and k_adj = c_adj*i
		// Instead, treat this as much bigger and apply the region in question for each adjoint.
		Matrix inputAdjoint = new Matrix(forward[0].rows, forward[0].columns);
		Matrix kernelAdjoint = new Matrix(forward[1].rows, forward[1].columns);
		Matrix input = forward[0];
		Matrix kernel = forward[1];
		// For each filter, sum the element-wise product with the input volume and assign it to the output.
		for(int r=0; r < this.rows; r++) {
			for(int c=0; c < this.columns; c++) {
				int inR = padding+(r*stride);
				int inC = padding+(c*stride);
				// Center kernel at r,c
				for(int rk = 0; rk < kernel.rows; rk++) {
					for(int ck=0; ck < kernel.columns; ck++) {
						int r2 = rk-kernel.rows/2;
						int c2 = ck-kernel.columns/2;
						if(inR+r2 >= 0 && inR+r2 < this.rows && inR+c2 >= 0 && inR+c2 <= this.columns) {
							inputAdjoint.set(inR+r2, inC+c2, inputAdjoint.get(inR+r2, inC+c2) + adjoint.get(r,c)*kernel.get(rk, ck));
							kernelAdjoint.set(rk, ck, kernelAdjoint.get(rk, ck)+adjoint.get(r,c)*input.get(inR+r2, inC+c2));
						}
					}
				}
			}
		}
		return new Matrix[]{inputAdjoint, kernelAdjoint};
	}

	// Used to augment serialization.
	public String extraDataToString() { return padding + "," + stride; };
	public void extraDataFromString(String s) {
		String[] tokens = s.split(",");
		this.padding = Integer.parseInt(tokens[0]);
		this.stride = Integer.parseInt(tokens[1]);
	}
}