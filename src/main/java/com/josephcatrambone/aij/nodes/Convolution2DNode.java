package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class Convolution2DNode extends Node {
	int rowStride = 0;
	int columnStride = 0;

	public Convolution2DNode() { super(); }

	public Convolution2DNode(Node input, Node kernel, int rowStride, int columnStride) {
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
		this.rowStride = rowStride;
		this.columnStride = columnStride;

		// F = kernel.width
		int outputRows = (input.rows - kernel.rows)/rowStride + 1;
		int outputColumns = (input.columns - kernel.columns)/columnStride + 1;
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

				int inRCenter = r*rowStride;
				int inCCenter = c*columnStride;
				float accumulator = 0;
				// Center kernel at r,c
				for(int rk = 0; rk < kernel.rows; rk++) {
					for(int ck=0; ck < kernel.columns; ck++) {
						// Delta position
						int inR = rk-kernel.rows/2+inRCenter;
						int inC = ck-kernel.columns/2 + inCCenter;
						if(inR >= 0 && inR < this.rows && inC >= 0 && inC < this.columns) {
							accumulator += input.get(inRCenter, inCCenter)*kernel.get(rk, ck);
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
				// Center kernel at r,c
				for(int rk = 0; rk < kernel.rows; rk++) {
					for(int ck=0; ck < kernel.columns; ck++) {
						int outRow = rk-kernel.rows/2 + (r*rowStride);
						int outColumn = ck-kernel.columns/2 + (c*columnStride);
						if(outRow >= 0 && outRow < inputAdjoint.rows && outColumn >= 0 && outColumn < inputAdjoint.columns) {
							inputAdjoint.set(outRow, outColumn, inputAdjoint.get(outRow, outColumn) + adjoint.get(r, c)*kernel.get(rk, ck));
							kernelAdjoint.set(rk, ck, kernelAdjoint.get(rk, ck) + adjoint.get(r, c)*input.get(outRow, outColumn));
						}
					}
				}
			}
		}
		return new Matrix[]{inputAdjoint, kernelAdjoint};
	}

	// Used to augment serialization.
	public String extraDataToString() { return rowStride + "," + columnStride; };
	public void extraDataFromString(String s) {
		String[] tokens = s.split(",");
		this.rowStride = Integer.parseInt(tokens[0]);
		this.columnStride = Integer.parseInt(tokens[1]);
	}
}
