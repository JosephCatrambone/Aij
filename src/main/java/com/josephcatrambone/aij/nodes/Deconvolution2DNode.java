package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class Deconvolution2DNode extends Node {
	int padding = 0;
	int rowStride = 0;
	int columnStride = 0;

	public Deconvolution2DNode() { super(); }

	public Deconvolution2DNode(Node input, Node kernel, int rowStride, int columnStride) {
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
		// We have W2 and need to determine W1 from the parameters.
		// W2 = ((W1 - F + 2P)/S) + 1
		// W2 - 1 = (W1 - F + 2P)/S
		// S*(W2-1) = W1 - F + 2P
		// S*(W2-1) - 2P + F = W1
		int outputRows = (input.rows-1)*rowStride + kernel.rows;
		int outputColumns = (input.columns-1)*columnStride + kernel.columns;
		this.rows = outputRows;
		this.columns = outputColumns;
		this.inputs = new Node[]{input, kernel};
	}

	public Matrix forward(Matrix[] args) {
		Matrix output = new Matrix(this.rows, this.columns);
		Matrix input = args[0];
		Matrix kernel = args[1];
		for(int inRow=0; inRow < input.rows; inRow++) {
			for(int inCol=0; inCol < input.columns; inCol++) {
				// inRow/Col gives us our position on the convolution object.
				// Calculate the center on our output image from our position on the convoluted input.
				int outcenterRow = (inRow*rowStride);
				int outcenterCol = (inCol*columnStride);

				// Iterate over the kernel and use that to apply our output.
				for(int kRow=0; kRow < kernel.rows; kRow++) {
					for(int kCol=0; kCol < kernel.columns; kCol++) {
						int outRow = outcenterRow - kernel.rows/2 + kRow;
						int outCol = outcenterCol - kernel.columns/2 + kCol;

						if(outRow >= 0 && outRow < output.rows && outCol >= 0 && outCol < output.columns) {
							output.set(outRow, outCol, output.get(outRow, outCol) + kernel.get(kRow, kCol)*input.get(inRow, inCol));
						}
					}
				}
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

		for(int inRow=0; inRow < input.rows; inRow++) {
			for(int inCol=0; inCol < input.columns; inCol++) {
				// inRow/Col gives us our position on the convolution object.
				// Calculate the center on our output image from our position on the convoluted input.
				int outcenterRow = (inRow*rowStride);
				int outcenterCol = (inCol*columnStride);

				// Iterate over the kernel and use that to apply our output.
				for(int kRow=0; kRow < kernel.rows; kRow++) {
					for(int kCol=0; kCol < kernel.columns; kCol++) {
						int outRow = outcenterRow - kernel.rows/2 + kRow;
						int outCol = outcenterCol - kernel.columns/2 + kCol;

						if(outRow >= 0 && outRow < adjoint.rows && outCol >= 0 && outCol < adjoint.columns) {
							// output.set(outRow, outCol, output.get(outRow, outCol) + kernel.get(kRow, kCol)*input.get(inRow, inCol));
							// Given ourput = input * kernel, and our adjoint applies to the output, 
							// adj(input) += adj(output)*kernel
							// adj(kernel) += adj(output)*input
							inputAdjoint.set(inRow, inCol, inputAdjoint.get(inRow, inCol)+(adjoint.get(outRow, outCol)*kernel.get(kRow, kCol)));
							kernelAdjoint.set(kRow, kCol, kernelAdjoint.get(kRow, kCol)+(adjoint.get(outRow, outCol)*input.get(inRow, inCol)));
						}
					}
				}
			}
		}

		// For each filter, sum the element-wise product with the input volume and assign it to the output.
		return new Matrix[]{inputAdjoint, kernelAdjoint};
	}

	// Used to augment serialization.
	public String extraDataToString() { return padding + "," + rowStride + "," + columnStride; };
	public void extraDataFromString(String s) {
		String[] tokens = s.split(",");
		this.padding = Integer.parseInt(tokens[0]);
		this.rowStride = Integer.parseInt(tokens[1]);
		this.rowStride = Integer.parseInt(tokens[2]);
	}
}
