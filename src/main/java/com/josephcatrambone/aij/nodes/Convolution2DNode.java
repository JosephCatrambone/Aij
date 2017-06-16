package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class Convolution2DNode extends Node {
	int rowStride = 0;
	int columnStride = 0;

	public Convolution2DNode() { super(); }

	public Convolution2DNode(Node input, Node kernel, int rowStride, int columnStride) {
		this(input, new Node[]{kernel}, rowStride, columnStride);
	}

	public Convolution2DNode(Node input, Node[] kernels, int rowStride, int columnStride) {
		this.rowStride = rowStride;
		this.columnStride = columnStride;

		// F = kernel.width
		//int outputRows = (input.rows - kernel.rows)/rowStride + 1;
		//int outputColumns = (input.columns - kernel.columns)/columnStride + 1;
		int outputRows = input.rows/rowStride + 1;
		int outputColumns = (input.columns/columnStride + 1)*kernels.length;
		this.rows = outputRows;
		this.columns = outputColumns;
		this.inputs = new Node[1 + kernels.length];
		this.inputs[0] = input;
		for(int i=0; i < kernels.length; i++) {
			this.inputs[1+i] = kernels[i];
		}
	}

	public Matrix forward(Matrix[] args) {
		// TODO: Should we add 'tocolumn' and 'torow'?
		// Something to do both so we can get packed filters?
		Matrix output = new Matrix(this.rows, this.columns);
		Matrix input = args[0];
		// If we have multiple kernels, copy all of them.
		Matrix[] kernels = new Matrix[args.length-1];
		for(int i=1; i < args.length; i++) {
			kernels[i-1] = args[i];
		}

		// For each filter, sum the element-wise product with the input volume and assign it to the output.
		for(int r=0; r < this.rows; r++) {
			for(int c=0; c < this.columns; c++) {
				// Should add padding.
				int inRCenter = r*rowStride;
				int inCCenter = c*columnStride;
				for(int k=0; k < kernels.length; k++) {
					Matrix kernel = kernels[k];
					double accumulator = 0;
					// Center kernel at r,c
					for (int rk = 0; rk < kernel.rows; rk++) {
						for (int ck = 0; ck < kernel.columns; ck++) {
							// Delta position
							int inR = rk - (kernel.rows / 2) + inRCenter;
							int inC = ck - (kernel.columns / 2) + inCCenter;
							if (inR >= 0 && inR < input.rows && inC >= 0 && inC < input.columns) {
								accumulator += input.get(inR, inC) * kernel.get(rk, ck);
							}
						}
					}
					output.set(r, c+k, accumulator);
				}
			}
		}
		return output;
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		// If this were c = i*k, then i_adj = c_adj*k and k_adj = c_adj*i
		// Instead, treat this as much bigger and apply the region in question for each adjoint.
		Matrix inputAdjoint = new Matrix(forward[0].rows, forward[0].columns);
		Matrix input = forward[0];
		Matrix[] kernelAdjoints = new Matrix[forward.length-1];
		Matrix[] kernels = new Matrix[forward.length-1];
		for(int i=0; i < kernelAdjoints.length; i++) {
			Matrix kernelAdjoint = new Matrix(forward[1+i].rows, forward[1+i].columns);
			kernelAdjoints[i] = kernelAdjoint;
			kernels[i] = forward[i+1];
		}
		// For each filter, sum the element-wise product with the input volume and assign it to the output.
		for(int r=0; r < this.rows; r++) {
			for(int c=0; c < this.columns; c++) {
				for(int k=0; k < kernels.length; k++) {
					Matrix kernel = kernels[k];
					// Center kernel at r,c
					for (int rk = 0; rk < kernel.rows; rk++) {
						for (int ck = 0; ck < kernel.columns; ck++) {
							int outRow = rk - (kernel.rows / 2) + (r * rowStride);
							int outColumn = ck - (kernel.columns / 2) + (c * columnStride);
							if (outRow >= 0 && outRow < inputAdjoint.rows && outColumn >= 0 && outColumn < inputAdjoint.columns) {
								inputAdjoint.set(outRow, outColumn, inputAdjoint.get(outRow, outColumn) + adjoint.get(r, c+k) * kernel.get(rk, ck));
								kernelAdjoints[k].set(rk, ck, kernelAdjoints[k].get(rk, ck) + adjoint.get(r, c+k) * input.get(outRow, outColumn));
							}
						}
					}
				}
			}
		}
		Matrix[] results = new Matrix[1+kernelAdjoints.length];
		results[0] = inputAdjoint;
		for(int k=0; k < kernelAdjoints.length; k++) {
			results[1+k] = kernelAdjoints[k];
		}
		return results;
	}

	// Used to augment serialization.
	public String extraDataToString() { return rowStride + "," + columnStride; };
	public void extraDataFromString(String s) {
		String[] tokens = s.split(",");
		this.rowStride = Integer.parseInt(tokens[0]);
		this.columnStride = Integer.parseInt(tokens[1]);
	}
}
