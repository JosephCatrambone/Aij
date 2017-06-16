package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

/**
 * Expand the input node to the given size or cut extra elements.
 * Created by jcatrambone on 6/13/17.
 */
public class PadCropNode extends Node {

	public PadCropNode(int rows, int columns, Node input) {
		super(rows, columns, input);
	}

	@Override
	public Matrix forward(Matrix[] args) {
		Matrix result = new Matrix(this.rows, this.columns, (i,j) -> {
			if(i < args[0].rows && j < args[0].columns) {
				return args[0].get(i, j);
			} else {
				return 0.0;
			}
		});
		return result;
	}

	@Override
	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		Matrix res = new Matrix(forward[0].rows, forward[0].columns, (i,j) -> {
			if(i < forward[0].rows && i < adjoint.rows && j < forward[0].columns && j < adjoint.columns) {
				return adjoint.get(i,j);
			} else {
				return 0.0;
			}
		});
		return new Matrix[]{res};
	}
}
