package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

import java.util.Random;

/**
 * Created by josephcatrambone on 2017/01/13.
 */
public class DropoutNode extends Node {
	Random random = new Random();
	Matrix lastNoiseVectorOutput = null;
	public double dropoutRate = 0.5;

	public DropoutNode() { super(); } // Need empty constructor.

	public DropoutNode(Node left, double dropoutRate) {
		super(left.rows, left.columns, left);
		this.dropoutRate = dropoutRate;
	}

	@Override
	public Matrix forward(Matrix[] args) {
		// TODO: This will cause issues when multithreading.  Find a better way to parallelize.
		if(dropoutRate == 0) {
			return args[0];
		} else {
			lastNoiseVectorOutput = new Matrix(args[0].rows, args[0].columns, (i, j) -> {
				if (random.nextDouble() > dropoutRate) {
					return 1.0;
				} else {
					return 0.0;
				}
			});
			Matrix result = args[0].elementOp(lastNoiseVectorOutput, (a, b) -> a * b);
			return result;
		}
	}

	@Override
	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		if(dropoutRate > 0 && lastNoiseVectorOutput != null) {
			return new Matrix[]{ new Matrix(forward[0].rows, forward[0].columns, (i,j) -> adjoint.get(i,j)*lastNoiseVectorOutput.get(i,j)) };
		} else {
			return new Matrix[]{ adjoint };
		}
	}

	public String extraDataToString() { return ""+dropoutRate; };
	public void extraDataFromString(String s) {
		dropoutRate = Double.parseDouble(s);
	}
}
