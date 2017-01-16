package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class PowerNode extends Node {
	float exponent;

	public PowerNode(Node input, float e) {
		super(input);
		this.exponent = e;
	}

	public Matrix forward(Matrix[] args) {
		return args[0].elementOp(a -> (float)Math.pow(a, exponent));
	}

	// z := elem(x, op)
	// x_adj += z_adj (dot) elem(x, delta op)
	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		return new Matrix[]{
			adjoint.elementOp(
				forward[0].elementOp(
					a -> exponent*(float)Math.pow(a, exponent-1.0f)
				),
				(adj,x)->adj*x
			)
		};
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""+exponent; };
	public void extraDataFromString(String s) {
		exponent = Float.parseFloat(s);
	}
}