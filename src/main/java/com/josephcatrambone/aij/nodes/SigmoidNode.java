package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class SigmoidNode extends Node {

	public SigmoidNode() { super(); }

	public SigmoidNode(Node input) {
		super(input);
	}

	public Matrix forward(Matrix[] args) {
		return args[0].elementOp(x -> 1.0f/(1.0f+(double)Math.exp(-x)));
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		return new Matrix[]{
			adjoint.elementOp(forward[0], (adj, x) ->
				// We use the long-form of the sigmoid derivative instead of sig(x)*(1.0f-sig(x)).
				// Fewer multiplies.
				(double)(adj*(Math.exp(-x)/(1.0f+Math.pow(1.0f + Math.exp(-x), 2.0f))))
			)
		};
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}