package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class ReLUNode extends Node {

	public ReLUNode() { super(); }

	public ReLUNode(Node input) {
		super(input);
	}

	public Matrix forward(Matrix[] args) {
		return args[0].elementOp(a -> (double)Math.max(0, a));
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		return new Matrix[] {
			adjoint.elementOp(forward[0], (adj, x) -> { if(x < 0) { return 0.001f*adj; } else { return adj; } } )
		};
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}
