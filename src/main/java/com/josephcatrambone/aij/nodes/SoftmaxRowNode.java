package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class SoftmaxRowNode extends Node {

	public SoftmaxRowNode() { super(); }

	public SoftmaxRowNode(Node input) {
		super(input);
	}

	public Matrix forward(Matrix[] args) {
		Matrix m = new Matrix(this.rows, this.columns);
		for(int r = 0; r < this.rows; r++) {
			// Do this max thing to improve the numerical stability.
			float max = 0;
			for(int c=0; c < args[0].columns; c++) {
				max = Math.max(max, args[0].get(r,c));
			}

			// And softmax as normal.
			float accumulator = 0;
			for(int c = 0; c < args[0].columns; c++) {
				accumulator += Math.exp(args[0].get(r, c)-max);
			}
			for(int c = 0; c < args[0].columns; c++) {
				m.set(r, c, (float)Math.exp(args[0].get(r, c)-max)/accumulator);
			}
		}
		return m;
	}

	public Matrix[] reverse(Matrix[] fwd, Matrix adjoint) {
		// Have to sum up the adjoints from all the rows, taking slices.
		Matrix newAdjoint = new Matrix(fwd[0].rows, fwd[0].columns);
		Matrix x = this.forward(fwd);
		for(int r = 0; r < this.rows; r++) {
			for(int c = 0; c < adjoint.columns; c++) {
				//newAdjoint.set(r, c, newAdjoint.get(r,c) + (float)(adjoint.get(r,c)-x.get(r,c)*fwd[0].get(r,c))); // Goes to NaN.
				//newAdjoint.set(r, c, newAdjoint.get(r,c) + (float)(adjoint.get(r,c)-x.get(r,c))); // Goes in the opposite direction.
				//newAdjoint.set(r, c, newAdjoint.get(r,c) + (float)(x.get(r,c) - adjoint.get(r,c))); // Error spirals off into oblivion.  Wrong.
				//newAdjoint.set(r, c, newAdjoint.get(r,c) + (float)(adjoint.get(r,c)*fwd[0].get(r,c))); // Nope.
				//newAdjoint.set(r, c, newAdjoint.get(r,c) + (float)(x.get(r,c)*(adjoint.get(r,c)-x.get(r,c))*fwd[0].get(r,c))); // Doesn't look like it's moving.
				//newAdjoint.set(r, c, newAdjoint.get(r,c) + (float)(x.get(r,c)*(adjoint.get(r,c)-x.get(r,c)))); // Gets really close and then flies into NaN.
				//newAdjoint.set(r, c, newAdjoint.get(r,c) + (float)(x.get(r,c)*(x.get(r,c)-adjoint.get(r,c)))); // Starts producing all zeros on output.  Presumably drives inputs to inf.
				//newAdjoint.set(r, c, newAdjoint.get(r,c) + (float)(fwd[0].get(r,c)*(adjoint.get(r,c)-fwd[0].get(r,c)))); // Goes to NaN after first step.
				newAdjoint.set(r, c, newAdjoint.get(r,c) + (float)(adjoint.get(r,c)*(1.0f-x.get(r,c)))); // Seems to converge to the right answer and never leave.  Maybe right?
				//newAdjoint.set(r, c, newAdjoint.get(r,c) + (float)(adjoint.get(r,c)*x.get(r,c))); // Most mathematically rigorous?  Seems like it works with a very low learning rate. 
				//newAdjoint.set(r, c, newAdjoint.get(r,c) + (float)adjoint.get(r,c)); // Seems to pass grad test, but flies off to NaN and doesn't converge on right answer.
			}
		}
		return new Matrix[]{newAdjoint};
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}
