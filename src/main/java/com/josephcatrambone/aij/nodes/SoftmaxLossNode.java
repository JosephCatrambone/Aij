package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

/**
 * Created by jcatrambone on 6/29/17.
 * NOTE: THIS NODE SHOULD NOT BE USED AS AN INTERMEDIATE LAYER!
 * IT WILL IGNORE THE ADJOINT MATRIX INPUT AND CALCULATE ONE FROM FORWARD VALUES!
 */
public class SoftmaxLossNode extends Node {
	Matrix lastOutput = null;

	/*** SoftmaxLossNode
	 *
	 * @param input
	 * @param target
	 */
	public SoftmaxLossNode(Node input, Node target) {
		super(input.rows, input.columns, new Node[]{input, target});
	}

	@Override
	public Matrix forward(Matrix[] args) {
		Matrix m = new Matrix(this.rows, this.columns);
		for(int r = 0; r < this.rows; r++) {
			// Do this max thing to improve the numerical stability.
			double max = 0;
			for(int c=0; c < args[0].columns; c++) {
				max = Math.max(max, args[0].get(r,c));
			}

			// And softmax as normal.
			double accumulator = 0;
			for(int c = 0; c < args[0].columns; c++) {
				accumulator += Math.exp(args[0].get(r, c)-max);
			}
			for(int c = 0; c < args[0].columns; c++) {
				m.set(r, c, Math.exp(args[0].get(r, c)-max)/accumulator);
			}
		}
		lastOutput = m;
		return m;
	}

	@Override
	public Matrix[] reverse(Matrix[] args, Matrix adjoint) {
		// Assume we've already gotten out forward pass set.
		// TODO: Might introduce a bug if used in parallel.
		// The loss for a 'softmax loss' node is softmaxout - [1 for the correct answer and zero everywhere else].
		// We assume that the target node (should be forward[1]) is set like this.
		if(lastOutput == null) {
			lastOutput = forward(args);
		}
		Matrix inputAdjoint = lastOutput.elementOp(args[1], (sm, lab) -> sm - lab);
		return new Matrix[]{inputAdjoint, new Matrix(args[1].rows, args[1].columns)}; // Should NOT be using the gradients on the labels.
	}
}
