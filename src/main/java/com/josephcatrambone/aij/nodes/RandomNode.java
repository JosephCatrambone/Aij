package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

import java.util.Random;

public class RandomNode extends Node {
	Random random = new Random();

	public RandomNode() { super(); }

	public RandomNode(Node mean, Node stddev) {
		this.rows = mean.rows;
		this.columns = mean.columns;
		this.inputs = new Node[]{mean, stddev};
	}

	public Matrix forward(Matrix[] args) {
		return args[0].elementOp(args[1], (m,s) -> m*2.0f*(0.5f-random.nextGaussian()) + Math.exp(random.nextGaussian()*s) );
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		return new Matrix[]{adjoint.elementOp(forward[0], (a, b) -> a*b), adjoint.elementOp(forward[1], (a,b) -> a*b)};
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}