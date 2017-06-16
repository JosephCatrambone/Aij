package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

import java.util.Random;

public class VariationalNode extends Node {
	Random random;

	public VariationalNode() { super(); }

	public VariationalNode(Node mean, Node exp) {
		random = new Random();
		this.rows = mean.rows;
		this.columns = mean.columns;
		this.inputs = new Node[]{mean, exp};
	}

	public Matrix forward(Matrix[] args) {
		return args[0].elementOp(args[1], (mu, exp) -> mu + Math.exp(exp)*random.nextGaussian());
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		// z := x+y
		// x += Adjz;  y += Adjz

		// z := f(x)
		// x += adjz*f'(x)

		// z := f(x,y)
		// x += adjz*f'(x,y) wrt x
		// y += adjz*f'(x,y) wrt y

		// f(x,y) = x + e^y
		// df/dx = 1
		// df/dy = e^y
		return new Matrix[]{
			adjoint,
			adjoint.elementOp(forward[1], (a, b) -> a*Math.exp(b))
		};
	}

	public Node getKLLoss() {
		// - 0.5 * sum(1 + z_variance - square(z_mean) - exp(z_variance))
		Node mean = this.inputs[0];
		Node variance = this.inputs[1];
		Node meanSquared = new PowerNode(mean, 2.0);
		Node variancePower = new ExpNode(variance);

		Node varianceLessSquare = new SubtractNode(variance, meanSquared);
		Node varianceLessExp = new SubtractNode(varianceLessSquare, variancePower);

		Node collapsedSum = new CollapseSumNode(new AddNode(new ConstantNode(1.0, varianceLessExp), varianceLessExp));
		return new MultiplyNode(new ConstantNode(-0.5, collapsedSum), collapsedSum);
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}