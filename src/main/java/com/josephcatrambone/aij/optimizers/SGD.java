package com.josephcatrambone.aij.optimizers;

import com.josephcatrambone.aij.Graph;
import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.nodes.Node;
import com.josephcatrambone.aij.nodes.VariableNode;

import java.util.Map;

/**
 * Created by jcatrambone on 5/5/17.
 */
public class SGD extends Optimizer {

	public double learningRate;

	public SGD(Graph g, VariableNode[] trainableVariables, double learningRate) {
		super(g, trainableVariables);
		this.learningRate = learningRate;
	}

	@Override
	public double minimize(Node loss, Map<Node, Matrix> inputFeed) {
		Matrix[] fwd = graph.forward(inputFeed);
		Matrix[] grads = graph.getGradient(inputFeed, fwd, loss);

		// Apply the gradients, scaled, to each of the learning variables.
		for(VariableNode n : variables) {
			n.getVariable().elementOp_i(grads[n.id], (w, dw) -> w - learningRate*dw);
		}
		return 0;
	}
}
