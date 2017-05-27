package com.josephcatrambone.aij.optimizers;

import com.josephcatrambone.aij.Graph;
import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.nodes.Node;
import com.josephcatrambone.aij.nodes.VariableNode;

import java.util.Map;

import static java.lang.Math.*;

/**
 * Created by jcatrambone on 5/5/17.
 */
public class Momentum extends Optimizer {

	public double learningRate;
	public double momentum;
	private Matrix[] previousUpdate;

	public Momentum(Graph g, VariableNode[] trainableVariables, double learningRate, double momentum) {
		super(g, trainableVariables);
		this.learningRate = learningRate;
		this.momentum = momentum;
		// Find the maximum ID of the trainable variables and use that to allocate the array.
		// It wastes a little bit of space but means we don't need a special data structure to map stuff.
		int maxId = -1;
		for(VariableNode tv : trainableVariables) {
			maxId = max(maxId, tv.id);
		}
		previousUpdate = new Matrix[maxId+1];
	}

	@Override
	public double minimize(Node loss, Map<Node, Matrix> inputFeed) {
		Matrix[] fwd = graph.forward(inputFeed);
		Matrix[] grads = graph.getGradient(inputFeed, fwd, loss);

		// Apply the gradients, scaled, to each of the learning variables.
		for(VariableNode n : variables) {
			if(previousUpdate[n.id] == null) {
				previousUpdate[n.id] = grads[n.id];
			} else {
				previousUpdate[n.id].elementOp_i(grads[n.id], (oldGrad, newGrad) -> (1.0 - momentum) * newGrad + (momentum) * oldGrad);
			}
			n.getVariable().elementOp_i(previousUpdate[n.id], (w, dw) -> w - learningRate*dw);
		}
		return 0;
	}
}
