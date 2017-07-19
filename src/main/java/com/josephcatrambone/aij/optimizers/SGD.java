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

	private int gradCount = 0;
	public double learningRate;

	public SGD(Graph g, VariableNode[] trainableVariables, double learningRate) {
		super(g, trainableVariables);
		this.learningRate = learningRate;
	}

	@Override
	public void accumulateGradients(Node loss, Map<Node, Matrix> inputFeed) {
		Matrix[] fwd = graph.forward(inputFeed);
		Matrix[] grads = graph.getGradient(inputFeed, fwd, loss);

		if(accumulatedGradients == null) {
			gradCount = 1;
			accumulatedGradients = grads;
		} else {
			gradCount++;
			for(VariableNode n : variables) {
				accumulatedGradients[n.id].elementOp_i(grads[n.id], (w1, w2) -> w1 + w2);
			}
		}
	}

	@Override
	public void applyGradients() {
		for(VariableNode n : variables) {
			n.getVariable().elementOp_i(accumulatedGradients[n.id], (w, dw) -> w - learningRate*(dw/((float)gradCount)));
		}
	}

	@Override
	public void clearGradients() {
		accumulatedGradients = null;
	}

	@Override
	public double minimize(Node loss, Map<Node, Matrix> inputFeed) {
		Matrix[] fwd = graph.forward(inputFeed);
		Matrix[] grads = graph.getGradient(inputFeed, fwd, loss);

		// Apply the gradients, scaled, to each of the learning variables.
		for(VariableNode n : variables) {
			n.getVariable().elementOp_i(grads[n.id], (w, dw) -> w - learningRate*dw);
		}
		return fwd[loss.id].data[0];
	}
}
