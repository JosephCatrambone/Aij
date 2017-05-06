package com.josephcatrambone.aij.optimizers;

import com.josephcatrambone.aij.Graph;
import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.nodes.Node;
import com.josephcatrambone.aij.nodes.VariableNode;

import java.util.Map;

/**
 * Created by jcatrambone on 5/5/17.
 */
public abstract class Optimizer {
	public Graph graph;
	public VariableNode[] variables;
	public Optimizer(Graph g, VariableNode[] variables) {
		this.graph = g;
		this.variables = variables;
	}

	public abstract double minimize(Node loss, Map<Node, Matrix> inputFeed);
}
