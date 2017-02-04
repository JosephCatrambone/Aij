package com.josephcatrambone.aij;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Stack;

import com.josephcatrambone.aij.nodes.*;

/**
 * Created by jcatrambone on 9/12/16.
 */
public class Graph {
	ArrayList<Node> nodes = new ArrayList<>();

	public Node addNode(Node n) {
		// Make sure all the dependencies happen first.
		for(Node inp : n.inputs) {
			if(inp.id == -1) {
				this.addNode(inp);
			}
		}
		n.id = nodes.size();
		nodes.add(n);
		return n; // A pass-through.
	}

	public String serializeToString() {
		StringBuilder s = new StringBuilder();
		for(Node n : nodes) {
			s.append(n.toString() + "\n");
		}
		return s.toString();
	}

	public void restoreFromString(String s) {
		String[] lines = s.split("[\\r\\n]+"); // Also removes empty lines.
		// Check for blank first line.
		if(lines[0].equals("")) {
			lines = Arrays.copyOfRange(lines, 1, lines.length); // Remove first line.
		}
		// Create new array and populate.
		nodes = new ArrayList<>(lines.length); // Initial capacity = lines.length.
		for(int i=0; i < lines.length; i++) {
			nodes.add(i, Node.fromString(lines[i], nodes));
		}
	}

	public double[] getOutput(HashMap<Node, double[]> inputs, Node node) {
		// getOutput is different slightly from forward in that we don't care about unused paths.
		// For forward, we want to be sure _all_ different node values are populated in order.
		// getOutput spends a cycle or two figuring out which paths it can ignore (so we don't pay for training paths).
		Stack<Node> toProcess = new Stack<>();
		Matrix[] results = new Matrix[node.id+1];

		// Traverse our graph from the output to the inputs.
		toProcess.push(node);
		while(!toProcess.empty()) {
			Node n = toProcess.pop();
			// Allocate an empty matrix for our results OR copy from input.
			if(n instanceof InputNode) {
				results[n.id] = new Matrix(n.rows, n.columns, inputs.get(n));
			} else {
				results[n.id] = new Matrix(n.rows, n.columns);
			}
			for(Node inp : n.inputs) {
				toProcess.push(inp);
			}
		}

		// Compute the values, skipping the dead nodes.
		for(int i=0; i < results.length; i++) {
			Node n = nodes.get(i);
			if(results[i] == null || n instanceof InputNode) { continue; }
			// Compile an array of values to be passed into the node.
			Matrix[] forwardInputs = new Matrix[n.inputs.length];
			for(int j=0; j < forwardInputs.length; j++) {
				forwardInputs[j] = results[n.inputs[j].id];
			}
			results[i] = nodes.get(i).forward(forwardInputs);
		}
		return results[node.id].data;

		/* // If we didn't care about path pruning, we could just do this:
		HashMap<Node, Matrix> remappedInputs = doubleMapToMatrixMap(inputs);
		return forward(remappedInputs)[node.id].data;
		*/
	}

	public Matrix[] forward(HashMap<Node, Matrix> datafeed) {
		Matrix[] results = new Matrix[nodes.size()];
		for(int i=0; i < nodes.size(); i++) {
			// Special case: inputs read from the input map.
			if(nodes.get(i) instanceof InputNode) {
				results[i] = datafeed.get(nodes.get(i));
			} else {
				// Compile an array of values to be passed into the node.
				Node n = nodes.get(i);
				Matrix[] forwardInputs = new Matrix[n.inputs.length];
				for(int j=0; j < forwardInputs.length; j++) {
					forwardInputs[j] = results[n.inputs[j].id];
				}
				results[i] = nodes.get(i).forward(forwardInputs);
			}
		}
		return results;
	}

	/*
	public double[][] getGradient(HashMap<Integer, double[]> inputs, double[][] forward, int node) {
		if(forward == null) {
			HashMap<Integer,Matrix> feed = doubleMapToMatrixMap(inputs);
			Matrix[] fwd = forward(feed);
			Matrix[] grad = getGradient(feed, fwd, node);
			double[][] res = new double[nodes.size()][];
			for(int i=0; i < grad.length; i++) {
				res[i] = grad[i].data;
			}
			return res;
		} else {
			// Convert forward to
		}
	}*/

	/***
	 * Calculate the gradient with respect to the given node.
	 * @param inputFeed A Hash Map of the input node -> matrix values.
	 * @param fwd The values from the forward pass if already computed.  If null, will compute them.
	 * @param node The value with respect to which we want the gradient.
	 * @return Returns an array of matrices wherein Matrix[node.id] corresponds to the node's gradient.
	 */
	public Matrix[] getGradient(HashMap<Node, Matrix> inputFeed, Matrix[] fwd, Node node) {
		// If forward pass isn't calculated, do that.
		if(fwd == null) {
			fwd = forward(inputFeed);
		}

		// Populate our initial adjoints/gradients.
		Matrix[] grads = new Matrix[nodes.size()];
		grads[node.id] = Matrix.ones(node.rows, node.columns); // The output/target gets 1.0.
		// Everything else gets 0.0.
		for(int i=0; i < node.id; i++) {
			grads[i] = new Matrix(nodes.get(i).rows, nodes.get(i).columns);
		}

		// Starting from the out and propagating backwards, calculate the adjoints.
		for(int i=node.id; i >= 0; i--) {
			// For all the inputs to this node, calculate their adjoints from this adjoint.
			Matrix[] argInputs = new Matrix[nodes.get(i).inputs.length];
			for(int j=0; j < argInputs.length; j++) {
				argInputs[j] = fwd[nodes.get(i).inputs[j].id];
			}
			Matrix[] nextAdjoints = nodes.get(i).reverse(argInputs, grads[i]);
			for(int j=0; j < nodes.get(i).inputs.length; j++) {
				grads[nodes.get(i).inputs[j].id].elementOp_i(nextAdjoints[j], (a,b) -> a+b);
			}
		}
		return grads;
	}
}
