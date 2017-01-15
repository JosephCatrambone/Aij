package com.josephcatrambone.aij;

import java.util.ArrayList;
import java.util.HashMap;

import com.josephcatrambone.aij.nodes.*;

/**
 * Created by jcatrambone on 9/12/16.
 */
public class Graph {
	ArrayList<Node> nodes = new ArrayList<>();

	public Node addNode(Node n) {
		n.id = nodes.size();
		for(int inp : n.inputs) {
			if(inp == -1 || inp >= nodes.size()) {
				throw new RuntimeException("Node added to graph without dependencies: ID " + n.id);
			}
		}
		nodes.add(n);
		return n; // A pass-through.
	}

	public String serializeToString() {
		return "";
	}

	public void restoreFromString(String s) {
		throw new RuntimeException("Not implemented.");
	}

	public float[] getOutput(HashMap<Node, float[]> inputs, Node node) {
		HashMap<Node, Matrix> remappedInputs = floatMapToMatrixMap(inputs);
		return forward(remappedInputs)[node.id].data;
	}

	private HashMap<Node, Matrix> floatMapToMatrixMap(HashMap<Node, float[]> map) {
		if(map == null) { return null; }
		HashMap<Node, Matrix> hm = new HashMap<>();
		for(Node k : map.keySet()) {
			hm.put(k, new Matrix(k.rows, k.columns, map.get(k)));
		}
		return hm;
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
					forwardInputs[j] = results[n.inputs[j]];
				}
				results[i] = nodes.get(i).forward(forwardInputs);
			}
		}
		return results;
	}

	/*
	public float[][] getGradient(HashMap<Integer, float[]> inputs, float[][] forward, int node) {
		if(forward == null) {
			HashMap<Integer,Matrix> feed = floatMapToMatrixMap(inputs);
			Matrix[] fwd = forward(feed);
			Matrix[] grad = getGradient(feed, fwd, node);
			float[][] res = new float[nodes.size()][];
			for(int i=0; i < grad.length; i++) {
				res[i] = grad[i].data;
			}
			return res;
		} else {
			// Convert forward to
		}
	}*/

	public Matrix[] getGradient(HashMap<Node, Matrix> inputFeed, Matrix[] fwd, Node node) {
		if(fwd == null) {
			fwd = forward(inputFeed);
		}
		Matrix[] grads = new Matrix[nodes.size()];
		grads[node.id] = Matrix.ones(node.rows, node.columns);
		for(int i=node.id; i >= 0; i--) {
			// For all the inputs to this node, calculate their adjoints from this adjoint.
			Matrix[] argInputs = new Matrix[nodes.get(i).inputs.length];
			for(int j=0; j < argInputs.length; j++) {
				argInputs[j] = fwd[nodes.get(i).inputs[j]];
			}
			Matrix[] nextAdjoints = nodes.get(i).reverse(argInputs, grads[i]);
			for(int j=0; j < nodes.get(i).inputs.length; j++) {
				grads[nodes.get(i).inputs[j]].elementOp_i(nextAdjoints[j], (a,b) -> a+b);
			}
		}
		return grads;
	}
}
