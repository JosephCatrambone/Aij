package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public abstract class Node {
	public int id = -1;
	public Node[] inputs; // int[] makes it easier to serialize, but it's more of a hassle to build the graph.
	public int rows, columns;

	public Node(){}

	public Node(Node input) {
		// Unary operator constructor.
		this.rows = input.rows;
		this.columns = input.columns;
		this.inputs = new Node[]{input};
	}

	public Node(int rows, int columns, Node... inputs) {
		this.rows = rows;
		this.columns = columns;
		this.inputs = inputs;
	}

	public abstract Matrix forward(Matrix[] args);
	public abstract Matrix[] reverse(Matrix[] forward, Matrix adjoint);

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}