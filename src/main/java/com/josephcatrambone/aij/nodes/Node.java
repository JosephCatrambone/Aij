package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public abstract class Node {
	private static final String STRING_DELIMITER = "|";

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

	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getCanonicalName());
		sb.append(STRING_DELIMITER);
		sb.append(this.id);
		sb.append(STRING_DELIMITER);
		sb.append(this.rows);
		sb.append(STRING_DELIMITER);
		sb.append(this.columns);
		sb.append(STRING_DELIMITER);
		for(Node n : inputs) {
			sb.append(n.id);
			sb.append(",");
		}
		sb.append(STRING_DELIMITER);
		sb.append(extraDataToString());
		return sb.toString();
	}

	public void fromString(String s, Node[] possibleInputs) {
		String[] tokens = s.split(STRING_DELIMITER); // Note: Regex!  Not str.
		this.id = Integer.parseInt(tokens[0]);
		this.rows = Integer.parseInt(tokens[1]);
		this.columns = Integer.parseInt(tokens[2]);
		String[] inputIDs = tokens[3].split(",");
		this.inputs = new Node[inputIDs.length];
		for(int i=0; i < inputIDs.length; i++) {
			this.inputs[i] = possibleInputs[Integer.parseInt(inputIDs[i])];
		}
		this.extraDataFromString(tokens[4]);
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}