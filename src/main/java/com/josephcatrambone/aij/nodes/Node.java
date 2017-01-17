package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

import java.util.ArrayList;
import java.util.StringJoiner;

public abstract class Node {
	private static final String STRING_DELIMITER = "|"; // Regex, so we need to delimit like this.

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
		StringJoiner sj = new StringJoiner(STRING_DELIMITER);
		sj.add(this.getClass().getCanonicalName());
		sj.add(""+this.id);
		sj.add(""+this.rows);
		sj.add(""+this.columns);

		StringJoiner sj2 = new StringJoiner(",");
		for(Node n : inputs) {
			sj2.add(""+n.id);
		}
		sj.add(sj2.toString());

		sj.add(extraDataToString());
		return sj.toString();
	}

	public static Node fromString(String s, ArrayList<Node> possibleInputs) {
		String[] tokens = s.split("\\"+STRING_DELIMITER); // Note: Regex!  Not str.  Need to prepend the \\ for the | char.
		Node instance = null;
		try {
			Class cl = Class.forName(tokens[0]);
			instance = (Node)cl.newInstance();
		} catch (ClassNotFoundException e) {
			System.err.println("Failed to load Node class with name " + tokens[0]);
			e.printStackTrace();
			return null;
		} catch(IllegalAccessException ea) {
			System.err.println("Problem instancing new Node class.");
			return null;
		} catch(InstantiationException ie) {
			System.err.println("Problem instancing new Node class.");
			return null;
		}

		instance.id = Integer.parseInt(tokens[1]);
		instance.rows = Integer.parseInt(tokens[2]);
		instance.columns = Integer.parseInt(tokens[3]);

		// If this node has inputs...
		if(tokens.length > 4 && !tokens[4].equals("")) {
			String[] inputIDs = tokens[4].split(",");
			instance.inputs = new Node[inputIDs.length];
			for (int i = 0; i < inputIDs.length; i++) {
				instance.inputs[i] = possibleInputs.get(Integer.parseInt(inputIDs[i]));
			}
		}

		// If this node has extra data...
		if(tokens.length > 5) {
			instance.extraDataFromString(tokens[5]);
		}
		return instance;
	}

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}