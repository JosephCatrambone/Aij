package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public abstract class Node {
	public int id = -1;
	public int[] inputs; // Input IDs.
	public int rows, columns;
	public abstract Matrix forward(Matrix[] args);
	public abstract Matrix[] reverse(Matrix[] forward, Matrix adjoint);

	// Used to augment serialization.
	public String extraDataToString() { return ""; };
	public void extraDataFromString(String s) {}
}