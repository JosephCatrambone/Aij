package com.josephcatrambone.aij.nodes;

import com.josephcatrambone.aij.Matrix;

public class BroadcastNode extends Node {

	int rowRepeats = 0;
	int columnRepeats = 0;

	public BroadcastNode() { super(); }

	public BroadcastNode(Node input, int columnRepeats, int rowRepeats) {
		this.rows = input.rows* rowRepeats;
		this.columns = input.columns* columnRepeats;
		this.rowRepeats = rowRepeats;
		this.columnRepeats = columnRepeats;
		this.inputs = new Node[]{input};
	}

	public BroadcastNode(Node input, Node match) {
		this.rowRepeats = match.rows/input.rows;
		this.columnRepeats = match.columns/input.columns;
		this.rows = input.rows* rowRepeats;
		this.columns = input.columns* columnRepeats;
		assert(this.rows == match.rows);
		assert(this.columns == match.columns);
		this.inputs = new Node[]{input};
	}

	public Matrix forward(Matrix[] args) {
		Matrix m = new Matrix(this.rows, this.columns);
		for(int r = 0; r < rowRepeats; r++) {
			for(int c = 0; c < columnRepeats; c++) {
				m.setSlice(r*args[0].rows, c*args[0].columns, args[0]);
			}
		}
		return m;
	}

	public Matrix[] reverse(Matrix[] forward, Matrix adjoint) {
		// Have to sum up the adjoints from all the rows, taking slices.
		Matrix newAdjoint = new Matrix(forward[0].rows, forward[0].columns);
		for(int r=0; r < rowRepeats; r++) {
			for(int c=0; c < columnRepeats; c++) {
				int startRow = r*forward[0].rows;
				int startColumn = c*forward[0].columns;
				int endRow = startRow+forward[0].rows;
				int endColumn = startColumn+forward[0].columns;
				newAdjoint.elementOp_i(adjoint.getSlice(startRow, endRow, startColumn, endColumn), (a,b) -> a+b);
			}
		}
		return new Matrix[]{newAdjoint};
	}

	// Used to augment serialization.
	public String extraDataToString() { return rowRepeats + "|" + columnRepeats; };
	public void extraDataFromString(String s) {
		String[] vals = s.split("|");
		rowRepeats = Integer.parseInt(vals[0]);
		columnRepeats = Integer.parseInt(vals[1]);
	}
}