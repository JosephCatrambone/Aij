package com.josephcatrambone.aij;

import java.io.Serializable;
import java.util.Arrays;
import java.util.StringJoiner;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;

/**
 * Created by josephcatrambone on 1/13/17.
 */
public class Matrix implements Serializable {
	public int rows;
	public int columns;
	public float[] data;

	public static Matrix identity(int rows) {
		return new Matrix(rows, rows, (a,b) -> { if(a==b) {return 1.0f;} else {return 0.0f;} } );
	}

	public static Matrix ones(int rows, int columns) {
		return new Matrix(rows, columns, (a,b) -> 1.0f);
	}

	public Matrix() {
		this.rows = 0;
		this.columns = 0;
		this.data = new float[0];
	}

	public Matrix(int rows, int columns) {
		this.rows = rows;
		this.columns = columns;
		this.data = new float[rows*columns];
	}

	public Matrix(int rows, int columns, float[] data) {
		this.rows = rows;
		this.columns = columns;
		this.data = data;
	}

	public Matrix(int rows, int columns, BiFunction<Integer, Integer, Float> initFunction) {
		this.rows = rows;
		this.columns = columns;
		this.data = new float[rows*columns];
		for(int r=0; r < rows; r++) {
			for(int c=0; c < columns; c++) {
				data[c+r*columns] = initFunction.apply(r, c);
			}
		}
	}

	public float get(int r, int c) {
		return this.data[c + r*columns];
	}

	public void set(int r, int c, float n) {
		this.data[c + r*columns] = n;
	}

	public void elementOp_i(UnaryOperator<Float> op) {
		for(int i=0; i < data.length; i++) {
			data[i] = op.apply(data[i]);
		}
	}

	public Matrix elementOp(UnaryOperator<Float> op) {
		Matrix m = new Matrix(this.rows, this.columns, Arrays.copyOf(this.data, rows*columns));
		m.elementOp_i(op);
		return m;
	}

	public void elementOp_i(Matrix other, BinaryOperator<Float> op) {
		for(int i=0; i < data.length; i++) {
			data[i] = op.apply(data[i], other.data[i]);
		}
	}

	public Matrix elementOp(Matrix other, BinaryOperator<Float> op) {
		Matrix m = new Matrix(this.rows, this.columns, Arrays.copyOf(this.data, rows*columns));
		m.elementOp_i(other, op);
		return m;
	}

	public Matrix matmul(Matrix other) {
		Matrix result = new Matrix(this.rows, other.columns);
		for(int i=0; i < rows; i++) {
			for(int j=0; j < other.columns; j++) {
				float accumulator = 0;
				for(int k=0; k < this.columns; k++) {
					// TODO: This must be made generic.
					accumulator += this.get(i,k)*other.get(k,j);
				}
				result.set(i,j, accumulator);
			}
		}
		return result;
	}

	public Matrix transpose() {
		return new Matrix(this.columns, this.rows, (r,c) -> this.get(c, r));
	}

	public Matrix getSlice(int startRow, int endRow, int startColumn, int endColumn) {
		if(startRow < 0) { startRow = this.rows+startRow; }
		if(endRow < 0) { endRow = this.rows+endRow; }
		if(startColumn < 0) { startColumn = this.columns+startColumn; }
		if(endColumn < 0) { endColumn = this.columns+endColumn; }

		Matrix slice = new Matrix(endRow-startRow, endColumn-startColumn);
		for(int row=startRow; row < endRow; row++) {
			for(int column=startColumn; column < endColumn; column++) {
				slice.set(row-startRow, column-startColumn, get(row, column));
			}
		}
		return slice;
	}

	public void setSlice(int startRow, int startColumn, Matrix m) {
		int endRow = startRow+m.rows;
		int endColumn = startColumn+m.columns;
		for(int row=startRow; row < endRow; row++) {
			for(int column=startColumn; column < endColumn; column++) {
				set(row, column, m.get(row-startRow, column-startColumn));
			}
		}
	}

	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("[MATRIX]");
		sb.append("[");
		sb.append(rows);
		sb.append(",");
		sb.append(columns);
		sb.append("]");

		StringJoiner sj = new StringJoiner(",", "[", "");
		for(float f : data) {
			sj.add(""+f);
		}
		sb.append(sj.toString());
		return sb.toString();
	}

	public static Matrix fromString(String s) {
		String[] tokens = s.split("]\\[");
		if(!tokens[0].equals("[MATRIX")) {
			System.err.println("Failed to deserialize Matrix from string.  Expected '[MATRIX'.  Got '" + tokens[0] + "'");
			return null;
		}

		String[] dims = tokens[1].split(",");
		String[] data = tokens[2].split(",");
		Matrix m = new Matrix(Integer.parseInt(dims[0]), Integer.parseInt(dims[1]));
		m.data = new float[m.rows*m.columns];
		for(int i=0; i < m.data.length; i++) {
			m.data[i] = Float.parseFloat(data[i]);
		}
		return m;
	}
}
