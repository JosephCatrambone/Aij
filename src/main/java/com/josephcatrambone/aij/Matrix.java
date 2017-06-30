package com.josephcatrambone.aij;

import java.io.Serializable;
import java.util.Arrays;
import java.util.StringJoiner;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;
import java.util.stream.IntStream;

/**
 * Created by josephcatrambone on 1/13/17.
 */
public class Matrix implements Serializable {
	public int rows;
	public int columns;
	public double[] data;

	public static Matrix identity(int rows) {
		return new Matrix(rows, rows, (a,b) -> { if(a==b) {return 1.0;} else {return 0.0;} } );
	}

	public static Matrix ones(int rows, int columns) {
		return new Matrix(rows, columns, (a,b) -> 1.0);
	}

	public Matrix() {
		this.rows = 0;
		this.columns = 0;
		this.data = new double[0];
	}

	public Matrix(int rows, int columns) {
		this.rows = rows;
		this.columns = columns;
		this.data = new double[rows*columns];
	}

	public Matrix(int rows, int columns, double[] data) {
		this.rows = rows;
		this.columns = columns;
		this.data = data;
	}

	public Matrix(int rows, int columns, BiFunction<Integer, Integer, Double> initFunction) {
		this.rows = rows;
		this.columns = columns;
		this.data = new double[rows*columns];
		/*
		Arrays.parallelSetAll(data, (int i) -> {
			int c = i%columns;
			int r = i/columns;
			return initFunction.apply(r, c);
		});
		*/
		for(int r=0; r < this.rows; r++) {
			for(int c=0; c < this.columns; c++) {
				this.set(r, c, initFunction.apply(r, c));
			}
		}
	}

	public double get(int r, int c) {
		return this.data[c + r*columns];
	}

	public void set(int r, int c, double n) {
		this.data[c + r*columns] = n;
	}

	public void elementOp_i(UnaryOperator<Double> op) {
		// Parallel version.
		//IntStream.range(0, this.data.length).parallel().forEach(i -> this.data[i] = op.apply(this.data[i]));
		// Serial version.
		for(int i=0; i < this.data.length; i++) { this.data[i] = op.apply(this.data[i]); }
	}

	public Matrix elementOp(UnaryOperator<Double> op) {
		// Parallel version:
		/*
		return new Matrix(this.rows, this.columns,
			Arrays.stream(this.data).parallel().map(x -> op.apply(x)).toArray()
		);
		*/
		// Serial is faster than parallel?
		return new Matrix(this.rows, this.columns, (r,c) -> op.apply(get(r,c)));
	}

	public void elementOp_i(Matrix other, BinaryOperator<Double> op) {
		// Parallel version:
		/*
		this.data = IntStream.range(0, data.length).parallel().mapToDouble(
			i -> op.apply(data[i], other.data[i])
		).toArray();
		*/
		// Serial version:
		for(int i=0; i < this.data.length; i++) {
			this.data[i] = op.apply(this.data[i], other.data[i]);
		}
	}

	public Matrix elementOp(Matrix other, BinaryOperator<Double> op) {
		// Parallel version:
		/*
		return new Matrix(this.rows, this.columns, IntStream.range(0, data.length).parallel().mapToDouble(
			i -> op.apply(this.data[i], other.data[i])
		).toArray());
		*/
		return new Matrix(this.rows, this.columns, (r, c) -> op.apply(this.get(r, c), other.get(r, c)));
	}

	public Matrix matmul(Matrix other) {
		// TODO: This function consumes around 60% of CPU time.
		Matrix result = new Matrix(this.rows, other.columns);

		for (int i = 0; i < rows; i++) {
			//IntStream.range(0, rows).parallel().forEach( i -> {
			for (int j = 0; j < other.columns; j++) {
				double accumulator = 0;
				for (int k = 0; k < this.columns; k++) {
					// TODO: This must be made generic.
					// TODO: The get currently consumes 32% of CPU time.
					accumulator += this.get(i, k) * other.get(k, j);
				}
				result.set(i, j, accumulator);
			}
			//});
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

	public double[] getRow(int r) {
		return Arrays.copyOfRange(this.data, r*this.columns, (r+1)*this.columns);
	}

	public void setRow(int r, double[] data) {
		assert(data.length == this.columns);
		System.arraycopy(data, 0, this.data, r*this.columns, data.length);
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
		for(double f : data) {
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
		m.data = new double[m.rows*m.columns];
		for(int i=0; i < m.data.length; i++) {
			m.data[i] = Double.parseDouble(data[i]);
		}
		return m;
	}

	/*
	// Unfortuntely, this pretty consistently crashes my machine.
	//import com.aparapi.Kernel;
	//import com.aparapi.Range;

	private class MultiplyKernel extends Kernel {
		final int aRows;
		final int aColumns;
		final int bRows;
		final int bColumns;
		final float[] aData;
		final float[] bData;
		final float[] cData;

		//Usage:
		//MultKern mk = MuliplyKernel(a, b);
		//mk.execute(c.data.length);
		//mk.copyResultToMatrix(c);

		public MultiplyKernel(Matrix a, Matrix b) {
			aRows = a.rows;
			aColumns = a.columns;
			bRows = b.rows;
			bColumns = b.columns;
			aData = new float[a.data.length];
			bData = new float[b.data.length];
			cData = new float[a.rows*b.columns];

			// TODO: FP64 not supported.  Have to copy arrays.  :/
			for(int i=0; i < aData.length; i++) { aData[i] = (float)a.data[i]; }
			for(int i=0; i < bData.length; i++) { bData[i] = (float)b.data[i]; }
			//for(int i=0; i < aData.length; i++) { cData[i] = (float)a.data[i]; }
		}

		public void copyResultToMatrix(Matrix c) {
			for(int i=0; i < cData.length; i++) { c.data[i] = cData[i]; }
		}

		@Override
		public void run() {
			int i = getGlobalId(0); // a-th Row
			int j = getGlobalId(1); // b-th Column
			int k = getGlobalId(2);
			// C[gid] -> gid = x+y*w = col + row*bColumns
			cData[j + i*bColumns] += aData[k + i*aColumns]*bData[j + k*bColumns];
		}
	}
	*/
}
