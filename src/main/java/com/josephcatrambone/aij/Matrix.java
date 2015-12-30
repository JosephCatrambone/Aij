package com.josephcatrambone.aij;

import org.jblas.DoubleMatrix;

import java.io.Serializable;
import java.util.Random;
import java.util.function.UnaryOperator;

/**
 * Created by jcatrambone on 5/28/15.
 * This is a generic matrix wrapper tool with most of the methods needed to NN/ML activities.
 * There are some static methods provided to emulate numpy funcionality as required.
 * All in-place operations are suffixed with 'i'.
 * This is a VERY thin wrapper around JBlas.
 */
public class Matrix implements Serializable, Cloneable {
	private DoubleMatrix m;

	// Static methods
	public static Matrix ones(int rows, int columns) {
		return new Matrix(DoubleMatrix.ones(rows, columns));
	}

	public static Matrix zeros(int rows, int columns) {
		return new Matrix(DoubleMatrix.zeros(rows, columns));
	}

	/*** random
	 * Returns a new matrix with random values between 0.0 and 1.0.
	 * @param rows
	 * @param columns
	 * @return
	 */
	public static Matrix random(int rows, int columns) {
		return new Matrix(DoubleMatrix.rand(rows, columns).muli(2.0).subi(1.0));
	}

	public static Matrix concatVertically(Matrix a, Matrix b) {
		return new Matrix(DoubleMatrix.concatVertically(a.m, b.m));
	}

	public static Matrix concatHorizontally(Matrix a, Matrix b) {
		return new Matrix(DoubleMatrix.concatHorizontally(a.m, b.m));
	}

	// Constructors
	private Matrix(DoubleMatrix m) { this.m = m; }

	public Matrix() { m = new DoubleMatrix(); }

	public Matrix(int rows, int columns) {
		m = new DoubleMatrix(rows, columns);
	}

	public Matrix(int rows, int columns, double initial) { this.m = DoubleMatrix.ones(rows, columns).mul(initial); }

	public Matrix(double[][] values) { m = new DoubleMatrix(values); }

	// Utils
	public Matrix clone() {
		return new Matrix(m.dup());
	}

	/*** shuffle_i
	 * Perform a Fisher-Yates shuffle on the rows.
	 */
	public void shuffle_i() {
		Random random = new Random();
		for(int i=0; i < m.getRows(); i++) {
			m.swapRows(i, random.nextInt(m.getRows()-i)+i+1);
		}
	}

	// Getters/setters
	public String toString() {
		return m.toString();
	}

	public int numRows() { return m.rows; }

	public int numColumns() { return m.columns; }

	public double get(int i, int j) { return m.get(i, j); }

	public void set(int i, int j, double value) { m.put(i, j, value); }

	public Matrix getRow(int row) {
		return new Matrix(m.getRow(row));
	}

	public Matrix getColumn(int column) { return new Matrix(m.getColumn(column)); }

	public double[] getRowArray(int row) {
		double[] output = new double[m.columns];
		for(int i=0; i < output.length; i++) {
			output[i] = m.get(row, i);
		}
		return output;
	}

	public Matrix getRows(int[] rows) {
		Matrix res = new Matrix(rows.length, numColumns());
		for(int i=0; i < rows.length; i++) {
			res.m.putRow(i, m.getRow(rows[i]));
		}
		return res;
	}

	public void setRow(int row, Matrix value) {
		m.putRow(row, value.m);
	}

	public void setRow(int row, double[] values) {
		m.putRow(row, new DoubleMatrix(values));
	}

	/*** appendRow
	 * Add a row after all the others.  There will be an array copy no matter what, so there is no in-place.
	 * Generally, you're probably better off using the concatVertically method for Matrices.
	 * @param values
	 * @return
	 */
	public Matrix appendRow(double[] values) {
		DoubleMatrix m2 = new DoubleMatrix(m.rows+1, m.columns);

		for(int i=0; i < m.rows; i++) {
			m2.putRow(i, m.getRow(i));
		}
		m2.putRow(m.rows, new DoubleMatrix(values));

		return new Matrix(m2);
	}

	public Matrix getSubmatrix(int startRow, int endRow, int startColumn, int endColumn) {
		return new Matrix(m.getRange(startRow, endRow, startColumn, endColumn));
	}

	/*** Blit a matrix into this matrix.
	 * @param other
	 * @param startRow
	 * @param startColumn
	 * @return
	 */
	public Matrix setSubmatrix_i(Matrix other, int startRow, int startColumn) {
		for(int y=0; y < other.numRows(); y++) {
			for(int x=0; x < other.numColumns(); x++) {
				this.m.put(y+startRow, x+startColumn, other.get(y, x));
			}
		}
		return this;
	}

	// Operations
	public Matrix add_i(double value) {
		m.addi(value);
		return this;
	}

	public Matrix add(double value) {
		return new Matrix(m.add(value));
	}

	public Matrix add_i(Matrix other) {
		m.addi(other.m);
		return this;
	}

	public Matrix add(Matrix other) {
		return new Matrix(m.add(other.m));
	}

	public Matrix addSubmatrix_i(Matrix other, int startRow, int startColumn) {
		return addSubmatrix_i(other, startRow, startColumn, false);
	}

	public Matrix addSubmatrix_i(Matrix other, int startRow, int startColumn, boolean skipOutOfBounds) {
		for(int y=0; y < other.numRows(); y++) {
			for(int x=0; x < other.numColumns(); x++) {
				if(y+startRow < 0 || x+startColumn < 0 || y+startRow >= m.getRows() || x+startColumn >= m.getColumns()) {
					if(skipOutOfBounds) {
						continue;
					}
				}
				this.m.put(y+startRow, x+startColumn, m.get(y+startRow, x+startColumn)+other.m.get(y, x));
			}
		}
		return this;
	}

	public Matrix subtract(Matrix other) {
		return new Matrix(m.sub(other.m));
	}

	public Matrix subtract_i(Matrix other) {
		m.subi(other.m);
		return this;
	}

	public Matrix multiply_i(Matrix other) {
		m.mmuli(other.m);
		return this;
	}

	public Matrix multiply(Matrix other) {
		return new Matrix(m.mmul(other.m));
	}

	public Matrix elementMultiply(double value) {
		return new Matrix(m.mul(value));
	}

	public Matrix elementMultiply(Matrix other) {
		return new Matrix(m.mul(other.m));
	}

	public Matrix elementMultiply_i(double value) {
		m.muli(value);
		return this;
	}

	public Matrix elementMultiply_i(Matrix other) {
		m.muli(other.m);
		return this;
	}

	public Matrix inverse_i() {
		m.rdivi(1.0);
		return this;
	}

	public Matrix transpose() {
		return new Matrix(m.transpose());
	}

	public Matrix reshape_i(int rows, int columns) {
		m.reshape(rows, columns);
		return this;
	}

	public Matrix neg_i() {
		m.negi();
		return this;
	}

	public double sum() {
		return m.sum();
	}

	/*** sumCols
	 * Squash each column into a single number.
	 * [1 2 3]
	 * [4 5 6] -> [5 7 9]
	 * @return
	 */
	public Matrix sumColumns() {
		return new Matrix(m.columnSums());
	}

	/*** meanRow
	 * Get the average row value. (sum all columns, divide by num columns.)
	 * Average row value = mean of each column.
	 * [1 2 3]
	 * [4 5 6] -> [2.5 3.5 4.5]
	 * @return
	 */
	public Matrix meanRow() {
		return new Matrix(m.columnMeans());
	}

	/*** max
	 * Get the maximum value of the matrix.
	 * @return
	 */
	public double max() { return this.m.max(); }

	/*** min
	 * Get the minimum value of the matrix.
	 * @return
	 */
	public double min() { return this.m.min(); }

	/*** normalize
	 *
	 * @return
	 */
	public void normalize_i() {
		double min = Double.MAX_VALUE;
		double max = -Double.MAX_VALUE;
		for(int i=0; i < m.rows; i++) {
			for(int j=0; j < m.columns; j++) {
				double value = m.get(i, j);
				if(value < min) { min = value; }
				if(value > max) { max = value; }
			}
		}
		for(int i=0; i < m.rows; i++) {
			for(int j=0; j < m.columns; j++) {
				m.put(i, j, (m.get(i, j)-min)/(max-min));
			}
		}
	}

	public Matrix repmat(int rows, int cols) {
		return new Matrix(m.repmat(rows, cols));
	}

	// Some more specialized operations
	// We're doing these operations inside the matrix class because there's a chance we can optimize transparently.
	public Matrix tanh() {
		return this.elementOp(x -> Math.tanh(x));
	}

	public Matrix tanh_i() {
		//List<Double> dataCollection = m.elementsAsList();
		//dataCollection.parallelStream().forEachOrdered(x -> Math.tanh(x));
		//return new Matrix(new DoubleMatrix(dataCollection));
		this.elementOp_i(x -> Math.tanh(x));
		return this;
	}

	public Matrix dtanhFromActivation() {
		return this.elementOp(x -> (1 - x * x));
	}

	public Matrix sigmoid() {
		return this.elementOp(x -> 1.0 / (1.0 + Math.exp(-x)));
	}

	public Matrix sigmoid_i() {
		return this.elementOp_i(x -> 1.0 / (1.0 + Math.exp(-x)));
	}


	public Matrix dsigmoidFromActivation() {
		return this.elementOp(sigx -> sigx * (1 - sigx));
	}

	public Matrix softplus() {
		return this.elementOp(x -> Math.log(1 + Math.exp(x)));
	}

	public Matrix softplus_i() {
		return this.elementOp_i(x -> Math.log(1 + Math.exp(x)));
	}

	public Matrix dsoftplusFromActivation() {
		// Derivative of log(1 + e^x) = sigmoid.
		return null;
	}

	/*** elementOp_i
	 * Run an operator in place on each of the elements of this matrix.
	 * MUCH slower than the above ops, but general purpose.
	 * @param op
	 */
	public Matrix elementOp_i(UnaryOperator<Double> op) {
		for(int y=0; y < m.rows; y++) {
			for(int x=0; x < m.columns; x++) {
				m.put(y, x, op.apply(m.get(y, x)));
			}
		}
		return this;
	}

	/*** elementOp
	 * Run an operator on each of the elements of this matrix.
	 * MUCH slower than the above ops, but general purpose.
	 * @param op
	 */
	public Matrix elementOp(UnaryOperator<Double> op) {
		DoubleMatrix output = new DoubleMatrix(m.rows, m.columns);
		for(int y=0; y < m.rows; y++) {
			for(int x=0; x < m.columns; x++) {
				output.put(y, x, op.apply(m.get(y, x)));
			}
		}
		return new Matrix(output);
	}
}
