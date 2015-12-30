import com.josephcatrambone.aij.Matrix;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by Jo on 5/28/2015.
 */
public class MatrixTest {
	@Test
	public void testTanh() {
		Matrix m = new Matrix(new double[][] {{-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0}});
		Matrix tanhm = new Matrix(new double[][] {{-0.761594, -0.635149, -0.462117, -0.244919, 0.000000, 0.244919, 0.462117, 0.635149, 0.761594}});
		assertArrayEquals("Failure tanh.", m.tanh().getRowArray(0), tanhm.getRowArray(0), 1e-5);

		// Also the tanh() should not be in place, but tanh_i should be.
		m = new Matrix(new double[][] {{-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0}});
		Matrix m2 = new Matrix(new double[][] {{-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0}});
		m.tanh();
		assertArrayEquals("Failure tanh.", m.getRowArray(0), m2.getRowArray(0), 1e-5);
		m.tanh_i();
		assertArrayEquals("Failure tanh.", m.getRowArray(0), tanhm.getRowArray(0), 1e-5);
	}

	@Test
	public void testSigmoid() {
		double[] values = new double[]{-10, -5, -2, -1, -0.75, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10};
		Matrix m = new Matrix(new double[][] {values});
		Matrix m2 = m.sigmoid();
		// Make sure original isn't changed.
		assertArrayEquals(values, m.getRowArray(0), 1.0e-10);
		// Make sure sigmoid computed correctly.
		assertArrayEquals(new double[]{
			4.53978687e-05,   6.69285092e-03,   1.19202922e-01,
			2.68941421e-01,   3.20821301e-01,   3.77540669e-01,
			4.37823499e-01,   4.75020813e-01,   5.00000000e-01,
			5.24979187e-01,   5.62176501e-01,   6.22459331e-01,
			6.79178699e-01,   7.31058579e-01,   8.80797078e-01,
			9.93307149e-01,   9.99954602e-01}, m2.getRowArray(0), 1.0e-4);
	}

	@Test
	public void testReshape() {
		// Verify basic reshape
		Matrix m = new Matrix(3, 5);
		assertEquals(m.numRows(), 3);
		assertEquals(m.numColumns(), 5);
		m.reshape_i(15, 1);
		assertEquals(m.numRows(), 15);
		assertEquals(m.numColumns(), 1);
		// Verify COLUMN-MAJOR reshaping.
		m.reshape_i(1, 15);
		m.setRow(0, new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5});
		m.reshape_i(5, 3);
		assertArrayEquals(m.getRowArray(0), new double[]{1, 6, 1}, 1e-8);
		assertArrayEquals(m.getRowArray(1), new double[]{2, 7, 2}, 1e-8);
		assertArrayEquals(m.getRowArray(2), new double[]{3, 8, 3}, 1e-8);
		assertArrayEquals(m.getRowArray(3), new double[]{4, 9, 4}, 1e-8);
		assertArrayEquals(m.getRowArray(4), new double[]{5, 0, 5}, 1e-8);
	}

	@Test
	public void testColumnMean() {
		Matrix m = new Matrix(2, 4);
		m.setRow(0, new double[]{1, 2, 4, 6});
		m.setRow(1, new double[]{1, 2, 4, 0});
		Matrix mean = m.meanRow();
		assertEquals(mean.numRows(), 1);
		assertEquals(mean.numColumns(), 4);
		assertEquals(mean.get(0, 0), 1.0, 1.0e-9);
		assertEquals(mean.get(0, 1), 2.0, 1.0e-9);
		assertEquals(mean.get(0, 2), 4.0, 1.0e-9);
		assertEquals(mean.get(0, 3), 3.0, 1.0e-9);
	}
}
