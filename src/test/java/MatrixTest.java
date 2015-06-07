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
	public void testReshape() {
		Matrix m = new Matrix(3, 5);
		assertEquals(m.numRows(), 3);
		assertEquals(m.numColumns(), 5);
		m.reshape_i(15, 1);
		assertEquals(m.numRows(), 15);
		assertEquals(m.numColumns(), 1);
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
