import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.networks.MaxPoolingNetwork;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Created by Jo on 5/28/2015.
 */
public class MaxPoolingTest {
	@Test
	public void testMax() {
		Matrix m = new Matrix(2, 3);
		assertEquals(m.numRows(), 2);
		assertEquals(m.numColumns(), 3);
		m.set(0, 0, 1.0);
		m.set(1, 2, 4.0);
		MaxPoolingNetwork mpn = new MaxPoolingNetwork(2*3);
		Matrix pred = mpn.predict(m.reshape_i(1, 2 * 3));
		assertEquals(pred.numRows(), 1);
		assertEquals(pred.numColumns(), 1);
		assertEquals("Max works.", pred.get(0, 0), 4.0, 1e-5);
	}
}
