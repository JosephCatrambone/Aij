import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.utilities.ImageTools;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

/**
 * Created by josephcatrambone on 8/17/15.
 */
public class ImageToolsTest {
	@Test
	public void testImageBitConversion() {
		double totalError = 0;
		Matrix bitMatrix;
		Matrix m2;
		Matrix diff;
		Matrix m = new Matrix(new double[][] {
				{0.0, 1.0},
				{0.1, 0.9},
				{0.2, 0.8},
				{0.3, 0.7},
				{0.4, 0.6},
				{0.5, 0.5},
				{0.6, 0.4},
				{0.7, 0.3},
				{0.8, 0.2},
				{0.9, 0.1},
				{1.0, 0.0}
		});

		for(Integer i : new int[]{4, 8, 16}) {
			// 16 bits
			int bits = i;
			bitMatrix = ImageTools.GreyMatrixToBitMatrix(m, bits);
			m2 = ImageTools.BitMatrixToGrayMatrix(bitMatrix, 0.9, bits);
			diff = m2.subtract(m);
			totalError = diff.sum();
			assertArrayEquals(bits + "-bit conversion diff " + diff + " error: " + totalError, m.getRowArray(0), m2.getRowArray(0), 1e-1);
			assertArrayEquals(bits + "-bit conversion diff " + diff + " error: " + totalError, m.getRowArray(1), m2.getRowArray(1), 1e-1);
		}

		//System.out.println(m);
		//System.out.println(m2);


	}
}
