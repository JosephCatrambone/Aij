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
		Matrix m = new Matrix(new double[][] {
				{0.0},
				{0.1},
				{0.2},
				{0.3},
				{0.4},
				{0.5},
				{0.6},
				{0.7},
				{0.8},
				{0.9},
				{1.0}
		});

		Matrix bitMatrix = ImageTools.GreyMatrixToBitMatrix(m);
		Matrix m2 = ImageTools.BitMatrixToGrayMatrix(bitMatrix, 0.99);

		System.out.println(m);
		System.out.println(m2);

		//System.out.println(m2.subtract(m));

		assertArrayEquals("f^-1(f(x)) != x :", m.getRowArray(0), m2.getRowArray(0), 1e-1);
		assertArrayEquals("f^-1(f(x)) != x :", m.getRowArray(1), m2.getRowArray(1), 1e-1);
	}
}
