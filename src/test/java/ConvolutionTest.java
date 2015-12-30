import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.networks.ConvolutionalNetwork;
import com.josephcatrambone.aij.networks.FunctionNetwork;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Created by Jo on 5/28/2015.
 */
public class ConvolutionTest {
	@Test
	public void test2DPrediction() {
		// 1 2 3 4 5
		// 6 7 8 9 0
		// 1 2 3 4 5
		// 6 7 8 9 0
		// 1 2 3 4 5
		Matrix m = new Matrix(new double[][] {{1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5}});
		FunctionNetwork oneToOne = new FunctionNetwork(9, 1); // 3x3 -> 1x1
		ConvolutionalNetwork cn = new ConvolutionalNetwork(oneToOne, 5, 5, 3, 3, 1, 1, 1, 1, ConvolutionalNetwork.EdgeBehavior.ZEROS);

		oneToOne.predictionFunction = (Matrix k) -> new Matrix(1, 1, k.sum());

		assertEquals(oneToOne.predict(new Matrix(new double[][]{{1, 2, 3}})).get(0, 0), 6.0, 1.0e-7);

		Matrix result = cn.predict(m);

		System.out.println(result);
		assertArrayEquals(result.getRowArray(0), new double[]{
				16, 27, 33, 29, 18,
				19, 33, 42, 41, 27,
				29, 48, 57, 46, 27,
				19, 33, 42, 41, 27,
				16, 27, 33, 29, 18
		}, 1e-5);
	}

	@Test
	public void test2DReconstruction() {
		final int WIDTH = 5;
		final int HEIGHT = 3;
		final int OPERATOR_ROOT_SIZE = 3;
		// double[][] is column-major
		Matrix prediction = Matrix.zeros(HEIGHT, WIDTH);
		prediction.set(1, 1, 1.0);
		prediction.set(1, 2, 1.0);
		prediction.set(1, 3, 1.0);
		prediction.reshape_i(1, WIDTH*HEIGHT);
		Matrix reconstruction = Matrix.zeros(HEIGHT, WIDTH);
		FunctionNetwork oneToOne = new FunctionNetwork(OPERATOR_ROOT_SIZE*OPERATOR_ROOT_SIZE, 1); // 3x3 -> 1x1
		oneToOne.predictionFunction = (Matrix k) -> new Matrix(1, 1, k.sum());
		oneToOne.reconstructionFunction = (Matrix k) -> new Matrix(OPERATOR_ROOT_SIZE, OPERATOR_ROOT_SIZE, k.sum()); // Take whatever is in the center and assign it to all values in area.
		// 0 0 0 0 0     1 2 3 2 1
		// 0 1 1 1 0 ->  1 2 3 2 1
		// 0 0 0 0 0     1 2 3 2 1

		ConvolutionalNetwork cn = new ConvolutionalNetwork(oneToOne,
				WIDTH, HEIGHT,
				OPERATOR_ROOT_SIZE, OPERATOR_ROOT_SIZE,
				1, 1,
				1, 1,
				ConvolutionalNetwork.EdgeBehavior.ZEROS);

		Matrix result = cn.reconstruct(prediction);

		System.out.println(result);
	}
}
