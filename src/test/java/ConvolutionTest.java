import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.networks.ConvolutionalNetwork;
import com.josephcatrambone.aij.networks.Network;
import com.josephcatrambone.aij.networks.OneToOneNetwork;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Created by Jo on 5/28/2015.
 */
public class ConvolutionTest {
	@Test
	public void test2d() {
		// 1 2 3 4 5
		// 6 7 8 9 0
		// 1 2 3 4 5
		// 6 7 8 9 0
		// 1 2 3 4 5
		Matrix m = new Matrix(new double[][] {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}});
		OneToOneNetwork oneToOne = new OneToOneNetwork(9); // 3x3
		ConvolutionalNetwork cn = new ConvolutionalNetwork(oneToOne, 5, 5, 3, 3, 3, 3, 1, 1, ConvolutionalNetwork.EdgeBehavior.ZEROS);

		oneToOne.predictionMonitor = new OneToOneNetwork.Monitor() {
			int i=0;
			@Override
			public void run(Matrix intermediate) {
				switch(i) {
					case 0:
						// 0 0 0
						// 0 1 2
						// 0 6 7
						assertArrayEquals(intermediate.getRowArray(0), new double[] {0, 0, 0, 0, 1, 2, 0, 6, 7}, 1e-5);
						break;
					case 1:
						// 0 0 0
						// 1 2 3
						// 6 7 8
						assertArrayEquals(intermediate.getRowArray(0), new double[] {0, 0, 0, 1, 2, 3, 6, 7, 8}, 1e-5);
						break;
					case 2:
						// 0 0 0
						// 2 3 4
						// 7 8 9
						assertArrayEquals(intermediate.getRowArray(0), new double[] {0, 0, 0, 2, 3, 4, 7, 8, 9}, 1e-5);
						break;
					case 3:
						// 0 0 0
						// 3 4 5
						// 8 9 10
						assertArrayEquals(intermediate.getRowArray(0), new double[] {0, 0, 0, 3, 4, 5, 8, 9, 10}, 1e-5);
						break;
					case 4:
						// 0 0 0
						// 4 5 0
						// 9 10 0
						assertArrayEquals(intermediate.getRowArray(0), new double[] {0, 0, 0, 4, 5, 0, 9, 10, 0}, 1e-5);
						break;
					case 5:
						// 0 1 2
						// 0 6 7
						// 0 11 12
						assertArrayEquals(intermediate.getRowArray(0), new double[] {0, 1, 2, 0, 6, 7, 0, 11, 12}, 1e-5);
						break;
				}
				i++;
			}
		};
	}

}
