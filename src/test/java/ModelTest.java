import com.josephcatrambone.aij.Graph;
import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.Model;
import com.josephcatrambone.aij.nodes.*;
import org.junit.Test;

import java.io.*;
import java.util.HashMap;
import java.util.Random;

/**
 * Created by jcatrambone on 9/13/16.
 */
public class ModelTest {

	@Test
	public void testXOR() {
		Model m = new Model(1, 2);
		m.addDenseLayer(10, Model.Activation.TANH);
		m.addDenseLayer(1, Model.Activation.SIGMOID);

		float[][] x = new float[][] {
			{0, 0},
			{0, 1},
			{1, 0},
			{1, 1}
		};

		float[][] y = new float[][] {
				{0},
				{1},
				{1},
				{0}
		};

		for(int i=0; i < 5000; i++) {
			m.fit(x, y, 0.5f, Model.Loss.SQUARED);
			System.out.print(m.predict(x[0])[0] + "\t\t");
			System.out.print(m.predict(x[1])[0] + "\t\t");
			System.out.print(m.predict(x[2])[0] + "\t\t");
			System.out.print(m.predict(x[3])[0] + "\t\t");
			System.out.println();
		}
	}

	@Test
	public void testParity() {
		Model m = new Model(1, 3);
		m.addDenseLayer(10, Model.Activation.TANH);
		m.addDenseLayer(2, Model.Activation.SOFTMAX); // 0 = odd parity, 1 = even parity.

		float[][] x = new float[][] {
			{0, 0, 0},
			{0, 0, 1},
			{0, 1, 0},
			{0, 1, 1},
			{1, 0, 0},
			{1, 0, 1},
			{1, 1, 0},
			{1, 1, 1}
		};

		float[][] y = new float[][] {
				{0, 1},
				{1, 0},
				{1, 0},
				{0, 1},
				{1, 0},
				{0, 1},
				{0, 1},
				{1, 0}
		};

		for(int i=0; i < 1000; i++) {
			m.fit(x, y, 0.1f, Model.Loss.SQUARED);
			float loss = 0;
			float[][] predictions = m.predict(x);
			for(int j=0; j < predictions.length; j++) {
				loss += Math.abs(predictions[j][0] - y[j][0]);
				loss += Math.abs(predictions[j][1] - y[j][1]);
			}
			System.out.println(i + "\tParity loss: " + loss);
		}
	}
}
