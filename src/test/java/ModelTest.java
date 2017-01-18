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
	public void testGrad() {
		Model m = new Model(1, 2);
		m.addDenseLayer(5, Model.Activation.TANH);
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

		for(int i=0; i < 100000; i++) {
			m.fit(x, y, 0.1f, Model.Loss.ABS);
			System.out.println(m.predict(x[0])[0]);
			System.out.println(m.predict(x[1])[0]);
			System.out.println(m.predict(x[2])[0]);
			System.out.println(m.predict(x[3])[0]);
			System.out.println();
		}
	}
}