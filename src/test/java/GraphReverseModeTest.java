import com.josephcatrambone.aij.*;

import com.josephcatrambone.aij.nodes.*;
import org.junit.Test;

import java.io.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

/**
 * Created by jcatrambone on 9/13/16.
 */
public class GraphReverseModeTest {

	@Test
	public void testGrad() {
		Graph g = new Graph();
		InputNode x = new InputNode(1, 1);
		VariableNode m = new VariableNode(1, 1);
		VariableNode b = new VariableNode(1, 1);
		InputNode y = new InputNode(1, 1); // Target.

		Node out = new AddNode(b, new MultiplyNode(m, x));
		Node error = new SubtractNode(y, out);
		Node loss = new PowerNode(error, 2.0f);

		g.addNode(loss);

		// Try and approximate some linear function.
		Random random = new Random();
		float target_m = (float)random.nextGaussian()*100f;
		float target_b = (float)random.nextGaussian()*100f;
		m.setVariable(new Matrix(1, 1, new float[]{random.nextFloat()}));

		// Do a few iterations.
		final float LEARNING_RATE = 0.1f;
		HashMap<Node, Matrix> inputFeed = new HashMap<>();
		for(int i=0; i < 1000; i++) {
			float xData = random.nextFloat();
			inputFeed.put(x, new Matrix(1, 1, new float[]{xData}));
			inputFeed.put(y, new Matrix(1, 1, new float[]{xData*target_m + target_b}));
			// Minimize loss wrt error:
			Matrix[] grad = g.getGradient(inputFeed, null, loss);
			m.setVariable(m.getVariable().elementOp(d -> d-grad[m.id].data[0]*LEARNING_RATE));
			b.setVariable(b.getVariable().elementOp(d -> d-grad[b.id].data[0]*LEARNING_RATE));
			System.out.println("Error: " + grad[out.id].data[0]);
		}

		System.out.println(" Expected: y = " + target_m + " * x + " + target_b);
		System.out.println(" Got: y = " + m.getVariable().data[0] + " * x + " + b.getVariable().data[0]);

		org.junit.Assert.assertArrayEquals(
			new float[]{m.getVariable().data[0], b.getVariable().data[0]},
			new float[]{target_m, target_b},
			0.01f
		);
	}
}
