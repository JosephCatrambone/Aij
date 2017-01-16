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

	@Test
	public void testMLP() {
		Graph g = new Graph();
		InputNode x = new InputNode(1, 2);
		VariableNode weight_ih = new VariableNode(2, 3);
		VariableNode bias_h = new VariableNode(1, 3);
		Node hidden = new TanhNode(new AddNode(new MatrixMultiplyNode(x, weight_ih), bias_h));
		VariableNode weight_ho = new VariableNode(3, 1);
		VariableNode bias_o = new VariableNode(1, 1);
		Node out = new AddNode(new MatrixMultiplyNode(hidden, weight_ho), bias_o);
		InputNode y = new InputNode(1, 1); // Target

		Node error = new SubtractNode(y, out);
		Node loss = new AbsNode(error);

		g.addNode(loss);

		// Test XOR.
		Random random = new Random();
		weight_ih.setVariable(new Matrix(2, 3, (i,j) -> random.nextFloat()));
		weight_ho.setVariable(new Matrix(3, 1, (i,j) -> random.nextFloat()));

		// Do a few iterations.
		final float LEARNING_RATE = 0.1f;
		HashMap<Node, Matrix> inputFeed = new HashMap<>();
		for(int i=0; i < 20000; i++) {
			float a = random.nextFloat();
			float b = random.nextFloat();
			inputFeed.put(x, new Matrix(1, 2, new float[]{a, b}));
			inputFeed.put(y, new Matrix(1, 1, new float[]{(a > 0.5) ^ (b > 0.5) ? 1.0f : 0.0f}));

			// Minimize loss wrt error:
			Matrix[] grad = g.getGradient(inputFeed, null, loss);
			weight_ih.setVariable(weight_ih.getVariable().elementOp(grad[weight_ih.id], (w, dw) -> w - LEARNING_RATE*dw));
			weight_ho.setVariable(weight_ho.getVariable().elementOp(grad[weight_ho.id], (w, dw) -> w - LEARNING_RATE*dw));
			bias_h.setVariable(bias_h.getVariable().elementOp(grad[bias_h.id], (w, dw) -> w - LEARNING_RATE*dw));
			bias_o.setVariable(bias_o.getVariable().elementOp(grad[bias_o.id], (w, dw) -> w - LEARNING_RATE*dw));
		}

		HashMap<Node, float[]> fd = new HashMap<>();
		fd.put(y, new float[]{0.0f});
		fd.put(x, new float[]{0.0f, 0.0f});
		float[] res = g.getOutput(fd, out);
		org.junit.Assert.assertEquals(0.0f, res[0], 0.2f);

		fd.put(x, new float[]{0.0f, 1.0f});
		res = g.getOutput(fd, out);
		org.junit.Assert.assertEquals(1.0f, res[0], 0.2f);

		fd.put(x, new float[]{1.0f, 0.0f});
		res = g.getOutput(fd, out);
		org.junit.Assert.assertEquals(1.0f, res[0], 0.2f);

		fd.put(x, new float[]{1.0f, 1.0f});
		res = g.getOutput(fd, out);
		org.junit.Assert.assertEquals(0.0f, res[0], 0.2f);

		// Try with resource.
		
		System.out.println(g.serializeToString());
	}
}
