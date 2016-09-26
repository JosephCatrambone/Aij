import com.josephcatrambone.aij.CPUGraph;
import com.josephcatrambone.aij.Dimension;
import com.josephcatrambone.aij.Graph;

import org.junit.Test;

import java.io.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

/**
 * Created by jcatrambone on 9/13/16.
 */
public class GraphReverseModeTest {
	public Graph makeGraph() {
		return new CPUGraph(); // For fast switching to GPU tests.
	}

	@Test
	public void testAdd() {
		// a = x+y
		// da/dx = 1  da/dy = 1
		Graph g = makeGraph();
		int x = g.addInput("x", new Dimension(1, 1));
		int y = g.addInput("y", new Dimension(1, 1));
		int a = g.addNode("a", Graph.NODE_OPERATION.ADD, new int[]{x, y});
		HashMap<Integer, float[]> inputs = new HashMap<>();
		inputs.put(x, new float[]{10});
		inputs.put(y, new float[]{20});
		float[][] grads = g.getGradient(inputs, a);
		System.out.println("a = x+y.  da WRT [x, y]: " + grads[x][0] + ", " + grads[y][0]);
		org.junit.Assert.assertArrayEquals(
			new float[]{1},
			g.getGradient(inputs, a)[x],
			0.0001f
		);
	}

	@Test
	public void testMultiply() {
		// a = (x*y).  b = a + y.  b = (x*y) + y.
		// d/dx b = d/dx a + d/dx y = d/dx a = y
		// d/dy b = d/dy a + 1 = d/dy x + 1
		Graph g = makeGraph();
		int x = g.addNode("x", Graph.NODE_OPERATION.INPUT, new int[]{}, new Dimension(3, 1));
		int y = g.addNode("y", Graph.NODE_OPERATION.INPUT, new int[]{}, new Dimension(3, 1));
		int a = g.addNode("a", Graph.NODE_OPERATION.MULTIPLY, new int[]{x, y}, new Dimension(3, 1));
		int b = g.addNode("b", Graph.NODE_OPERATION.ADD, new int[]{a, y}, new Dimension(3, 1));
		HashMap<Integer, float[]> inputs = new HashMap<>();
		inputs.put(x, new float[]{10, 20, 30});
		inputs.put(y, new float[]{1, 0, -1});
		float[][] grads = g.getGradient(inputs, b);
		System.out.println("b = (x*y) + y.  da WRT [x, y] @ x=10, y=1: " + grads[x][0] + ", " + grads[y][0]);
		org.junit.Assert.assertArrayEquals(
				inputs.get(y),
				grads[x],
				0.0001f
		);
		org.junit.Assert.assertArrayEquals(
				new float[]{11, 21, 31}, // x + 1.
				grads[y],
				0.0001f
		);
	}

	@Test
	public void testMatrixMultiply() throws IOError, IOException {
		Random random = new Random();
		Graph g = makeGraph();
		int const_two = g.addInput("2", new Dimension(1, 1));
		int y = g.addInput("y", new Dimension(1, 1));
		int x = g.addInput("x", new Dimension(2, 1));

		int w1 = g.addNode("w1", Graph.NODE_OPERATION.INPUT, new int[]{}, new Dimension(3, 2));
		int w2 = g.addNode("w2", Graph.NODE_OPERATION.INPUT, new int[]{}, new Dimension(1, 3));
		int b1 = g.addInput("bias1", new Dimension(3, 1));

		int h_act = g.addNode("h_act", Graph.NODE_OPERATION.MATRIXMULTIPLY, new int[]{x, w1});
		int h_act_biased = g.addNode("h_act_biased", Graph.NODE_OPERATION.ADD, new int[]{h_act, b1});
		int h = g.addNode("h", Graph.NODE_OPERATION.TANH, new int[]{h_act_biased});
		int out = g.addNode("pre_out", Graph.NODE_OPERATION.MATRIXMULTIPLY, new int[]{h, w2});
		//int out = g.addNode("out", Graph.NODE_OPERATION.TANH, new int[]{pre_out});
		int inv_out = g.addNode("difference", Graph.NODE_OPERATION.NEGATE, new int[]{out});
		int diff = g.addNode("sum", Graph.NODE_OPERATION.ADD, new int[]{y, inv_out});
		int err = g.addNode("pow", Graph.NODE_OPERATION.POWER, new int[]{diff, const_two});

		HashMap<Integer, float[]> inputs = new HashMap<>();
		inputs.put(const_two, new float[]{2.0f});
		inputs.put(x, new float[]{});
		inputs.put(y, new float[]{});
		inputs.put(b1, new float[]{});
		inputs.put(w1, new float[]{});
		inputs.put(w2, new float[]{});

		float[] w1Data = new float[]{0.1f, 0.001f, -1.0f, 0.01f, 0.00001f, 1.0f};
		float[] w2Data = new float[]{0.1f, 0.001f, -0.1f};
		float[] bData = new float[]{0f, 0f, 0f};

		float learningRate = 1e-2f;
		float gradCap = 1.0f;
		for(int i=1; i < 10000; i++) {
			inputs.replace(w1, w1Data);
			inputs.replace(w2, w2Data);
			inputs.replace(b1, bData);

			float x0 = random.nextFloat() > 0.5f ? 1.0f : 0.0f;
			float x1 = random.nextFloat() > 0.5f ? 1.0f : 0.0f;
			float y0 = (x0 > 0.5f) ^ (x1 > 0.5f) ? 1.0f : 0.0f;
			inputs.replace(x, new float[]{x0, x1});
			inputs.replace(y, new float[]{y0});
			float[][] grads = g.getGradient(inputs, err);
			for(int j=0; j < w1Data.length; j++) {
				w1Data[j] -= Math.abs(learningRate*grads[w1][j]) < gradCap ? learningRate*grads[w1][j] : gradCap*Math.signum(grads[w1][j]);
			}
			for(int j=0; j < w2Data.length; j++) {
				w2Data[j] -= Math.abs(learningRate*grads[w2][j]) < gradCap ? learningRate*grads[w2][j] : gradCap*Math.signum(grads[w2][j]);
			}
			for(int j=0; j < bData.length; j++) {
				bData[j] -= Math.abs(learningRate*grads[b1][j]) < gradCap ? learningRate*grads[b1][j] : gradCap*Math.signum(grads[b1][j]);
			}

			// Test the error for validation.
			float errorAccumulator = 0.0f;
			HashMap<Integer, float[]> testInputs = new HashMap<>();
			testInputs.put(const_two, new float[]{2.0f});
			testInputs.put(x, new float[]{0f, 0f});
			testInputs.put(y, new float[]{0f});
			testInputs.put(w1, w1Data);
			testInputs.put(w2, w2Data);
			testInputs.put(b1, bData);
			errorAccumulator += g.getOutput(testInputs, err)[0];
			testInputs.replace(x, new float[]{0f, 1f});
			testInputs.replace(y, new float[]{1f});
			errorAccumulator += g.getOutput(testInputs, err)[0];
			testInputs.replace(x, new float[]{1f, 0f});
			testInputs.replace(y, new float[]{1f});
			errorAccumulator += g.getOutput(testInputs, err)[0];
			testInputs.replace(x, new float[]{1f, 1f});
			testInputs.replace(y, new float[]{0f});
			errorAccumulator += g.getOutput(testInputs, err)[0];
			if(i % 1000 == 0) {
				System.out.print(errorAccumulator + "\n");
			}
		}

		inputs.put(x, new float[]{0.0f, 0.0f});
		inputs.put(w1, w1Data);
		inputs.put(w2, w2Data);
		float[] outs = new float[4];
		float[] res = g.getOutput(inputs, out);
		outs[0] = res[0];

		inputs.put(x, new float[]{1.0f, 0.0f});
		res = g.getOutput(inputs, out);
		outs[1] = res[0];

		inputs.put(x, new float[]{0.0f, 1.0f});
		res = g.getOutput(inputs, out);
		outs[2] = res[0];

		inputs.put(x, new float[]{1.0f, 1.0f});
		res = g.getOutput(inputs, out);
		outs[3] = res[0];

		System.out.println("Outputs: " + Arrays.toString(outs));
		System.out.println("End weights: \nw1:" + Arrays.toString(w1Data) + "\nw2:" + Arrays.toString(w2Data) + "\nb:" + Arrays.toString(bData));
		org.junit.Assert.assertTrue(outs[0] < 0.1f);
		org.junit.Assert.assertTrue(outs[1] > 0.9f);
		org.junit.Assert.assertTrue(outs[2] > 0.9f);
		org.junit.Assert.assertTrue(outs[3] < 0.1f);

	}


}
