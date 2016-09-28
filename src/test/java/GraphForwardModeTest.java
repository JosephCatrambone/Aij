import com.josephcatrambone.aij.*;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

/**
 * Created by jcatrambone on 9/13/16.
 */
public class GraphForwardModeTest {
	final float TOLERANCE = 0.00001f;
	public Graph makeGraph() {
		return new GPUGraph(); // For fast switching to GPU tests.
	}

	public void testOpGradient(Graph.NODE_OPERATION op, float xStep, float xRange, float threshold) {
		Graph g = makeGraph();
		int x = g.addInput("x", new Dimension(3, 1));
		int y = g.addInput("y", new Dimension(3, 1));

		HashMap<Integer, float[]> inputs = new HashMap<>();
		inputs.put(x, new float[]{0.0f, 0.1f, 0.2f});
		inputs.put(y, new float[]{-1.0f, 0.0f, 1.0f});
		//for(Graph.NODE_OPERATION op : Graph.NODE_OPERATION.values()) {
		int out = g.addNode("res", op, new int[]{x, y}); // y will be ignored for some operations.
		float xIn = -xRange;
		while(xIn < xRange) {
			inputs.replace(x, new float[]{xIn-xStep, xIn, xIn+xStep}); // Two steps.
			float[] res = g.getOutput(inputs, out);
			float[] grads = g.getGradient(inputs, out)[x];
			float approxGrad = (res[2]-res[0])/(2.0f*xStep);
			float gradientDiff = Math.abs(approxGrad - grads[1]); // / Math.abs(Math.max(approxGrad, grads[1]) + 1e-6f);
			// Gradient magnitude = abs(p - q) / max(p, q).
			System.out.println(op.name() + ": At: " + xIn + " Numerical gradient: " + approxGrad + "  Returned gradient: " + grads[1] + "  Order: " + gradientDiff + " Res: " + Arrays.toString(res));
			org.junit.Assert.assertTrue("Numerical gradient: " + approxGrad + ".  Returned gradient: " + grads[1], gradientDiff < threshold);
			xIn += xStep;
		}
	}

	public void testUnaryOp(Graph.NODE_OPERATION op, float in, float out) {
		Graph g = makeGraph();
		int x = g.addInput("x", new Dimension(1, 1));
		int a = g.addNode("a", op, new int[]{x});
		HashMap<Integer, float[]> inputs = new HashMap<>();
		inputs.put(x, new float[]{in});
		org.junit.Assert.assertArrayEquals(new float[]{out}, g.getOutput(inputs, a), TOLERANCE);
	}

	public void testBinaryOp(Graph.NODE_OPERATION op, float inA, float inB, float out) {
		Graph g = makeGraph();
		int x = g.addInput("x", new Dimension(1, 1));
		int y = g.addInput("y", new Dimension(1, 1));
		int a = g.addNode("a", op, new int[]{x, y});
		HashMap<Integer, float[]> inputs = new HashMap<>();
		inputs.put(x, new float[]{inA});
		inputs.put(y, new float[]{inB});
		org.junit.Assert.assertArrayEquals(new float[]{out}, g.getOutput(inputs, a), TOLERANCE);
	}

	@Test
	public void testAddGrad() {
		testOpGradient(Graph.NODE_OPERATION.ADD, 0.1f, -10.0f, 0.001f);
	}

	@Test
	public void testOps() {
		testBinaryOp(Graph.NODE_OPERATION.ADD, 3, 6, 3+6);
		testUnaryOp(Graph.NODE_OPERATION.ABS, -3, 3);
		testUnaryOp(Graph.NODE_OPERATION.ABS, 3, 3);
		testUnaryOp(Graph.NODE_OPERATION.EXP, 0, 1);
		testUnaryOp(Graph.NODE_OPERATION.EXP, 1, (float)Math.exp(1));
		testUnaryOp(Graph.NODE_OPERATION.INVERT, 2.0f, 0.5f);
		testUnaryOp(Graph.NODE_OPERATION.LOG, 8, (float)Math.log(8.0f));
		testBinaryOp(Graph.NODE_OPERATION.MULTIPLY, 5, 6, 5*6);
		testUnaryOp(Graph.NODE_OPERATION.NEGATE, 10, -10f);
		testUnaryOp(Graph.NODE_OPERATION.NEGATE, -10f, 10f);
		testBinaryOp(Graph.NODE_OPERATION.POWER, 10, 2f, 100f);
		testUnaryOp(Graph.NODE_OPERATION.POWER2, 10, 100f);
		testBinaryOp(Graph.NODE_OPERATION.SUBTRACT, 10, 2f, 10f-2f);
		testUnaryOp(Graph.NODE_OPERATION.TANH, 4, (float)Math.tanh(4.0f));
	}

	@Test
	public void testRowAdd() {
		Graph g = makeGraph();
		int x = g.addNode("x", Graph.NODE_OPERATION.INPUT, null, new Dimension(3, 2));
		int y = g.addNode("y", Graph.NODE_OPERATION.INPUT, null, new Dimension(3, 2));
		int a = g.addNode("a", Graph.NODE_OPERATION.ADD, new int[]{x, y}, new Dimension(3, 2));
		HashMap<Integer, float[]> inputs = new HashMap<>();
		inputs.put(x, new float[]{
			1, 2, 3,
			4, 5, 6
		});
		inputs.put(y, new float[]{
			1, 0, 1,
			0, 1, 0
		});
		org.junit.Assert.assertArrayEquals(new float[]{2, 2, 4, 4, 6, 6}, g.getOutput(inputs, a), 0.0001f);
	}

	@Test
	public void testTranspose() {
		Graph g = makeGraph();
		int x = g.addNode("x", Graph.NODE_OPERATION.INPUT, null, new Dimension(3, 2));
		int xt = g.addNode("xt", Graph.NODE_OPERATION.TRANSPOSE, new int[]{x}, new Dimension(2, 3));
		HashMap<Integer, float[]> inputs = new HashMap<>();
		inputs.put(x, new float[]{
				1, 2, 3,
				0, 0, 0
		});
		org.junit.Assert.assertArrayEquals(new float[]{1, 0, 2, 0, 3, 0}, g.getOutput(inputs, xt), 0.0001f);
	}

	@Test
	public void testMatMulIdentity() {
		Graph g = makeGraph();
		int x = g.addNode("x", Graph.NODE_OPERATION.INPUT, null, new Dimension(3, 3));
		int y = g.addNode("y", Graph.NODE_OPERATION.INPUT, null, new Dimension(3, 3));
		int a = g.addNode("a", Graph.NODE_OPERATION.MATRIXMULTIPLY, new int[]{x, y}, new Dimension(3, 3));
		HashMap<Integer, float[]> inputs = new HashMap<>();
		inputs.put(x, new float[]{
				1, 0, 0,
				0, 1, 0,
				0, 0, 1
		});
		inputs.put(y, new float[]{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9
		});
		float[] out = g.getOutput(inputs, a);
		System.out.println("I * A = " + Arrays.toString(out));
		org.junit.Assert.assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, out, 0.0001f);

		// Clear and start over.
		g = makeGraph();
		x = g.addNode("x", Graph.NODE_OPERATION.INPUT, null, new Dimension(2, 2));
		y = g.addNode("y", Graph.NODE_OPERATION.INPUT, null, new Dimension(3, 2));
		a = g.addNode("a", Graph.NODE_OPERATION.MATRIXMULTIPLY, new int[]{x, y}, new Dimension(3, 2));
		inputs = new HashMap<>();

		inputs.put(x, new float[]{
				1, 2,
				0, 1
		});
		inputs.put(y, new float[]{
				1, 2, 3,
				4, 5, 6,
		});
		org.junit.Assert.assertArrayEquals(new float[]{9, 12, 15, 4, 5, 6}, g.getOutput(inputs, a), 0.0001f);
	}

	@Test
	public void bigMatrixMultiply() {
		final int size = 10000;
		Random random = new Random();
		Graph g = makeGraph();
		int x = g.addInput("x", new Dimension(size, size));
		int y = g.addInput("y", new Dimension(size, size));
		int z = g.addNode("mmul", Graph.NODE_OPERATION.MATRIXMULTIPLY, new int[]{x, y});
		HashMap<Integer, float[]> inputs = new HashMap<>();
		inputs.put(x, new float[size*size]);
		inputs.put(y, new float[size*size]);
		for(int i=0; i < size*size; i++) {
			inputs.get(x)[i] = (float)random.nextGaussian();
			inputs.get(y)[i] = (float)random.nextGaussian();
		}
		long startTime = System.currentTimeMillis();
		float[] res = g.getOutput(inputs, z);
		long endTime = System.currentTimeMillis();
		System.out.println("Matrix multiply with " + size + "^2 elements: " + (endTime - startTime) + " ms.");
		//org.junit.Assert.assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6}, g.getOutput(inputs, a), 0.0001f);
	}
}

