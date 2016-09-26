import com.josephcatrambone.aij.CPUGraph;
import com.josephcatrambone.aij.Dimension;
import com.josephcatrambone.aij.Graph;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.function.UnaryOperator;

/**
 * Created by jcatrambone on 9/13/16.
 */
public class GraphForwardModeTest {
	public Graph makeGraph() {
		return new CPUGraph(); // For fast switching to GPU tests.
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

	@Test
	public void testAddGrad() {
		testOpGradient(Graph.NODE_OPERATION.ADD, 0.1f, -10.0f, 0.001f);
	}

	@Test
	public void testAdd() {
		Graph g = makeGraph();
		int x = g.addInput("x", new Dimension(1, 1));
		int y = g.addInput("y", new Dimension(1, 1));
		int a = g.addNode("a", Graph.NODE_OPERATION.ADD, new int[]{x, y});
		HashMap<Integer, float[]> inputs = new HashMap<>();
		inputs.put(x, new float[]{1});
		inputs.put(y, new float[]{2});
		org.junit.Assert.assertArrayEquals(new float[]{3}, g.getOutput(inputs, a), 0.0001f);
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
		org.junit.Assert.assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, g.getOutput(inputs, a), 0.0001f);

		// Clear and start over.
		g = makeGraph();
		x = g.addNode("x", Graph.NODE_OPERATION.INPUT, null, new Dimension(2, 2));
		y = g.addNode("y", Graph.NODE_OPERATION.INPUT, null, new Dimension(3, 2));
		a = g.addNode("a", Graph.NODE_OPERATION.MATRIXMULTIPLY, new int[]{x, y}, new Dimension(3, 2));
		inputs = new HashMap<>();

		inputs.put(x, new float[]{
				1, 0,
				0, 1
		});
		inputs.put(y, new float[]{
				1, 2, 3,
				4, 5, 6,
		});
		org.junit.Assert.assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6}, g.getOutput(inputs, a), 0.0001f);
	}

	@Test
	public void testForwardMethods() {
		Graph g = makeGraph();
		int x = g.addNode("x", Graph.NODE_OPERATION.INPUT, null, new Dimension(3, 3));

		//org.junit.Assert.assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6}, g.getOutput(inputs, a), 0.0001f);
	}
}
