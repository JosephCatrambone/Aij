import com.josephcatrambone.aij.*;
import com.josephcatrambone.aij.nodes.*;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

/**
 * Created by jcatrambone on 9/13/16.
 */
public class GraphForwardModeTest {
	final float TOLERANCE = 0.00001f;
	public void testOpGradient(Node op, float xStep, float xRange, float threshold) {
		Graph g = new Graph();
		Node x = new InputNode(3, 1);
		Node y = new InputNode(3, 1);
		g.addNode(x);
		g.addNode(y);

		HashMap<Node, Matrix> inputs = new HashMap<>();
		inputs.put(x, new Matrix(1, 3, new float[]{0.0f, 0.1f, 0.2f}));
		inputs.put(y, new Matrix(1, 3, new float[]{-1.0f, 0.0f, 1.0f}));
		//for(Graph.NODE_OPERATION op : Graph.NODE_OPERATION.values()) {
		//int out = g.addNode("res", op, new int[]{x, y}); // y will be ignored for some operations.
		op.inputs = new Node[]{x, y}; // Bit of a hack.
		g.addNode(op);
		float xIn = -xRange;
		while(xIn < xRange) {
			inputs.replace(x, new Matrix(1, 3, new float[]{xIn-xStep, xIn, xIn+xStep})); // Two steps.
			Matrix[] fwd = g.forward(inputs);
			float[] res = fwd[op.id].data;
			float[] grads = g.getGradient(inputs, fwd, op)[x.id].data;
			float approxGrad = (res[2]-res[0])/(2.0f*xStep);
			float gradientDiff = Math.abs(approxGrad - grads[1]); // / Math.abs(Math.max(approxGrad, grads[1]) + 1e-6f);
			// Gradient magnitude = abs(p - q) / max(p, q).
			System.out.println(op + ": At: " + xIn + " Numerical gradient: " + approxGrad + "  Returned gradient: " + grads[1] + "  Order: " + gradientDiff + " Res: " + Arrays.toString(res));
			org.junit.Assert.assertTrue("Numerical gradient: " + approxGrad + ".  Returned gradient: " + grads[1], gradientDiff < threshold);
			xIn += xStep;
		}
	}

	@Test
	public void testMatMulIdentity() {
		Graph g = new Graph();
		Node x = g.addNode(new InputNode(3, 3));
		Node a = g.addNode(new InputNode(3, 3));
		Node out = g.addNode(new MatrixMultiplyNode(x, a));

		HashMap<Node, float[]> inputFeed = new HashMap<>();
		inputFeed.put(a, new float[]{1, 0, 0, 0, 1, 0, 0, 0, 1});
		inputFeed.put(x, new float[]{1, 1, 1, 1, 1, 1, 1, 1, 1});

		org.junit.Assert.assertArrayEquals(new float[]{1, 1, 1, 1, 1, 1, 1, 1, 1}, g.getOutput(inputFeed, out), TOLERANCE);
	}
}

