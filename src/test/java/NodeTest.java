import com.josephcatrambone.aij.*;

import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.nodes.*;
import com.josephcatrambone.aij.optimizers.Optimizer;
import com.josephcatrambone.aij.optimizers.SGD;
import org.junit.Test;
import org.junit.Assert;

import java.io.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Created by jcatrambone on 9/13/16.
 */
public class NodeTest {

	public void testGradient(Node n, double[] domain, double dx, double threshold) {
		n.id = 0; // Hack in case we index by this.
		// Do a forward pass on the node with a matrix of three rows.  
		// In a verticle line we've got DX.
		// forward: Matrix[] args -> Matrix.
		// Assumes single operator to the matrix.
		Matrix arg = new Matrix(3, domain.length);
		for(int i=0; i < domain.length; i++) {
			arg.set(0, i, domain[i]-dx);
			arg.set(1, i, domain[i]);
			arg.set(2, i, domain[i]+dx);
		}
		Matrix fwd = n.forward(new Matrix[]{arg});

		// f'(x) = f(x+dx) - f(x-dx) / (2*dx)
		double[] numericalGradient = new double[domain.length];
		for(int i=0; i < domain.length; i++) {
			numericalGradient[i] = (fwd.get(2,i) - fwd.get(0, i)) / (2.0f*dx);
		}

		// Calculate the exact gradient.
		Matrix grad = n.reverse(new Matrix[]{arg}, Matrix.ones(fwd.rows, fwd.columns))[0];
		double[] calculatedGradient = grad.getSlice(1, 2, 0, 10).data;

		// Dump output.
		System.out.println(n.getClass().getName() +" gradient:");
		System.out.println(" - Numerical gradient: " + Arrays.toString(numericalGradient));
		System.out.println(" - Calculated gradient: " + Arrays.toString(calculatedGradient));

		// Gradient error magnitude.
		for(int i=0; i < domain.length; i++) {
			if(calculatedGradient[i] == 0 && numericalGradient[i] == 0) {
				continue; // If there's no grad, we're okay.
			}
			double p = calculatedGradient[i];
			double q = numericalGradient[i];
			org.junit.Assert.assertTrue("Gradient order less than threshold?  " + p + " vs " + q, Math.abs(p-q)/Math.max(p,q) < threshold);
		}

		// Calculate and check the gradient order.
		//org.junit.Assert.assertArrayEquals(numericalGradient, calculatedGradient, threshold);
	}

	@Test
	public void testGrad() {
		InputNode x = new InputNode(1, 10);
		double[] values = new double[]{-10, -5, -2, -1, -0.1f, 0.1f, 1, 2, 5, 10};

		testGradient(new TanhNode(x), values, 0.01f, 0.1f);
		testGradient(new SigmoidNode(x), values, 0.01f, 0.5f);
		testGradient(new ReLUNode(x), values, 0.01f, 0.2f);
		testGradient(new InverseNode(x), values, 0.01f, 0.2f);
		testGradient(new ExpNode(x), values, 0.01f, 0.2f);
		testGradient(new PowerNode(x, 2), values, 0.01f, 0.2f);
		testGradient(new NegateNode(x), values, 0.01f, 0.2f);

		// These require non-negative values.
		values = new double[]{0.1f, 1, 2, 5, 10, 100, 1000, 10000, 1000000, 1000000000};
		testGradient(new LogNode(x), values, 0.01f, 0.2f);

		// Has a discontinuity at zero.
		values = new double[]{-10, -5, -2, -1, -0.5f, 0.5f, 1, 2, 5, 10};
		testGradient(new AbsNode(x), values, 0.01f, 0.2f);
	}

	@Test
	public void testConvolution() {
		Node x = new InputNode(5, 5);
		VariableNode kernel = new VariableNode(3, 3);
		kernel.setVariable(new Matrix(3, 3, new double[]{
			1.0, 1.0, 1.0,
			1.0, 1.0, 1.0,
			1.0, 1.0, 1.0
		}));
		Node c = new Convolution2DNode(x, kernel, 2, 2);
		Graph g = new Graph();
		g.addNode(c);

		double[] m = new double[]{
			1, 2, 3, 4, 5,
			6, 7, 8, 9, 0,
			1, 2, 3, 4, 5,
			6, 7, 8, 9, 0,
			1, 2, 3, 4, 5
		};
		HashMap<Node, double[]> inputFeed = new HashMap<>();
		inputFeed.put(x, m);
		double[] res = g.getOutput(inputFeed, c);
		assert(res[0] == 1+2+6+7);
		assert(res[1] == 2+3+4+7+8+9);
		assert(res[2] == 4+5+9+0);
		assert(res[3] == 6+7+1+2+6+7);
		assert(res[4] == 7+8+9+2+3+4+7+8+9);
	}

	@Test
	public void testDeconvolution() {
		Random random = new Random();
		VariableNode convKernel = new VariableNode(new Matrix(3, 3, (i,j) -> 0.1*random.nextGaussian()));
		VariableNode deconvKernel = new VariableNode(new Matrix(3, 3, (i,j) -> 0.1*random.nextGaussian()));

		Node x = new InputNode(5, 5);
		Node conv = new Convolution2DNode(x, convKernel, 2, 2);
		Node act = new TanhNode(conv);
		Node deconv = new Deconvolution2DNode(act, deconvKernel, 2, 2);
		Node y = new InputNode(5, 5); // Target

		Node loss = new RowSumNode(new PowerNode(new SubtractNode(deconv, y), 2.0));

		Graph g = new Graph();
		g.addNode(loss);

		Optimizer sgd = new SGD(g, new VariableNode[]{convKernel, deconvKernel}, 0.1);

		Map<Node, Matrix> inputFeed = new HashMap<>();
		for(int i=0; i < 10000; i++) {
			inputFeed.put(x, new Matrix(5, 5, new double[]{
					0, 1, 0, 1, 0,
					1, 0, 1, 0, 1,
					0, 1, 0, 1, 0,
					1, 0, 1, 0, 1,
					0, 1, 0, 1, 0
			})); // Can we learn a checkerboard?
			inputFeed.put(y, inputFeed.get(x));
			sgd.minimize(loss, inputFeed);
		}

		Matrix resultantMatrix = g.forward(inputFeed)[deconv.id];
		double[] res = resultantMatrix.data; //g.getOutput(inputFeed, deconv);
		System.out.println(resultantMatrix);
		for(int i=0; i < 25; i++) {
			if(i%2 == 0) {
				assert (res[i] < 0.5);
			} else {
				assert(res[i] > 0.5);
			}
		}
	}

	@Test
	public void resizeTest() {
		Node input = new InputNode(2, 3);
		Node flatten = new ReshapeNode(input, 1, -1);
		assert(flatten.rows == 1 && flatten.columns == 2*3);

		Matrix m = new Matrix(2, 3, new double[]{1, 2, 3, 4, 5, 6});
		Matrix res = flatten.forward(new Matrix[]{m});
		System.out.println(m);
		System.out.println(res);
		for(int i=0; i < 6; i++) {
			Assert.assertEquals(res.data[i], m.data[i], 1e-3);
		}

		Matrix g = new Matrix(1, 2*3, new double[]{1, 2, 3, 4, 5, 6});
		Matrix back = flatten.reverse(new Matrix[]{res}, g)[0];
		System.out.println(g);
		System.out.println(back);
	}
}
