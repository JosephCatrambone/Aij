import com.josephcatrambone.aij.*;

import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.nodes.*;
import org.junit.Test;

import java.io.*;
import java.util.Arrays;
import java.util.HashMap;
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
		System.out.println("Numerical gradient: " + Arrays.toString(numericalGradient));
		System.out.println("Calculated gradient: " + Arrays.toString(calculatedGradient));

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

		testGradient(new TanhNode(x), values, 0.01f, 0.5f);
		testGradient(new SigmoidNode(x), values, 0.01f, 1.0f);
		testGradient(new ReLUNode(x), values, 0.01f, 0.2f);

	}

}
