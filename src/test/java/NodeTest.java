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

	public void testGradient(Node n, float[] domain, float dx, float threshold) {
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
		float[] numericalGradient = new float[domain.length];
		for(int i=0; i < domain.length; i++) {
			numericalGradient[i] = (fwd.get(2,i) - fwd.get(0, i)) / (2.0f*dx);
		}

		// Calculate the exact gradient.
		Matrix grad = n.reverse(new Matrix[]{fwd}, Matrix.ones(fwd.rows, fwd.columns))[0];
		float[] calculatedGradient = grad.getSlice(1, 2, 0, -1).data;

		// Calculate and check the gradient order.
		org.junit.Assert.assertArrayEquals(numericalGradient, calculatedGradient, threshold);
	}

	@Test
	public void testGrad() {
		InputNode x = new InputNode(1, 10);
		TanhNode n = new TanhNode(x);
		float[] values = new float[]{-10, -5, -2, -1, 0.1f, 0.1f, 1, 2, 5, 10};

		testGradient(n, values, 0.1f, 0.01f);

	}

}