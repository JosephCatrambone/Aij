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
			//System.out.println("Error: " + grad[out.id].data[0]);
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
	public void testConvShape() {
		InputNode x = new InputNode(60, 60);
		VariableNode k = new VariableNode(3, 3);
		Convolution2DNode c = new Convolution2DNode(x, k, 1, 0);
		VariableNode j = new VariableNode(3, 3);
		Deconvolution2DNode d = new Deconvolution2DNode(c, j, 1, 0);

		Graph g = new Graph();

		org.junit.Assert.assertArrayEquals(
			new int[]{k.rows, k.columns},
			new int[]{j.rows, j.columns}
		);
	}

	@Test
	public void testConvolution() {
		final int ITERATIONS = 1000;
		final float LEARNING_RATE = 0.1f;

		Random random = new Random();

		// We want to train a classifier to recognize circles and rectangles.
		Graph g = new Graph();
		InputNode img = new InputNode(64, 64);
		VariableNode conv1_weight = new VariableNode(5, 5);
		Node conv1 = new TanhNode(new Convolution2DNode(img, conv1_weight, 1, 0));
		VariableNode conv2_weight = new VariableNode(5, 5);
		Node conv2 = new TanhNode(new Convolution2DNode(conv1, conv2_weight, 3, 0));
		Node flatten = new ReshapeNode(conv2, 1, -1);
		VariableNode fc_weight = new VariableNode(flatten.columns, 2);
		Node fc = new MatrixMultiplyNode(flatten, fc_weight);
		Node prediction = new TanhNode(fc);

		InputNode truth = new InputNode(1, 2);
		Node difference = new SubtractNode(prediction, truth);
		Node error = new AbsNode(difference);
		g.addNode(error);

		// Make our training data.
		HashMap <Node,Matrix> inputFeed = new HashMap<>();
		for(int i=0; i < 1000; i++) {
			Matrix canvas = new Matrix(64, 64);
			Matrix label = new Matrix(1, 2);
			if(random.nextBoolean()) { // Square or circle?
				// Square.
				int fromX = random.nextInt(64);
				int fromY = random.nextInt(64);
				int toX = random.nextInt(64);
				int toY = random.nextInt(64);
				// Fill in the rectangle.
				for(int r=Math.min(fromY, toY); r < Math.max(fromY, toY); r++) {
					for(int c=Math.min(fromX, toX); c < Math.max(fromX, toX); c++) {
						canvas.set(r, c, 1.0f);
					}
				}
				label.set(0, 0, 1.0f); // Square is 0,0.
			} else {
				// Circle.
				int radius = random.nextInt(8);
				int centerX = random.nextInt(64-8);
				int centerY = random.nextInt(64-8);
				for(int r=0; r < 64; r++) {
					for(int c=0; c < 64; c++) {
						int dx = c-centerX;
						int dy = r-centerY;
						if(dx*dx + dy*dy < radius*radius) {
							canvas.set(r, c, 1.0f);
						}
					}
				}
				label.set(0, 1, 1.0f); // Circle.
			}

			// Add our data.
			inputFeed.put(img, canvas);
			inputFeed.put(truth, label);

			// Every 100, we test instead of train.
			if(false) { //i % 100 == 0) {
				Matrix[] res = g.forward(inputFeed);
				System.out.println("Predicted: " + res[prediction.id]);
				System.out.println("Correct: " + label);
			} else {
				// Train one iteration.
				Matrix[] grads = g.getGradient(inputFeed, null, error);

				// Apply the gradients to all the variables.
				fc_weight.setVariable(fc_weight.getVariable().elementOp(grads[fc_weight.id], (a, b) -> a - (LEARNING_RATE * b)));
				conv2_weight.setVariable(conv2_weight.getVariable().elementOp(grads[conv2_weight.id], (a, b) -> a - (LEARNING_RATE * b)));
				conv1_weight.setVariable(conv1_weight.getVariable().elementOp(grads[conv1_weight.id], (a, b) -> a - (LEARNING_RATE * b)));
			}
		}
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
		Node out = new ReLUNode(new AddNode(new MatrixMultiplyNode(hidden, weight_ho), bias_o));
		InputNode y = new InputNode(1, 1); // Target

		Node error = new SubtractNode(y, out);
		Node loss = new PowerNode(error, 2);

		g.addNode(loss);

		// Test XOR.
		Random random = new Random();
		weight_ih.setVariable(new Matrix(2, 3, (i,j) -> random.nextFloat()));
		weight_ho.setVariable(new Matrix(3, 1, (i,j) -> random.nextFloat()));

		// Do a few iterations.
		final float LEARNING_RATE = 0.01f;
		HashMap<Node, Matrix> inputFeed = new HashMap<>();
		for(int i=0; i < 2000; i++) {
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

			System.out.println("Grad:" + grad[weight_ho.id]);
		}

		HashMap<Node, float[]> fd = new HashMap<>();
		//fd.put(y, new float[]{0.0f});
		fd.put(x, new float[]{0.0f, 0.0f});
		float[] res = g.getOutput(fd, out);
		//org.junit.Assert.assertEquals(0.0f, res[0], 0.2f);
		//org.junit.Assert.assertTrue(res[0] < 0.5f);
		System.out.println(res[0]);

		fd.put(x, new float[]{0.0f, 1.0f});
		res = g.getOutput(fd, out);
		System.out.println(res[0]);
		//org.junit.Assert.assertTrue(res[0] > 0.5f);

		fd.put(x, new float[]{1.0f, 0.0f});
		res = g.getOutput(fd, out);
		System.out.println(res[0]);
		//org.junit.Assert.assertTrue(res[0] > 0.5f);

		fd.put(x, new float[]{1.0f, 1.0f});
		res = g.getOutput(fd, out);
		System.out.println(res[0]);
		org.junit.Assert.assertTrue(res[0] < 0.5f);

		// Try with resource.
		
		System.out.println(g.serializeToString());
		try(BufferedWriter fout = new BufferedWriter(new FileWriter("xor_test.model"))) {
			fout.write(g.serializeToString());
		} catch(IOException ioe) {

		}

		try(BufferedReader fin = new BufferedReader(new FileReader("xor_test.model"))) {
			g.restoreFromString(fin.lines().reduce("", (a, b) -> a+"\n"+b));
		} catch(FileNotFoundException fnfe) {

		} catch(IOException ioe) {

		}
	}
}
