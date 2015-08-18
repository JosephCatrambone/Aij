import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.networks.NeuralNetwork;
import com.josephcatrambone.aij.trainers.BackpropTrainer;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by Jo on 8/15/2015.
 */
public class TestNeuralNetworkTrainer {

	private static boolean passesXOR(Matrix predictions) {
		return
		((predictions.get(0, 0) < 0.5) && (predictions.get(0, 1) < 0.5) && (predictions.get(0, 2) < 0.5)) &&
		((predictions.get(1, 0) < 0.5) && (predictions.get(1, 1) > 0.5) && (predictions.get(1, 2) > 0.5)) &&
		((predictions.get(2, 0) > 0.5) && (predictions.get(2, 1) < 0.5) && (predictions.get(2, 2) > 0.5)) &&
		((predictions.get(3, 0) > 0.5) && (predictions.get(3, 1) > 0.5) && (predictions.get(3, 2) < 0.5));
	}

	@Test
	public void testXOR() {

		NeuralNetwork nn = new NeuralNetwork(new int[]{2, 3, 1}, new String[]{"linear", "tanh", "tanh"});
		Matrix x = new Matrix(4, 2, 0.0);
		x.setRow(0, new double[]{0,0});
		x.setRow(1, new double[]{0,1});
		x.setRow(2, new double[]{1,0});
		x.setRow(3, new double[]{1,1});

		Matrix y = new Matrix(4, 1, 0.0);
		y.set(0, 0, 0.0);
		y.set(1, 0, 1.0);
		y.set(2, 0, 1.0);
		y.set(3, 0, 0.0);

		BackpropTrainer trainer = new BackpropTrainer();
		trainer.momentum = 0.7;
		trainer.learningRate = 0.3;
		trainer.batchSize = 1;
		trainer.maxIterations = 2000000000;
		trainer.earlyStopError = 0.0001;

		// Run
		trainer.train(nn, x, y, null);

		// Test XOR
		Matrix predictions = new Matrix(4, 2);
		predictions.setRow(0, new double[]{0, 0});
		predictions.setRow(1, new double[]{0, 1});
		predictions.setRow(2, new double[]{1, 0});
		predictions.setRow(3, new double[]{1, 1});
		predictions = nn.predict(predictions);

		System.out.println(predictions);

		/*
		assertTrue(predictions.get(0, 0) < 0.5);
		assertTrue(predictions.get(1, 0) > 0.5);
		assertTrue(predictions.get(2, 0) > 0.5);
		assertTrue(predictions.get(3, 0) < 0.5);
		*/

		//assertEquals(predictions.get(0, 1), 15);
	}
}
