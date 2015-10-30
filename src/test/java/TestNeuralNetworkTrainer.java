import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.networks.NeuralNetwork;
import com.josephcatrambone.aij.trainers.BackpropTrainer;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by Jo on 8/15/2015.
 */
public class TestNeuralNetworkTrainer {

	@Test
	public void testXOR() {
		NeuralNetwork nn = null;

		Matrix x = new Matrix(4, 2, 0.0);
		x.setRow(0, new double[]{0, 0});
		x.setRow(1, new double[]{0, 1});
		x.setRow(2, new double[]{1, 0});
		x.setRow(3, new double[]{1, 1});

		Matrix y = new Matrix(4, 1, 0.0);
		y.set(0, 0, 0.0);
		y.set(1, 0, 1.0);
		y.set(2, 0, 1.0);
		y.set(3, 0, 0.0);

		BackpropTrainer trainer = new BackpropTrainer();
		trainer.momentum = 0.0;
		trainer.learningRate = 0.1;
		trainer.batchSize = 10;
		trainer.maxIterations = 10000;
		trainer.earlyStopError = 0.0;

		// Test XOR
		// Run
		nn = new NeuralNetwork(new int[]{2, 3, 1}, new String[]{"tanh", "tanh", "tanh"});
		trainer.train(nn, x, y, null);

		Matrix predictions = nn.predict(x);
		assertTrue(predictions.get(0, 0) < 0.5);
		assertTrue(predictions.get(1, 0) > 0.5);
		assertTrue(predictions.get(2, 0) > 0.5);
		assertTrue(predictions.get(3, 0) < 0.5);

		nn = new NeuralNetwork(new int[]{2, 3, 1}, new String[]{"logistic", "logistic", "logistic"});
		trainer.train(nn, x, y, null);
		assertTrue(predictions.get(0, 0) < 0.5);
		assertTrue(predictions.get(1, 0) > 0.5);
		assertTrue(predictions.get(2, 0) > 0.5);
		assertTrue(predictions.get(3, 0) < 0.5);

		nn = new NeuralNetwork(new int[]{2, 3, 1}, new String[]{"linear", "logistic", "linear"});
		trainer.train(nn, x, y, null);
		assertTrue(predictions.get(0, 0) < 0.5);
		assertTrue(predictions.get(1, 0) > 0.5);
		assertTrue(predictions.get(2, 0) > 0.5);
		assertTrue(predictions.get(3, 0) < 0.5);
	}
}
