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

		NeuralNetwork nn = new NeuralNetwork(new int[]{2, 3, 1}, new String[]{"tanh", "tanh", "tanh"});
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

		// Run
		for(int i=0; i < 2; i++) {
			System.out.println("Weights " + i + ":" + nn.getWeights(i));
		}
		for(int i=0; i < 3; i++) {
			System.out.println("Biases " + i + ": " + nn.getBiases(i));
		}
		trainer.train(nn, x, y, null);
		for(int i=0; i < 2; i++) {
			System.out.println("Weights " + i + ":" + nn.getWeights(i));
		}
		for(int i=0; i < 3; i++) {
			System.out.println("Biases " + i + ": " + nn.getBiases(i));
		}

		// Test XOR
		Matrix predictions = new Matrix(4, 2);
		predictions.setRow(0, new double[]{0, 0});
		predictions.setRow(1, new double[]{0, 1});
		predictions.setRow(2, new double[]{1, 0});
		predictions.setRow(3, new double[]{1, 1});
		predictions = nn.predict(predictions);

		System.out.println(predictions);

		assertTrue(predictions.get(0, 0) < 0.5);
		assertTrue(predictions.get(1, 0) > 0.5);
		assertTrue(predictions.get(2, 0) > 0.5);
		assertTrue(predictions.get(3, 0) < 0.5);

		//assertEquals(predictions.get(0, 1), 15);
	}
}
