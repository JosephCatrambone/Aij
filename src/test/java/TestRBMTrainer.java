import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.networks.RestrictedBoltzmannMachine;
import com.josephcatrambone.aij.trainers.RBMTrainer;
import org.junit.Test;

import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Jo on 8/15/2015.
 */
public class TestRBMTrainer {
	private static final Logger LOGGER = Logger.getLogger(TestRBMTrainer.class.getName());

	private static boolean passesXOR(Matrix predictions) {
		return
		((predictions.get(0, 0) < 0.5) && (predictions.get(0, 1) < 0.5) && (predictions.get(0, 2) < 0.5)) &&
		((predictions.get(1, 0) < 0.5) && (predictions.get(1, 1) > 0.5) && (predictions.get(1, 2) > 0.5)) &&
		((predictions.get(2, 0) > 0.5) && (predictions.get(2, 1) < 0.5) && (predictions.get(2, 2) > 0.5)) &&
		((predictions.get(3, 0) > 0.5) && (predictions.get(3, 1) > 0.5) && (predictions.get(3, 2) < 0.5));
	}

	@Test
	public void testXOR() {
		//LOGGER.log(Level.INFO, "Building RBM + training data.");

		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(3, 5);
		Matrix x = new Matrix(4, 3, 0.0);
		x.setRow(0, new double[]{0,0,0});
		x.setRow(1, new double[]{0,1,1});
		x.setRow(2, new double[]{1,0,1});
		x.setRow(3, new double[]{1,1,0});

		RBMTrainer trainer = new RBMTrainer();
		trainer.learningRate = 0.1;
		trainer.batchSize = 1;
		trainer.notificationIncrement = 0;
		trainer.maxIterations = 20000;
		//trainer.earlyStopError = 0.01;

		Runnable updateFunction = new Runnable() {
			int i=0;
			@Override
			public void run() {
				System.out.println("Iteration " + i + ":" + trainer.lastError);
				i++;
			}
		};

		// Run
		trainer.train(rbm, x, null, updateFunction);

		// Test XOR
		Matrix predictions = new Matrix(4, 3);
		predictions.setRow(0, new double[]{0, 0, 0});
		predictions.setRow(1, new double[]{0, 1, 0});
		predictions.setRow(2, new double[]{1, 0, 0});
		predictions.setRow(3, new double[]{1, 1, 0});
		predictions = rbm.reconstruct(rbm.predict(predictions));

		System.out.println(predictions);

		System.out.println(rbm.daydream(1, 10));

		//assertEquals(predictions.get(0, 1), 15);
	}

	@Test
	public void testPattern() {
		//LOGGER.log(Level.INFO, "Building RBM + training data.");

		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(6, 4);
		Matrix x = new Matrix(6, 6, 0.0);
		x.setRow(0, new double[]{1,1,1,0,0,0});
		x.setRow(1, new double[]{1,0,1,0,0,0});
		x.setRow(2, new double[]{1,1,1,0,0,0});
		x.setRow(3, new double[]{0,0,1,1,1,0});
		x.setRow(4, new double[]{0,0,1,0,1,0});
		x.setRow(5, new double[]{0,0,1,1,1,0});

		RBMTrainer trainer = new RBMTrainer();
		trainer.learningRate = 0.1;
		trainer.batchSize = 1;
		trainer.notificationIncrement = 10000;
		trainer.maxIterations = 250000;
		//trainer.earlyStopError = 0.0000001;

		Runnable updateFunction = new Runnable() {
			int i=0;
			@Override
			public void run() {
				System.out.println("Iteration " + i + ":" + trainer.lastError);
				i++;
			}
		};

		// Run
		trainer.train(rbm, x, null, updateFunction);

		System.out.println(rbm.daydream(1, 1));
		System.out.println(rbm.daydream(1, 2));
		System.out.println(rbm.daydream(1, 4));
		System.out.println(rbm.daydream(1, 10));
	}
}
