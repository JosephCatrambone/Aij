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

	@Test
	public void testPattern() {
		LOGGER.log(Level.INFO, "Building RBM + training data: Test Pattern 0");
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(6, 3);

		Matrix x = new Matrix(6, 6, 0.0);
		x.setRow(0, new double[]{1,1,1,0,0,0});
		x.setRow(1, new double[]{1,0,1,0,0,0});
		x.setRow(2, new double[]{1,1,1,0,0,0});
		x.setRow(3, new double[]{0,0,1,1,1,0});
		x.setRow(4, new double[]{0,0,1,1,0,0});
		x.setRow(5, new double[]{0,0,1,1,1,0});

		RBMTrainer trainer = new RBMTrainer();
		trainer.learningRate = 0.1;
		trainer.batchSize = 6;
		trainer.notificationIncrement = 0;
		trainer.maxIterations = 25000;
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

		Matrix input = new Matrix(new double[][]{{0, 0, 0, 1, 1, 0}, {0, 0, 1, 1, 0, 0}});
		Matrix output = rbm.predict(input);
		//System.out.println(rbm.reconstruct(output));
	}
}