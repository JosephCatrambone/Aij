import com.josephcatrambone.aij.Model;
import org.junit.Test;

import java.io.*;
import java.util.Random;
import java.util.zip.GZIPInputStream;

/**
 * Created by jcatrambone on 9/13/16.
 */
public class ModelTest {

	@Test
	public void saveRestoreTest() {
		Model m = new Model(20, 20);
		m.addConvLayer(2, 2, 1, 1, Model.Activation.NONE);
		m.addConvLayer(2, 2, 1, 1, Model.Activation.NONE);
		m.addFlattenLayer();
		m.addDenseLayer(10, Model.Activation.NONE);
		m.addDenseLayer(11, Model.Activation.TANH);
		m.addDenseLayer(12, Model.Activation.RELU);

		String model = m.serializeToString();
		Model m2 = new Model(20, 20);
		m2.restoreFromString(model);
	}

	@Test
	public void testXORSerial() {
		testXOR(false);
	}

	@Test
	public void testXORParallel() {
		testXOR(true);
	}

	public void testXOR(boolean parallel) {
		Model m = new Model(1, 2);
		m.addDenseLayer(10, Model.Activation.TANH);
		m.addDenseLayer(1, Model.Activation.SIGMOID);

		double[][] x = new double[][] {
			{0, 0},
			{0, 1},
			{1, 0},
			{1, 1}
		};

		double[][] y = new double[][] {
				{0},
				{1},
				{1},
				{0}
		};

		for(int i=0; i < 5000; i++) {
			if(parallel) {
				m.fitBatch(x, y, 0.5, Model.Loss.SQUARED);
			} else {
				m.fit(x, y, 0.5f, Model.Loss.SQUARED);
			}
			System.out.print(m.predict(x[0])[0] + "\t\t");
			System.out.print(m.predict(x[1])[0] + "\t\t");
			System.out.print(m.predict(x[2])[0] + "\t\t");
			System.out.print(m.predict(x[3])[0] + "\t\t");
			System.out.println();
		}
	}

	@Test
	public void testCountSoftmax() {
		Model m = new Model(1, 3);
		// Are there one, two, or three bits set?
		m.addDenseLayer(5, Model.Activation.SOFTMAX);
		m.addDenseLayer(3, Model.Activation.SOFTMAX);

		double[][] x = new double[][] {
			{0, 0, 1},
			{0, 1, 0},
			{1, 0, 0},
			{0, 1, 1},
			{1, 0, 1},
			{1, 1, 0},
			{1, 1, 1}
		};

		double[][] y = new double[][] {
			{0, 0, 1},
			{0, 0, 1},
			{0, 0, 1},
			{0, 1, 0},
			{0, 1, 0},
			{0, 1, 0},
			{1, 0, 0}
		};

		for(int i=0; i < 1000; i++) {
			m.fit(x, y, 0.1f, Model.Loss.SQUARED);
			double loss = 0;
			double[][] predictions = m.predict(x);
			for(int j=0; j < predictions.length; j++) {
				loss += Math.abs(predictions[j][0] - y[j][0]);
				loss += Math.abs(predictions[j][1] - y[j][1]);
			}
			System.out.println(i + "\tParity loss: " + loss);
		}
	}

	@Test
	public void testMNIST() throws IOException {
		final int ITERATION_COUNT = 1000000;
		final int BATCH_SIZE = 10;
		final int REPORT_INTERVAL = 1000;
		Model model;
		double[][] images;
		double[][] labels; // One-hot.

		DataInputStream image_in = new DataInputStream(new GZIPInputStream(new FileInputStream("train-images-idx3-ubyte.gz")));
		DataInputStream labels_in = new DataInputStream(new GZIPInputStream(new FileInputStream("train-labels-idx1-ubyte.gz")));

		// Read the images first.
		int magicNumber = image_in.readInt();
		assert(magicNumber == 0x00000803); // 2051 for training images.  2049 for training labels.
		int imageCount = image_in.readInt();
		int rows = image_in.readInt();
		int columns = image_in.readInt();
		// Images are row-wise, which is great because so is our model.
		images = new double[imageCount][rows*columns];
		for(int imageNumber=0; imageNumber < imageCount; imageNumber++) {
			for(int r=0; r < rows; r++) {
				for(int c=0; c < columns; c++) {
					images[imageNumber][c + r*columns] = image_in.readUnsignedByte()/255.0f;
				}
			}
		}

		// Read the labels.
		magicNumber = labels_in.readInt();
		assert(magicNumber == 0x00000801);
		int labelCount = labels_in.readInt();
		labels = new double[labelCount][10];
		for(int labelNumber=0; labelNumber < labelCount; labelNumber++) {
			int label = labels_in.readUnsignedByte();
			labels[labelNumber][label] = 1.0f;
		}

		// Verify we've got all the data and labels.
		assert(labelCount == imageCount);

		// Build and train our model.
		model = new Model(rows, columns);
		//model.addConvLayer(3, 3, 2, 2, Model.Activation.TANH);
		//model.addConvLayer(3, 3, 2, 2, Model.Activation.TANH);
		//model.addConvLayer(3, 3, 2, 2, Model.Activation.TANH);
		model.addFlattenLayer();
		model.addDenseLayer(64, Model.Activation.TANH);
		model.addDenseLayer(128, Model.Activation.TANH);
		model.addDenseLayer(10, Model.Activation.SOFTMAX);

		// Split up the training data into target and test.
		// Start by shuffling the data.
		Random random = new Random();
		for(int i=0; i < imageCount; i++) {
			// Randomly assign another index to this value.
			int swapTarget = random.nextInt(imageCount);
			double[] tempImage = images[i];
			images[i] = images[swapTarget];
			images[swapTarget] = tempImage;

			double[] tempLabel = labels[i];
			labels[i] = labels[swapTarget];
			labels[swapTarget] = tempLabel;
		}

		// Pick a cutoff.  80% training?
		float learningRate = 0.1f;
		int trainingCutoff = (int)(imageCount*0.8f);
		for(int i=0; i < ITERATION_COUNT; i++) {
			double[][] batch = new double[BATCH_SIZE][images[0].length];
			double[][] target = new double[BATCH_SIZE][labels[0].length];
			// Pick N items at random.
			for(int j=0; j < BATCH_SIZE; j++) {
				int ex = random.nextInt(trainingCutoff);
				batch[j] = images[ex];
				target[j] = labels[ex];
			}
			// Train the model for an iteration.
			model.fitBatch(batch, target, learningRate, Model.Loss.SQUARED);
			// Check if we should report:
			if(i % REPORT_INTERVAL == 0) {
				learningRate *= 0.999;
				// Select an example from the test set.
				int ex = trainingCutoff+random.nextInt(imageCount-trainingCutoff);
				double[] guess = model.predict(images[ex]);
				// Display the image on the left and the guesses on the right.
				for(int r=0; r < rows; r++) {
					// Show the image.
					for(int c=0; c < columns; c++) {
						if(images[ex][c+r*columns] > 0.5f) {
							System.out.print("#");
						} else {
							System.out.print(".");
						}
					}

					// For each of our guesses, display some pretty graphs.
					if(r < 10) {
						if(labels[ex][r] > 0) {
							System.out.print(" [CORRECT]");
						} else {
							System.out.print(" [_______]");
						}
						System.out.print(" " + r + ": ");
						for(int m=0; m < guess[r]*10; m++) {
							System.out.print("#");
						}
					} else if(r == 10) {
						System.out.print(" ITERATION: " + i + "   LEARNING RATE: " + learningRate);
					}
					System.out.println();
				}
				System.out.println();
			}
		}

		// Save the model to a file.
		model.serializeToString();
	}
}
