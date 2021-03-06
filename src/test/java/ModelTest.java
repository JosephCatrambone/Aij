import com.josephcatrambone.aij.models.Sequential;
import com.josephcatrambone.aij.nodes.PadCropNode;
import org.junit.Test;
import org.junit.Assume;

import java.io.*;
import java.util.Random;
import java.util.zip.GZIPInputStream;

/**
 * Created by jcatrambone on 9/13/16.
 */
public class ModelTest {

	@Test
	public void saveRestoreTest() {
		Sequential m = new Sequential(20, 20);
		m.addConvLayer(1, 2, 2, 1, 1, Sequential.Activation.NONE);
		m.addConvLayer(1, 2, 2, 1, 1, Sequential.Activation.NONE);
		m.addFlattenLayer();
		m.addDenseLayer(10, Sequential.Activation.NONE);
		m.addDenseLayer(11, Sequential.Activation.TANH);
		m.addDenseLayer(12, Sequential.Activation.RELU);

		String model = m.serializeToString();
		Sequential m2 = new Sequential(20, 20);
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
		Sequential m = new Sequential(1, 2);
		m.addDenseLayer(10, Sequential.Activation.TANH);
		m.addDenseLayer(1, Sequential.Activation.SIGMOID);

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
				m.fitBatch(x, y, 0.5, Sequential.Loss.SQUARED);
			} else {
				m.fit(x, y, 0.5f, Sequential.Loss.SQUARED);
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
		Sequential m = new Sequential(1, 3);
		// Are there one, two, or three bits set?
		m.addDenseLayer(5, Sequential.Activation.SOFTMAX);
		m.addDenseLayer(3, Sequential.Activation.SOFTMAX);

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
			m.fit(x, y, 0.1f, Sequential.Loss.SQUARED);
			double loss = 0;
			double[][] predictions = m.predict(x);
			for(int j=0; j < predictions.length; j++) {
				loss += Math.abs(predictions[j][0] - y[j][0]);
				loss += Math.abs(predictions[j][1] - y[j][1]);
			}
			System.out.println(i + "\tParity loss: " + loss);
		}
	}

	private String luminanceToCharacter(double val) {
		if(val > 0.8f) {
			return "█";
		} else if(val > 0.6) {
			return "▓";
		} else if(val > 0.4) {
			return "▒";
		} else if(val > 0.2) {
			return "░";
		} else {
			return ".";
		}
	}

	private double[][] loadMNISTExamples(String filename) throws IOException {
		double[][] images;

		DataInputStream image_in = new DataInputStream(new GZIPInputStream(new FileInputStream(filename)));

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

		return images;
	}

	private double[][] loadMNISTLabels(String filename) throws IOException {
		double[][] labels; // One-hot.
		DataInputStream labels_in = new DataInputStream(new GZIPInputStream(new FileInputStream(filename)));

		// Read the labels.
		int magicNumber = labels_in.readInt();
		assert(magicNumber == 0x00000801);
		int labelCount = labels_in.readInt();
		labels = new double[labelCount][10];
		for(int labelNumber=0; labelNumber < labelCount; labelNumber++) {
			int label = labels_in.readUnsignedByte();
			labels[labelNumber][label] = 1.0f;
		}

		return labels;
	}

	@Test
	public void testMNIST() throws IOException {
		Assume.assumeTrue(new File("train-images-idx3-ubyte.gz").exists());
		Assume.assumeTrue(new File("train-labels-idx1-ubyte.gz").exists());

		final int ITERATION_COUNT = 10000000;
		final int BATCH_SIZE = 10;
		final int REPORT_INTERVAL = 100;
		Sequential model;
		double[][] images = loadMNISTExamples("train-images-idx3-ubyte.gz");
		double[][] labels = loadMNISTLabels("train-labels-idx1-ubyte.gz");

		// Verify we've got all the data and labels.
		assert(images.length == labels.length);

		int imageCount = images.length;
		int rows = 28;
		int columns = 28;

		// Build and train our model.
		model = new Sequential(rows, columns);
		model.addConvLayer(1, 3, 3, 2, 2, Sequential.Activation.RELU);
		model.addConvLayer(1, 3, 3, 2, 2, Sequential.Activation.RELU);
		model.addFlattenLayer();
		model.addDenseLayer(64, Sequential.Activation.RELU);
		model.addDenseLayer(32, Sequential.Activation.TANH);
		model.addDenseLayer(10, Sequential.Activation.SOFTMAX);

		// Split up the training data into target and test.
		// Start by shuffling the data.
		Random random = new Random();
		for(int i=0; i < imageCount; i++) {
			// Randomly assign another index to this value.
			int swapTarget = random.nextInt(imageCount-i)+i;
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
			model.fitBatch(batch, target, learningRate, Sequential.Loss.SQUARED);
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
						System.out.print(luminanceToCharacter(images[ex][c+r*columns]));
					}

					// For each of our guesses, display some pretty graphs.
					if(r < 10) {
						if(labels[ex][r] > 0) {
							System.out.print(" [C]");
						} else {
							System.out.print(" [_]");
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

	@Test
	public void testGenerateMNIST() throws IOException {
		final String trainingImagesFilename = "train-images-idx3-ubyte.gz";
		Assume.assumeTrue(new File(trainingImagesFilename).exists());

		float learningRate = 1e-3f;
		final double NOISE_LEVEL = 0.1;
		final int ITERATION_COUNT = 100000;
		final int BATCH_SIZE = 10;
		final int REPORT_INTERVAL = 100;
		Sequential model;

		int rows = 28;
		int columns = 28;

		// Build and train our model.
		model = new Sequential(rows, columns);
		model.addConvLayer(4, 3, 3, 2, 2, Sequential.Activation.RELU);
		model.addConvLayer(1, 3, 3, 2, 2, Sequential.Activation.RELU);
		model.addFlattenLayer();
		model.addDenseLayer(128, Sequential.Activation.RELU);
		model.addDenseLayer(20, Sequential.Activation.RELU); // Representation.
		model.addDenseLayer(128, Sequential.Activation.RELU);
		model.addDenseLayer(7*7*4, Sequential.Activation.RELU);
		model.addReshapeLayer(7, 7*4);
		model.addDeconvLayer(1, 3, 3, 2, 2, Sequential.Activation.RELU);
		model.addDeconvLayer(4, 3, 3, 2, 2, Sequential.Activation.RELU);
		//model.addLayer(new PadCropNode(rows, columns, model.getOutputNode())); // Pad an extra on the end.
		System.out.println("Sequential output size: " + model.getOutputNode().rows + ", " + model.getOutputNode().columns);

		// Load data.
		double[][] images = loadMNISTExamples(trainingImagesFilename);
		int imageCount = images.length;

		// Split up the training data into target and test.
		// Start by shuffling the data.
		Random random = new Random();
		for(int i=0; i < imageCount; i++) {
			// Randomly assign another index to this value.
			int swapTarget = random.nextInt(imageCount-i)+i;
			double[] tempImage = images[i];
			images[i] = images[swapTarget];
			images[swapTarget] = tempImage;
		}

		// Pick a cutoff.  80% training?
		for(int i=0; i < ITERATION_COUNT; i++) {
			double[][] batch = new double[BATCH_SIZE][images[0].length];
			double[][] label = new double[BATCH_SIZE][images[0].length];
			// Pick N items at random.
			for(int j=0; j < BATCH_SIZE; j++) {
				int ex = random.nextInt(imageCount);
				label[j] = images[ex];
				batch[j] = new double[images[0].length];
				for(int k=0; k < batch[j].length; k++) {
					batch[j][k] = images[ex][k] + random.nextGaussian()*NOISE_LEVEL;
				}
			}
			// Train the model for an iteration.
			model.fit(batch, label, learningRate, Sequential.Loss.SQUARED);
			// Check if we should report:
			if(i % REPORT_INTERVAL == 0 || i < 10) {
				System.out.println(i);
				//learningRate *= 0.999;
				// Select an example from the test set.
				int ex = random.nextInt(imageCount);
				double[] guess = model.predict(images[ex]);
				// Display the image on the left and the guesses on the right.
				for(int r=0; r < rows; r++) {
					// Show the image.
					for(int c=0; c < columns; c++) {
						double pixval = images[ex][c+r*columns];
						System.out.print(luminanceToCharacter(pixval));
					}
					System.out.print(" | ");
					for(int c=0; c < columns; c++) {
						double pixval = guess[c+r*columns];
						System.out.print(luminanceToCharacter(pixval));
					}
					System.out.println();
				}
				System.out.println();
			}
		}

		// Save the model to a file.
		System.out.println(model.serializeToString());
	}
}
