import com.josephcatrambone.aij.Model;
import org.junit.Test;

import java.io.*;
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
		Model model;
		float[][] images;
		float[][] labels; // One-hot.

		DataInputStream image_in = new DataInputStream(new GZIPInputStream(new FileInputStream("train-images-idx3-ubyte.gz")));
		DataInputStream labels_in = new DataInputStream(new GZIPInputStream(new FileInputStream("train-labels-idx1-ubyte.gz")));

		// Read the images first.
		int magicNumber = image_in.readInt();
		assert(magicNumber == 0x00000803); // 2051 for training images.  2049 for training labels.
		int imageCount = image_in.readInt();
		int rows = image_in.readInt();
		int columns = image_in.readInt();
		// Images are row-wise, which is great because so is our model.
		images = new float[imageCount][rows*columns];
		for(int imageNumber=0; imageNumber < imageCount; imageNumber++) {
			for(int c=0; c < columns; c++) {
				for(int r=0; r < rows; r++) {
					images[imageNumber][c + r*columns] = image_in.readUnsignedByte()/255.0f;
				}
			}
		}

		model = new Model(imageCount, rows*columns);
	}
}
