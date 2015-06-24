package com.josephcatrambone.aij;

import com.josephcatrambone.aij.networks.ConvolutionalNetwork;
import com.josephcatrambone.aij.networks.NeuralNetwork;
import com.josephcatrambone.aij.networks.OneToOneNetwork;
import com.josephcatrambone.aij.networks.RestrictedBoltzmannMachine;
import com.josephcatrambone.aij.trainers.BackpropTrainer;
import com.josephcatrambone.aij.trainers.ConvolutionalTrainer;
import com.josephcatrambone.aij.trainers.RBMTrainer;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Application;
import javafx.concurrent.Task;
import javafx.embed.swing.SwingFXUtils;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.*;
import javafx.scene.layout.GridPane;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import javafx.util.Duration;

import javax.imageio.ImageIO;
import java.io.*;
import java.util.Random;


public class Main extends Application {
	public final int WIDTH = 800;
	public final int HEIGHT = 600;

	@Override
	public void start(Stage stage) {
		mnistDemo(stage);
	}

	public void mnistDemo(Stage stage) {
		Matrix data = loadMNIST("train-images-idx3-ubyte");

		RestrictedBoltzmannMachine edgeDetector = new RestrictedBoltzmannMachine(5*5, 3*3);
		RBMTrainer rbmTrainer = new RBMTrainer();
		rbmTrainer.batchSize = 10;
		rbmTrainer.learningRate = 0.1;

		ConvolutionalNetwork layer0 = new ConvolutionalNetwork(
				edgeDetector, 28, 28, 5, 5, 3, 3, 1, 1, ConvolutionalNetwork.EdgeBehavior.ZEROS);

		ConvolutionalTrainer convTrainer = new ConvolutionalTrainer();
		convTrainer.operatorTrainer = rbmTrainer;
		convTrainer.learningRate = 0.1;
		convTrainer.minibatchSize = 10;
		convTrainer.batchSize = 100;

		convTrainer.train(layer0, data, null, null);

		System.exit(0);
	}

	public void threeParityDemo(Stage stage) {
		// Purely console demo.

		// Set up training.
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(6, 5);
		Matrix x = new Matrix(3, 6, 0.0);
		x.setRow(0, new double[]{1,1,0,0,0,0});
		x.setRow(1, new double[]{0,0,1,1,0,0});
		x.setRow(2, new double[]{0,0,0,0,1,1});

		RBMTrainer trainer = new RBMTrainer();
		trainer.learningRate = 0.1;
		trainer.batchSize = 10;
		trainer.notificationIncrement = 100;
		trainer.maxIterations = 100000000;

		Runnable updateFunction = new Runnable() {
			int i=0;

			@Override
			public void run() {
				Matrix daydream = rbm.daydream(1, 2); // One sample, two gibbs cycle
				System.out.println(i + ":" + trainer.lastError + " - " + daydream);
				Thread.yield();
				i++;
			}
		};

		// Run
		trainer.train(rbm, x, null, updateFunction);
	}

	public void imageDemo(Stage stage) {
		// Build UI
		stage.setTitle("Aij Test UI");
		GridPane pane = new GridPane();
		pane.setAlignment(Pos.CENTER);
		Scene scene = new Scene(pane, WIDTH, HEIGHT);
		Canvas canvas = new Canvas(WIDTH, HEIGHT);
		pane.add(canvas, 0, 0);
		stage.setScene(scene);
		stage.show();

		// Build data
		final int NUM_EXAMPLES = 1;
		Image img = new Image("D:\\tmp\\test.png", true);
		final Matrix examples = new Matrix(NUM_EXAMPLES, (int)(img.getWidth()*img.getHeight()));
		PixelReader pr = img.getPixelReader();
		for(int y=0; y < img.getHeight(); y++) {
			for(int x=0; x < img.getWidth(); x++) {
				examples.set(0, (int)(x+y*img.getWidth()), pr.getColor(x, y).getBrightness());
			}
		}
		final Matrix y = null; // Unsupervised.

		// Set up the drawing target.
		// Clear the image
		GraphicsContext gc = canvas.getGraphicsContext2D();
		gc.setFill(Color.BLACK);
		gc.fillRect(0, 0, WIDTH, HEIGHT);

		WritableImage imgOut = new WritableImage((int)img.getWidth(), (int)img.getHeight());
		PixelWriter pw = imgOut.getPixelWriter();
		//pane.add(new ImageView(img), 0, 0);

		// Build backend
		int CONV_SIZE = 16;
		OneToOneNetwork network = new OneToOneNetwork(CONV_SIZE*CONV_SIZE);
		ConvolutionalNetwork convnet = new ConvolutionalNetwork(network, (int)img.getWidth(), (int)img.getHeight(), CONV_SIZE, CONV_SIZE, CONV_SIZE, CONV_SIZE, CONV_SIZE/2, CONV_SIZE/2, ConvolutionalNetwork.EdgeBehavior.ZEROS);

		OneToOneNetwork.Monitor imageDisplay = new OneToOneNetwork.Monitor() {
			@Override
			public void run(Matrix intermediate) {
			}
		};

		Runnable updateFunction = new Runnable() {
			@Override
			public void run() {
			}
		};

		Timeline timeline = new Timeline();
		timeline.setCycleCount(Timeline.INDEFINITE);
		timeline.getKeyFrames().add(new KeyFrame(Duration.seconds(0.2), new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent event) {

			}
		}));
		timeline.playFromStart();
	}

	public void shapeDemo(Stage stage) {
		// Build UI
		stage.setTitle("Aij Test UI");
		GridPane pane = new GridPane();
		pane.setAlignment(Pos.CENTER);
		Scene scene = new Scene(pane, WIDTH, HEIGHT);
		Canvas canvas = new Canvas(WIDTH, HEIGHT);
		pane.add(canvas, 0, 0);
		stage.setScene(scene);
		stage.show();

		// Build data
		Random random = new Random();
		final int NUM_EXAMPLES = 100;
		final int IMAGE_WIDTH = 16;
		final int IMAGE_HEIGHT = 16;
		final int MIN_RECT_SIZE = 4;
		final Matrix x = new Matrix(NUM_EXAMPLES, IMAGE_WIDTH*IMAGE_HEIGHT);
		for(int i=0; i < NUM_EXAMPLES; i++) {
			// Draw a square
			Matrix example = Matrix.zeros(IMAGE_WIDTH, IMAGE_HEIGHT);
			int xStart = random.nextInt(IMAGE_WIDTH-MIN_RECT_SIZE-1);
			int width = random.nextInt(IMAGE_WIDTH-xStart);
			int yStart = random.nextInt(IMAGE_HEIGHT-MIN_RECT_SIZE);
			int height = random.nextInt(IMAGE_HEIGHT-yStart);
			for(int j=0; j < width; j++) {
				example.set(yStart, j, 1.0);
				example.set(yStart+height, j, 1.0);
			}
			for(int j=0; j < height; j++) {
				example.set(j, xStart, 1.0);
				example.set(j, xStart+width, 1.0);
			}
			// Flatten and add to examples
			example.reshape_i(1, IMAGE_WIDTH*IMAGE_HEIGHT);
			x.setRow(i, example);
		}
		final Matrix y = null; // Unsupervised.

		// Build backend
		final RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(IMAGE_WIDTH*IMAGE_HEIGHT, 16);
		RBMTrainer trainer = new RBMTrainer();
		trainer.batchSize = 10;
		trainer.learningRate = 0.01;
		trainer.notificationIncrement = 10;
		trainer.maxIterations = 11;

		Runnable updateFunction = new Runnable() {
			int iteration=0;

			@Override
			public void run() {
				// Clear the image
				GraphicsContext gc = canvas.getGraphicsContext2D();
				gc.setFill(Color.BLACK);
				gc.fillRect(0, 0, WIDTH, HEIGHT);

				// Draw a daydream
				Matrix preimage = rbm.daydream(1, 1);
				WritableImage img = new WritableImage(IMAGE_WIDTH, IMAGE_HEIGHT);
				PixelWriter pw = img.getPixelWriter();
				//pane.add(new ImageView(img), 0, 0);

				for(int i=0; i < IMAGE_WIDTH*IMAGE_HEIGHT; i++) {
					int x = i%IMAGE_WIDTH;
					int y = i/IMAGE_WIDTH;
					double color = preimage.get(0, i);
					if(color < 0) { color = 0; }
					if(color > 1.0) { color = 1.0; }
					pw.setColor(x, y, Color.gray(color));
				}
				gc.drawImage(img, 0, 0, IMAGE_WIDTH*10, IMAGE_HEIGHT*10);

				// Draw all the weights below it.
				for(int i=0; i < rbm.getNumOutputs(); i++) {
					img = new WritableImage(IMAGE_WIDTH, IMAGE_HEIGHT);
					pw = img.getPixelWriter();
					Matrix output = Matrix.zeros(1, rbm.getNumOutputs());
					output.set(0, i, 1.0);
					Matrix reconstruction = rbm.reconstruct(output);

					for(int j=0; j < IMAGE_WIDTH*IMAGE_HEIGHT; j++) {
						int x = j%IMAGE_WIDTH;
						int y = j/IMAGE_WIDTH;
						double color = reconstruction.get(0, j);
						if(color < 0) { color = 0; }
						if(color > 1.0) { color = 1.0; }
						pw.setColor(x, y, Color.gray(color));
					}
					pw.setColor(i, 0, Color.gray(1.0));

					gc.drawImage(img, i*(IMAGE_WIDTH*5), IMAGE_HEIGHT*10, IMAGE_WIDTH*5, IMAGE_HEIGHT*5);
				}

				System.out.println(iteration + ":" + trainer.lastError);
			}
		};

		Timeline timeline = new Timeline();
		timeline.setCycleCount(Timeline.INDEFINITE);
		timeline.getKeyFrames().add(new KeyFrame(Duration.seconds(0.2), new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent event) {
				trainer.train(rbm, x, y, updateFunction);
				try {
					Image visualized = visualizeRBM(rbm);
					ImageIO.write(SwingFXUtils.fromFXImage(visualized, null), "png", new File("output.png"));
				} catch(IOException ioe) {
					System.out.println("Problem writing output.png");
				}
			}
		}));
		timeline.playFromStart();
	}

	public void sinDemo(Stage stage) {
		// Build UI
		stage.setTitle("Aij Test UI");

		GridPane pane = new GridPane();
		pane.setAlignment(Pos.CENTER);

		Scene scene = new Scene(pane, WIDTH, HEIGHT);

		Canvas canvas = new Canvas(WIDTH, HEIGHT);
		pane.add(canvas, 0, 0);

		stage.setScene(scene);
		stage.show();

		// Build data
		final int RESOLUTION = 1000;
		final Matrix x = new Matrix(RESOLUTION, 1);
		for(int i=0; i < RESOLUTION; i++) {
			x.set(i, 0, Math.PI * i * 2.0 / (float) RESOLUTION);
		}
		final Matrix y = x.elementOp(v -> Math.sin(v));

		// Build backend
		final NeuralNetwork nn = new NeuralNetwork(new int[]{1, 10, 1}, new String[]{"linear", "tanh", "linear"});
		BackpropTrainer trainer = new BackpropTrainer();
		trainer.batchSize = 1;
		trainer.momentum = 0.9;
		trainer.learningRate = 0.01;
		trainer.notificationIncrement = 1000;
		trainer.maxIterations = 1001;

		Runnable updateFunction = new Runnable() {
			@Override
			public void run() {
				Matrix prediction = nn.predict(x);
				GraphicsContext gc = canvas.getGraphicsContext2D();
				gc.setFill(Color.BLACK);
				gc.fillRect(0, 0, WIDTH, HEIGHT);
				gc.setFill(Color.WHITE);
				for(int i=0; i < RESOLUTION-1; i++) {
					gc.fillOval(i*WIDTH/RESOLUTION, (1+y.get(i, 0))*HEIGHT/2.0, 1.0, 1.0);
				}
				gc.setFill(Color.BLUE);
				for(int i=0; i < RESOLUTION-1; i++) {
					gc.fillOval(i*WIDTH/RESOLUTION, (1+prediction.get(i, 0))*HEIGHT/2.0, 1.0, 1.0);
				}
			}
		};

		// Redraw the UI in the main thread.
		// We're abusing Java animation because we can only redraw from the main thread and need to do so with events.
		Timeline timeline = new Timeline();
		timeline.setCycleCount(Timeline.INDEFINITE);
		timeline.getKeyFrames().add(new KeyFrame(Duration.seconds(0.01), new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent event) {
				trainer.train(nn, x, y, updateFunction);
			}
		}));
		timeline.playFromStart();
	}

	/*** visualizeRBM
	 * Given an RBM as input, return an image which shows the sensitivity of each pathway.
	 * Attempts to produce a square image.
	 * @param rbm
	 * @return
	 */
	public Image visualizeRBM(RestrictedBoltzmannMachine rbm) {
		int outputNeurons = rbm.getNumOutputs();
		int inputNeurons = rbm.getNumInputs();
		int subImgWidth = (int)Math.sqrt(inputNeurons);
		int imgWidth = (int)Math.ceil(Math.sqrt(outputNeurons))*subImgWidth;
		WritableImage output = new WritableImage(imgWidth, imgWidth);
		PixelWriter pw = output.getPixelWriter();

		for(int i=0; i < outputNeurons; i++) {
			int subImgOffsetX = i%((int)Math.sqrt(outputNeurons));
			int subImgOffsetY = i/((int)Math.sqrt(outputNeurons));

			// Set one item hot and reconstruct
			Matrix stim = new Matrix(1, outputNeurons);
			stim.set(0, i, 1.0);
			Matrix reconstruction = rbm.reconstruct(stim);

			// Rebuild and draw input to image
			for(int j=0; j < inputNeurons; j++) {
				double val = reconstruction.get(0, j);
				if(val < 0) { val = 0; }
				if(val > 1) { val = 1; }
				pw.setColor(subImgOffsetX*subImgWidth + j%subImgWidth, subImgOffsetY*subImgWidth + j/subImgWidth, Color.gray(val));
			}
		}

		return output;
	}

	public Matrix loadMNIST(final String filename) {
		int numImages = -1;
		int imgWidth = -1;
		int imgHeight = -1;
		Matrix trainingData = null;

		// Load MNIST training data.
		try(FileInputStream fin = new FileInputStream(filename); DataInputStream din = new DataInputStream(fin)) {
			din.readInt();
			assert(din.readInt() == 2051);
			numImages = din.readInt();
			imgHeight = din.readInt();
			imgWidth = din.readInt();
			System.out.println("numImages: " + numImages);
			System.out.println("height: " + imgHeight);
			System.out.println("width: " + imgWidth);
			trainingData = new Matrix(numImages, imgWidth*imgHeight);
			for(int i=0; i < numImages; i++) {
				for(int y=0; y < imgHeight; y++) {
					for(int x=0; x < imgWidth; x++) {
						int grey = ((int)din.readByte()) & 0xFF; // Java is always signed.  Need to and with 0xFF to undo it.
						trainingData.set(i, x+y*imgWidth, ((double)(grey-128.0))/128.0);
					}
				}
				if(i%1000 == 0) {
					System.out.println("Loaded " + i + " out of " + numImages);
				}
			}
			fin.close();
		} catch(FileNotFoundException fnfe) {
			System.out.println("Unable to find and load file.");
			System.exit(-1);
		} catch(IOException ioe) {
			System.out.println("IO Exception while reading data.");
			System.exit(-1);
		}
		return trainingData;
	}

	public static void main(String[] args) {
		launch(args);
	}
}
