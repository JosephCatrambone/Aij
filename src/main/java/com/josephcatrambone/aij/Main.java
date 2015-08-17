package com.josephcatrambone.aij;

import com.josephcatrambone.aij.networks.*;
import com.josephcatrambone.aij.trainers.BackpropTrainer;
import com.josephcatrambone.aij.trainers.ConvolutionalTrainer;
import com.josephcatrambone.aij.trainers.MeanFilterTrainer;
import com.josephcatrambone.aij.trainers.RBMTrainer;
import com.josephcatrambone.aij.utilities.ImageTools;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Application;
import javafx.embed.swing.SwingFXUtils;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.SnapshotParameters;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.*;
import javafx.scene.layout.GridPane;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import javafx.stage.WindowEvent;
import javafx.util.Duration;

import javax.imageio.ImageIO;
import java.io.*;
import java.util.Random;
import java.util.Scanner;
import java.util.function.Consumer;


public class Main extends Application {
	public final int WIDTH = 800;
	public final int HEIGHT = 600;

	@Override
	public void start(Stage stage) {
		imageDemo(stage);
	}

	public void imageDemo(Stage stage) {
		// Force all our loaded images to this size, smoothly resizing, and NOT loading in the backgrouns.
		final int NUM_EXAMPLES = 500;
		int IMG_WIDTH = 64;
		int IMG_HEIGHT = 64;
		int DISPLAY_WIDTH = 512;
		int DISPLAY_HEIGHT = 512;
		int RBM_WIDTH = 4;
		int RBM_HEIGHT = 4;
		int OUT_WIDTH = 8;
		int OUT_HEIGHT = 8;

		final Matrix examples = new Matrix(NUM_EXAMPLES, IMG_WIDTH*IMG_HEIGHT);

		try {
			Scanner scanner = new Scanner(new File("test_images.txt")); // Read filenames from stdin

			// Load training data
			// Build data
			for(int i=0; i < NUM_EXAMPLES; i++) {
				examples.setRow(i, ImageTools.ImageFileToMatrix(scanner.next(), IMG_WIDTH, IMG_HEIGHT).reshape_i(1, IMG_WIDTH*IMG_HEIGHT));
			}
		} catch(FileNotFoundException fnfe) {
			System.err.println("Unable to load image: " + fnfe);
			System.exit(-1);
		}

		final Matrix y = null; // Unsupervised.

		// DEBUG

		Consumer <Matrix> trainingDataMonitor = (Matrix m) -> {
			Image visualized = ImageTools.MatrixToFXImage(m.reshape_i(RBM_WIDTH, RBM_HEIGHT));
			ImageTools.FXImageToDisk(visualized, "output.png");
		};

		// DEBUG

		// Build network

		// Train the mean filter on the input
		MeanFilterNetwork meanFilter = new MeanFilterNetwork(RBM_WIDTH*RBM_HEIGHT);
		MeanFilterTrainer meanFilterTrainer = new MeanFilterTrainer();

		ConvolutionalNetwork layer0 = new ConvolutionalNetwork(meanFilter, IMG_WIDTH, IMG_HEIGHT, RBM_WIDTH, RBM_HEIGHT, RBM_WIDTH, RBM_HEIGHT, RBM_WIDTH, RBM_HEIGHT, ConvolutionalNetwork.EdgeBehavior.ZEROS);
		ConvolutionalTrainer layer0Trainer = new ConvolutionalTrainer();
		layer0Trainer.operatorTrainer = meanFilterTrainer;
		layer0Trainer.subwindowsPerExample = 10000;
		layer0Trainer.examplesPerBatch = NUM_EXAMPLES;
		layer0Trainer.maxIterations = 1;

		layer0Trainer.train(layer0, examples, null, null);

		Matrix normalizedExamples = layer0.predict(examples);

		RestrictedBoltzmannMachine edgeDetector = new RestrictedBoltzmannMachine(RBM_WIDTH*RBM_HEIGHT, OUT_WIDTH*OUT_HEIGHT);
		//RestrictedBoltzmannMachine edgeDetector = new RestrictedBoltzmannMachine(28*28, 8*8);
		RBMTrainer rbmTrainer = new RBMTrainer();
		rbmTrainer.batchSize = 10;
		rbmTrainer.learningRate = 0.01;
		rbmTrainer.maxIterations = 10;

		//ConvolutionalNetwork layer1 = new ConvolutionalNetwork(edgeDetector, IMG_WIDTH, IMG_HEIGHT, RBM_WIDTH, RBM_HEIGHT, OUT_WIDTH, OUT_HEIGHT, 1, 1, ConvolutionalNetwork.EdgeBehavior.ZEROS);
		ConvolutionalNetwork layer1 = new ConvolutionalNetwork(edgeDetector, IMG_WIDTH, IMG_HEIGHT, RBM_WIDTH, RBM_HEIGHT, OUT_WIDTH, OUT_HEIGHT, RBM_WIDTH, RBM_HEIGHT, ConvolutionalNetwork.EdgeBehavior.ZEROS);
		ConvolutionalTrainer layer1Trainer = new ConvolutionalTrainer();
		layer1Trainer.operatorTrainer = rbmTrainer;
		layer1Trainer.subwindowsPerExample = 1000;
		layer1Trainer.examplesPerBatch = NUM_EXAMPLES;
		layer1Trainer.maxIterations = 1;

		// Set up UI
		stage.setTitle("Aij Test UI");
		GridPane pane = new GridPane();
		pane.setAlignment(Pos.CENTER);
		Scene scene = new Scene(pane, WIDTH, HEIGHT);
		ImageView imageView = new ImageView(visualizeRBM(edgeDetector, false));
		imageView.setFitWidth(DISPLAY_WIDTH);
		imageView.setFitHeight(DISPLAY_HEIGHT);
		pane.getChildren().add(imageView);
		//pane.add(imageView);
		stage.setScene(scene);
		stage.show();

		Runnable trainerRunnable = () -> {
			while(true) {
				synchronized (layer1Trainer) {
					if (!layer1Trainer.isTraining) {
						System.out.println("Training...");
						layer1Trainer.train(layer1, normalizedExamples, y, null);
					}
				}
				// Stop if interrupted.
				if(Thread.currentThread().isInterrupted()) {
					return;
				}
				// Yield to give someone else a chance to run.
				try {
					System.out.println("Training error:" + rbmTrainer.lastError);
					Thread.sleep(10);
				} catch(InterruptedException ie) {
					// If interrupted, abort.
					return;
				}
			}
		};
		Thread trainerThread = new Thread(trainerRunnable);
		trainerThread.start();

		// Repeated draw.
		Timeline timeline = new Timeline();
		timeline.setCycleCount(Timeline.INDEFINITE);
		timeline.getKeyFrames().add(new KeyFrame(Duration.seconds(2.0), new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent event) {
				// If another action happens to fire in the meantime, don't go crazy.
				synchronized (layer1Trainer) {
					if (!layer1Trainer.isTraining) {
						//rbmTrainer.train(edgeDetector, data, null, null);
						System.out.println("Drawing.");
						Image img = visualizeRBM(edgeDetector, true);
						imageView.setImage(img);
					} else {
						System.out.println("Drawing deferred.  Training in progres...");
					}
				}
			}
		}));
		timeline.playFromStart();

		stage.setOnCloseRequest((WindowEvent w) -> {
			timeline.stop();
			trainerThread.interrupt();
			try{
				trainerThread.join(1000);
			} catch(InterruptedException ie) {}
			saveNetwork(meanFilter, "meanFilter_8x8.net");
			saveNetwork(edgeDetector, "rbm0_8x8_8x8.net");
		});
	}

	public void mnistDemo(Stage stage) {
		// Load training data
		Matrix data = loadMNIST("train-images-idx3-ubyte");

		RestrictedBoltzmannMachine edgeDetector = new RestrictedBoltzmannMachine(5*5, 16*16);
		//RestrictedBoltzmannMachine edgeDetector = new RestrictedBoltzmannMachine(28*28, 8*8);
		RBMTrainer rbmTrainer = new RBMTrainer();
		rbmTrainer.batchSize = 20; // 5x20
		rbmTrainer.learningRate = 0.05;
		rbmTrainer.maxIterations = 10;

		ConvolutionalNetwork layer0 = new ConvolutionalNetwork(edgeDetector, 28, 28, 5, 5, 16, 16, 1, 1, ConvolutionalNetwork.EdgeBehavior.ZEROS);
		ConvolutionalTrainer convTrainer = new ConvolutionalTrainer();
		convTrainer.operatorTrainer = rbmTrainer;
		convTrainer.subwindowsPerExample = 1000;
		convTrainer.examplesPerBatch = 5;
		convTrainer.maxIterations = 1;

		// Set up UI
		stage.setTitle("Aij Test UI");
		GridPane pane = new GridPane();
		pane.setAlignment(Pos.CENTER);
		Scene scene = new Scene(pane, WIDTH, HEIGHT);
		ImageView imageView = new ImageView(visualizeRBM(edgeDetector, false));
		pane.getChildren().add(imageView);
		//pane.add(imageView);
		stage.setScene(scene);
		stage.show();

		// Repeated draw.
		Timeline timeline = new Timeline();
		timeline.setCycleCount(Timeline.INDEFINITE);
		timeline.getKeyFrames().add(new KeyFrame(Duration.seconds(0.2), new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent event) {
				System.out.println("Training...");
				convTrainer.train(layer0, data, null, null);
				//rbmTrainer.train(edgeDetector, data, null, null);
				System.out.println("Trained.  Drawing...");
				Image img = visualizeRBM(edgeDetector, false);
				imageView.setImage(img);
				System.out.println(img.getWidth() + " x " + img.getHeight() + " image drawn.  Looping.");
			}
		}));
		timeline.playFromStart();

		//System.exit(0);
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
		final int NUM_EXAMPLES = 1000;
		final int IMAGE_WIDTH = 16;
		final int IMAGE_HEIGHT = 16;
		final int MIN_RECT_SIZE = 4;
		final Matrix x = new Matrix(NUM_EXAMPLES, IMAGE_WIDTH*IMAGE_HEIGHT);
		for(int i=0; i < NUM_EXAMPLES; i++) {
			// Draw a square
			Matrix example = Matrix.zeros(IMAGE_WIDTH, IMAGE_HEIGHT);
			int xStart = 1+random.nextInt(IMAGE_WIDTH-MIN_RECT_SIZE-1);
			int width = random.nextInt(IMAGE_WIDTH-xStart);
			int yStart = 1+random.nextInt(IMAGE_HEIGHT-MIN_RECT_SIZE-1);
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
		trainer.learningRate = 0.1;
		trainer.notificationIncrement = 2000;
		trainer.maxIterations = 2001;

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
				//try {
				Image visualized = visualizeRBM(rbm, false);
				//	ImageIO.write(SwingFXUtils.fromFXImage(visualized, null), "png", new File("output.png"));
				//} catch(IOException ioe) {
				//	System.out.println("Problem writing output.png");
				//}
			}
		}));
		timeline.playFromStart();

		stage.setOnCloseRequest((WindowEvent w) -> {
			timeline.stop();
		});
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
	 * @param normalizeIntensity
	 * @return
	 */
	public Image visualizeRBM(RestrictedBoltzmannMachine rbm, boolean normalizeIntensity) {
		int outputNeurons = rbm.getNumOutputs();
		int inputNeurons = rbm.getNumInputs();
		int xSampleCount = (int)Math.ceil(Math.sqrt(outputNeurons));
		int subImgWidth = (int)Math.ceil(Math.sqrt(inputNeurons));
		int imgWidth = (int)Math.ceil(Math.sqrt(outputNeurons))*subImgWidth;
		WritableImage output = new WritableImage(imgWidth, imgWidth);
		PixelWriter pw = output.getPixelWriter();

		for(int i=0; i < outputNeurons; i++) {
			int subImgOffsetX = subImgWidth*(i%xSampleCount);
			int subImgOffsetY = subImgWidth*(i/xSampleCount);

			// Set one item hot and reconstruct
			Matrix stim = new Matrix(1, outputNeurons);
			stim.set(0, i, 1.0);
			Matrix reconstruction = rbm.reconstruct(stim);

			// Normalize data if needed
			double low = 0;
			double high = 1;
			if(normalizeIntensity) {
				low = Double.MAX_VALUE;
				high = Double.MIN_VALUE;
				for(int j=0; j < reconstruction.numColumns(); j++) {
					double val = reconstruction.get(0, j);
					if (val < low) { low = val; }
					if (val > high) { high = val; }
				}
			}

			// Rebuild and draw input to image
			for(int j=0; j < reconstruction.numColumns(); j++) {
				double val = reconstruction.get(0, j);
				val = (val-low)/(high-low);
				if(val < 0) { val = 0; }
				if(val > 1) { val = 1; }
				pw.setColor(subImgOffsetX + (j%subImgWidth), subImgOffsetY + (j/subImgWidth), Color.gray(val));
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

	public boolean saveNetwork(Network n, String filename) {
		try {
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(filename)));
			oos.writeObject(n);
			oos.close();
		} catch(IOException ioe) {
			return false;
		}
		return true;
	}

	public Network loadNetwork(String filename) {
		Network net = null;
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(new File(filename)));
			net = (Network)ois.readObject();
			ois.close();
		} catch(IOException ioe) {
			return null;
		} catch(ClassNotFoundException cnfe) {
			return null;
		}
		return net;
	}

	public static void main(String[] args) {
		launch(args);
	}
}
