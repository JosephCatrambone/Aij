package com.josephcatrambone.aij;

import com.josephcatrambone.aij.networks.*;
import com.josephcatrambone.aij.trainers.BackpropTrainer;
import com.josephcatrambone.aij.trainers.ConvolutionalTrainer;
import com.josephcatrambone.aij.trainers.RBMTrainer;
import com.josephcatrambone.aij.utilities.ImageTools;
import com.josephcatrambone.aij.utilities.NetworkIOTools;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import javafx.stage.WindowEvent;
import javafx.util.Duration;

import java.io.*;
import java.util.Random;
import java.util.Scanner;
import java.util.function.Consumer;
import java.util.logging.Logger;


public class Main extends Application {
	private static final Logger LOGGER = Logger.getLogger(Main.class.getName());
	public final int WIDTH = 800;
	public final int HEIGHT = 600;

	@Override
	public void start(Stage stage) {
		mnistDemo(stage);
	}

	public void pokemonDemo(Stage stage) {
		LOGGER.info("Training pokemon RBM.");

		// Learning data and consts
		Random random = new Random();
		final String POKEMON_PATH = "D:\\Source\\pokemon\\main-sprites\\red-blue\\";
		final String POKEMON_RBM = "pokemon_rbm.net";
		final int HIDDEN_SIZE = 20*20;
		final int NUM_POKEMON = 151;
		final int IMG_WIDTH = 40;
		final int IMG_HEIGHT = 40;
		Matrix data = Matrix.zeros(NUM_POKEMON, IMG_WIDTH*IMG_HEIGHT);
		Network temp = null; // Used to load because the others have to be final.
		final RestrictedBoltzmannMachine rbm;

		// Load or build RBM for learning
		try(BufferedReader fin = new BufferedReader(new FileReader(POKEMON_RBM))) {
			StringBuilder sb = new StringBuilder();
			String line = "";
			while((line = fin.readLine()) != null) {
				sb.append(line + "\n");
			}
			Matrix weights = NetworkIOTools.StringToMatrix(sb.toString()); //NetworkIOTools.LoadNetworkFromDisk(POKEMON_RBM);
			//temp = new RestrictedBoltzmannMachine(weights.numRows(), weights.numColumns());
			//temp.setWeights(0, weights);
		} catch(IOException ioe) {}

		if(temp != null) {
			// Load RBM
			LOGGER.info("Loaded RBM from file." + POKEMON_RBM);
			rbm = (RestrictedBoltzmannMachine)temp;
		} else {
			LOGGER.info("Training RBM on Pokemon from folder " + POKEMON_PATH);

			// Load data
			for(int i=0; i < NUM_POKEMON; i++) {
				LOGGER.info("Loaded Pokemon " + i);
				Matrix mat = ImageTools.ImageFileToMatrix(POKEMON_PATH + i + ".png", IMG_WIDTH, IMG_HEIGHT);
				data.setRow(i, mat.reshape_i(1, IMG_WIDTH * IMG_HEIGHT));
			}

			// Train RBM
			rbm = new RestrictedBoltzmannMachine(IMG_WIDTH*IMG_HEIGHT, HIDDEN_SIZE);
			RBMTrainer rbmTrainer = new RBMTrainer();
			rbmTrainer.batchSize = 10;
			rbmTrainer.learningRate = 0.1;
			rbmTrainer.notificationIncrement = 100;
			rbmTrainer.maxIterations = 20000;
			rbmTrainer.earlyStopError = 0.0; // Disable early out.

			// Notification
			Runnable notification = new Runnable() {
				int i = 0;
				@Override
				public void run() {
					LOGGER.info("Iteration " + (i * rbmTrainer.notificationIncrement) + " error: " + rbmTrainer.lastError);
					ImageTools.FXImageToDisk(visualizeRBM(rbm, null, true), "rbm_in_training.png");
					i++;
				}
			};

			rbmTrainer.train(rbm, data, null, notification);

			// Save
			LOGGER.info("Saving network to file " + POKEMON_RBM);
			try(BufferedWriter fout = new BufferedWriter(new FileWriter(POKEMON_RBM))) {
				fout.write(NetworkIOTools.MatrixToString(rbm.getWeights(0)));
				fout.close();
			} catch(IOException ioe) {
				System.err.println("Error writing network to disk: " + ioe);
			}
		}

		// Set up UI
		// We should do this in Scene Builder and make FXML, but it's only a demo.
		stage.setTitle("Aij Test UI");
		HBox back = new HBox();
		ImageView imageView = new ImageView();
		imageView.setSmooth(false);
		imageView.setFitWidth(IMG_WIDTH * 10);
		imageView.setFitHeight(IMG_HEIGHT * 10);
		back.getChildren().add(imageView);
		VBox controls = new VBox();
		TextField cyclesInput = new TextField();
		cyclesInput.setText("1");
		controls.getChildren().add(cyclesInput);
		TextField threshold = new TextField();
		threshold.setText("0.9");
		controls.getChildren().add(threshold);
		CheckBox stochasticIntermediate = new CheckBox();
		controls.getChildren().add(stochasticIntermediate);
		CheckBox stochasticFinal = new CheckBox();
		controls.getChildren().add(stochasticFinal);
		Button dreamButton = new Button();
		dreamButton.setText("Generate");
		dreamButton.setOnAction((ActionEvent ae) -> {
			final Matrix greyImage = rbm.reconstruct(Matrix.random(1, HIDDEN_SIZE));
			greyImage.reshape_i(IMG_WIDTH, IMG_HEIGHT);
			greyImage.normalize_i();
			//final Matrix daydream = rbm.daydream(1, Integer.parseInt(cyclesInput.getText()), stochasticIntermediate.isSelected(), stochasticFinal.isSelected());
			//final Matrix reshaped = daydream.reshape_i(IMG_WIDTH, IMG_HEIGHT * BPP);
			//final Matrix greyImage = ImageTools.BitMatrixToGrayMatrix(reshaped, Double.parseDouble(threshold.getText()), BPP);
			Image img = ImageTools.MatrixToFXImage(greyImage);
			imageView.setImage(img);
			System.out.println("Image drawn.  Looping.");
		});
		controls.getChildren().add(dreamButton);
		back.getChildren().add(controls);

		Scene scene = new Scene(back, WIDTH, HEIGHT);
		stage.setScene(scene);
		stage.show();
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
				examples.setRow(i, ImageTools.ImageFileToMatrix(scanner.next(), IMG_WIDTH, IMG_HEIGHT).reshape_i(1, IMG_WIDTH * IMG_HEIGHT));
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
		RestrictedBoltzmannMachine edgeDetector = new RestrictedBoltzmannMachine(RBM_WIDTH*RBM_HEIGHT, OUT_WIDTH*OUT_HEIGHT);
		//RestrictedBoltzmannMachine edgeDetector = new RestrictedBoltzmannMachine(28*28, 8*8);
		RBMTrainer rbmTrainer = new RBMTrainer();
		rbmTrainer.batchSize = 10;
		rbmTrainer.learningRate = 0.1;
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
		ImageView imageView = new ImageView(visualizeRBM(edgeDetector, null, false));
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
						layer1Trainer.train(layer1, examples, y, null);
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
						Image img = visualizeRBM(edgeDetector, null, true);
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
			NetworkIOTools.SaveNetworkToDisk(edgeDetector, "rbm0_8x8_8x8.net");
		});
	}

	public void mnistDemo(Stage stage) {
		final int HIDDEN_SIZE = 400;
		final int GIBBS_SAMPLES = 1;
		Random random = new Random();

		// Load training data
		Matrix data = loadMNIST("train-images-idx3-ubyte");

		// Build RBM for learning
		final RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(28*28, HIDDEN_SIZE);
		final RBMTrainer rbmTrainer = new RBMTrainer();
		rbmTrainer.batchSize = 10; // 5x20
		rbmTrainer.learningRate = 0.1;
		rbmTrainer.maxIterations = 100;
		rbmTrainer.gibbsSamples = 1;

		/*
		// Make mean filter
		MeanFilterNetwork mfn = new MeanFilterNetwork(28*28);
		MeanFilterTrainer mft = new MeanFilterTrainer();
		mft.train(mfn, data, null, null);

		// Remove mean
		Matrix examples = mfn.predict(data);
		*/
		final Matrix examples = data;

		// Spawn a separate training thread.
		Thread trainerThread = new Thread(() -> {
			int cycles = 0;
			while(true) {
				synchronized (rbm) {
					rbmTrainer.train(rbm, examples, null, null);
				}
				// Change training params.
				if(cycles++ > 100) {
					try(BufferedWriter fout = new BufferedWriter(new FileWriter(new File("rbm.txt")))) {
						fout.write("visible_bias ");
						fout.write(NetworkIOTools.MatrixToString(rbm.getVisibleBias()));
						fout.write("\n");
						fout.write("hidden_bias");
						fout.write(NetworkIOTools.MatrixToString(rbm.getHiddenBias()));
						fout.write("\n");
						fout.write("weights ");
						fout.write(NetworkIOTools.MatrixToString(rbm.getWeights(0)));
					} catch(IOException ioe) {

					}
					rbmTrainer.gibbsSamples += 1;
					rbmTrainer.learningRate *= 0.8;
					System.out.println("Bumping steps to " + rbmTrainer.gibbsSamples);
					cycles = 0;
				}

				// Stop if interrupted.
				if(Thread.currentThread().isInterrupted()) {
					return;
				}
				// Yield to give someone else a chance to run.
				try {
					System.out.println("Training error:" + rbmTrainer.lastError);
					Thread.sleep(1);
				} catch(InterruptedException ie) {
					// If interrupted, abort.
					return;
				}
			}
		});
		trainerThread.start();

		// Set up UI
		stage.setTitle("Aij Test UI");
		GridPane pane = new GridPane();
		pane.setAlignment(Pos.CENTER);
		Scene scene = new Scene(pane, WIDTH, HEIGHT);
		ImageView imageView = new ImageView(visualizeRBM(rbm, null, true));
		ImageView exampleView = new ImageView(ImageTools.MatrixToFXImage(Matrix.random(28, 28), true));
		pane.add(imageView, 0, 0);
		pane.add(exampleView, 1, 0);
		//pane.add(imageView);
		stage.setScene(scene);
		stage.show();

		// Repeated draw.
		Timeline timeline = new Timeline();
		timeline.setCycleCount(Timeline.INDEFINITE);
		timeline.getKeyFrames().add(new KeyFrame(Duration.seconds(1.0), new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent event) {
				if (stage.isFocused()) {
					//synchronized (rbm) {
					// Draw RBM
					//rbmTrainer.train(edgeDetector, data, null, null);
					System.out.println("Trained.  Drawing...");
					Image img = visualizeRBM(rbm, null, true);
					imageView.setImage(img);

					// Render an example
					final Matrix input = Matrix.random(1, 28 * 28);
					Matrix ex = rbm.daydream(input, GIBBS_SAMPLES);

					exampleView.setImage(ImageTools.MatrixToFXImage(ex.reshape_i(28, 28), true));

					System.out.println("Error: " + rbmTrainer.lastError);
					//}
				}
			}
		}));
		timeline.playFromStart();

		stage.setOnCloseRequest((WindowEvent w) -> {
			timeline.stop();
			trainerThread.interrupt();
		});
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
			for(int k=yStart; k < yStart+height; k++) {
				for(int j=xStart; j < xStart+width; j++) {
					example.set(k, j, 1.0);
				}
			}

			// Flatten and add to examples
			example.reshape_i(1, IMAGE_WIDTH*IMAGE_HEIGHT);
			x.setRow(i, example);
		}
		final Matrix y = null; // Unsupervised.

		// Build backend
		final RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(IMAGE_WIDTH*IMAGE_HEIGHT, 16);
		RBMTrainer trainer = new RBMTrainer();
		trainer.batchSize = 1;
		trainer.learningRate = 0.1;
		trainer.notificationIncrement = 200;
		trainer.maxIterations = 201;

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
		timeline.getKeyFrames().add(new KeyFrame(Duration.seconds(0.5), new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent event) {
				trainer.train(rbm, x, y, updateFunction);
				//try {
				Image visualized = visualizeRBM(rbm, null, false);
				//	ImageIO.write(SwingFXUtils.fromFXImage(visualized, null), "png", new File("output.png"));
				//} catch(IOException ioe) {
				//	System.out.println("Problem writing output.png");
				//}
				Thread.yield();
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
	public Image visualizeRBM(RestrictedBoltzmannMachine rbm, MeanFilterNetwork mean, boolean normalizeIntensity) {
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
			Matrix reconstruction = rbm.reconstruct(stim, false);

			if(mean != null) {
				reconstruction = mean.reconstruct(reconstruction);
			}

			// Normalize data if needed
			if(normalizeIntensity) {
				reconstruction.normalize_i();
			}

			// Rebuild and draw input to image
			for(int j=0; j < reconstruction.numColumns(); j++) {
				double val = reconstruction.get(0, j);
				if(val < 0) { val = 0; }
				if(val > 1) { val = 1; }
				pw.setColor(subImgOffsetX + (j%subImgWidth), subImgOffsetY + (j/subImgWidth), Color.gray(val));
			}
		}

		return output;
	}

	public Matrix loadMNIST(final String filename) {
		int numImages = 50000;
		int imgWidth = -1;
		int imgHeight = -1;
		Matrix trainingData = null;

		// Load MNIST training data.
		try(FileInputStream fin = new FileInputStream(filename); DataInputStream din = new DataInputStream(fin)) {
			din.readInt();
			assert(din.readInt() == 2051);
			numImages = Math.min(din.readInt(), numImages);
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
						trainingData.set(i, x+(y*imgWidth), ((double)grey)/256.0);
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
