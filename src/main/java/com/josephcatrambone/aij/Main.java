package com.josephcatrambone.aij;

import com.josephcatrambone.aij.networks.NeuralNetwork;
import com.josephcatrambone.aij.networks.RestrictedBoltzmannMachine;
import com.josephcatrambone.aij.trainers.BackpropTrainer;
import com.josephcatrambone.aij.trainers.RBMTrainer;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Application;
import javafx.concurrent.Task;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.GridPane;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import javafx.util.Duration;

import java.util.Random;


public class Main extends Application {
	public final int WIDTH = 800;
	public final int HEIGHT = 600;

	@Override
	public void start(Stage stage) {
		shapeDemo(stage);
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
		final RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(IMAGE_WIDTH*IMAGE_HEIGHT, 8);
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
			x.set(i, 0, Math.PI*i*2.0/(float)RESOLUTION);
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

	public static void main(String[] args) {
		launch(args);
	}
}
