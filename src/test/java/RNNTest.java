import com.josephcatrambone.aij.*;

import com.josephcatrambone.aij.models.LSTM;
import com.josephcatrambone.aij.nodes.*;
import org.junit.Test;

import java.io.*;
import java.util.*;
import java.util.function.DoubleConsumer;

/**
 * Created by jcatrambone on 2/6/17.
 */
public class RNNTest {

	public Random random = new Random();
	public int inputSize = 127;
	public int hiddenSize = 127;
	public int numSteps = 5;
	public int batchSize = 1;

	private String echoStringWhileLessThanMinLength(String s, int minLength) {
		do {
			s = s + "~" + s; // At least one duplicate.
		} while(s.length() < minLength);
		return s;
	}

	public double[] encodeCharacterToProbabilityDistribution(String s) {
		double[] enc = new double[127];
		int chr = (int)s.charAt(0);
		if(chr > 127) { chr = (int)' '; }
		enc[chr] = 1.0;
		return enc;
	}

	public String selectCharacterFromProbabilityDistribution(double[] probs) {
		// First calculate sum so we can normalize it.
		double min = 10;
		double max = -10;
		double sum = 0;
		for(int i=32; i < probs.length; i++) {
			min = Math.min(min, probs[i]);
			max = Math.max(max, probs[i]);
		}
		if(min == max) { max += 1.0e-6; }
		for(int i=32; i < probs.length; i++) {
			sum += (probs[i]-min)/(max-min);
		}

		// Then generate a random number.
		double energy = random.nextDouble()*sum; // From 0 to 1.
		for(int charPosition=0; charPosition < probs.length; charPosition++) {
			// The amount of energy required to get to the next spot...
			double barrierEnergy = (probs[charPosition]-min)/(max-min);
			if(energy < barrierEnergy) {
				if(charPosition < 32) {
					return "";
				} else {
					return "" + (char)charPosition;
				}
			} else {
				energy -= barrierEnergy;
			}
		}
		return "\n";
	}

	public TrainingPair sentenceToTrainingPair(String s) {
		TrainingPair t = new TrainingPair();
		t.x = new double[s.length()][];
		t.y = new double[s.length()][];
		for(int i=0; i < s.length()-1; i++) {
			t.x[i] = encodeCharacterToProbabilityDistribution(""+s.charAt(i));
			t.y[i] = encodeCharacterToProbabilityDistribution(""+s.charAt(i + 1));
		}
		t.x[s.length()-1] = encodeCharacterToProbabilityDistribution(" ");
		t.y[s.length()-1] = encodeCharacterToProbabilityDistribution(" ");
		return t;
	}

	private Matrix[][] sentencesToTrainingSet(String[] sentences) {

		//Matrix[input|output][time step][example (one of many in batch), --- example data ---]

		int maxSentenceLength = 0;
		for(String s : sentences) {
			maxSentenceLength = Math.max(s.length(), maxSentenceLength);
		}

		Matrix[] inputs = new Matrix[maxSentenceLength-1];
		Matrix[] outputs = new Matrix[maxSentenceLength-1];

		for(int t=0; t < maxSentenceLength-1; t++) {
			inputs[t] = new Matrix(sentences.length, inputSize);
			outputs[t] = new Matrix(sentences.length, inputSize);

			for(int batchRowIndex = 0; batchRowIndex < sentences.length; batchRowIndex++) {
				// Skip the sentences whose length is less than this index.
				if(sentences[batchRowIndex].length() < t) {
					continue;
				}
				// inputs[t][batch,x])
				inputs[t].setRow(batchRowIndex, encodeCharacterToProbabilityDistribution("" + sentences[batchRowIndex].charAt(t)));
				outputs[t].setRow(batchRowIndex, encodeCharacterToProbabilityDistribution(""+sentences[batchRowIndex].charAt(t+1)));
			}
		}

		return new Matrix[][]{inputs, outputs};
	}

	@Test
	public void run() {
		// Load data.
		ArrayList <String> sentences = new ArrayList<>();
		try {
			//Scanner scanner = new Scanner(new File("SICK_train.txt"));
			//Scanner scanner = new Scanner(new File("sentences.txt"));
			Scanner scanner = new Scanner(new File("tiny-shakespeare.txt"));
			while(scanner.hasNext()) {
				/*
				String line = scanner.nextLine();
				String[] chunks = line.split("\t");
				sentences.add(chunks[1]);
				*/
				String sentence = scanner.nextLine();
				if(sentence.length() > 5) {
					sentences.add(sentence);
				}
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		// Build the model.
		LSTM lstm = new LSTM(inputSize, hiddenSize);

		// For each batch,
		final int REPORT_INTERVAL = 10;
		double learningRate = 0.1;

		int iteration = 0;
		while(learningRate > 0 && iteration < 10000000) {
			iteration += 1;

			// Sample a random set of sentences.
			String[] batch = new String[batchSize];
			for(int i=0; i < batchSize; i++) {
				//batch[i] = sentences.get(random.nextInt(sentences.size()));
				batch[i] = "Hand Ana a banana, please.";
			}

			//String sent = sentences.get(random.nextInt(sentences.size()));
			//sent = sent.replace('~', ' ');
			//sent = echoStringWhileLessThanMinLength(sent, numTrainingStepsUnrolled);
			//TrainingPair p = sentencesToTrainingSet(sent);
			Matrix[][] trainingData = sentencesToTrainingSet(batch);

			// Apply the training and, every N iterations, get a loss report.
			lstm.unrollAndTrain(trainingData[0], trainingData[1], numSteps, learningRate);

			if(iteration % REPORT_INTERVAL == 0) {
				// Sample output.
				LSTM runner = lstm.makeRunOnlyLSTM();
				//Matrix start = new Matrix(1, inputSize, encodeCharacterToProbabilityDistribution("b"));
				Matrix start = new Matrix(1, inputSize);
				Matrix[] pred = runner.generate(start, 20);
				for (Matrix m : pred) {
					System.out.print(selectCharacterFromProbabilityDistribution(m.getRow(0)));
				}
				System.out.println();
			}
		}
	}

	private void capGradientAtL1Norm(Matrix grad, float cap) {
		float accumulator = 0f;
		for(int i=0; i < grad.data.length; i++) {
			accumulator += Math.abs(grad.data[i]);
		}
		final float finalAccumulator = accumulator;
		if(accumulator > cap) {
			grad.elementOp_i(w -> w/finalAccumulator);
		}
	}

	class TrainingPair {
		public double[][] x;
		public double[][] y;
	}
}
