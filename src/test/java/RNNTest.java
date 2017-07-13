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
	public int hiddenSize = 16;

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
					sentences.add(scanner.nextLine());
				}
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		// Build the model.
		LSTM lstm = new LSTM(inputSize, hiddenSize);

		// For each batch,
		final int REPORT_INTERVAL = 100;
		double learningRate = 0.05;

		int iteration = 0;
		while(learningRate > 0 && iteration < 10000000) {
			iteration += 1;

			String sent = sentences.get(random.nextInt(sentences.size()));
			//String sent = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"; // About 2k iters.
			sent = sent.replace('~', ' ');
			//sent = echoStringWhileLessThanMinLength(sent, numTrainingStepsUnrolled);
			TrainingPair p = sentenceToTrainingPair(sent);

			// Apply the training and, every N iterations, get a loss report.

		}
	}

	private void adaGradUpdate(Matrix mem, Matrix grad, Matrix w, double learningRate) {
		mem.elementOp_i(grad, (m, dp) -> m + dp*dp); //mem += dparam * dparam
		Matrix dw = grad.elementOp(mem, (dp, m) -> -learningRate*dp/Math.sqrt(1e-8+m));  // param += -learning_rate * dparam / np.sqrt(mem + 1e-8)
		w.elementOp_i(dw, (p, dp) -> p+dp);  // NOTE: Negative in the learningRate multiplication above.
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
