import com.josephcatrambone.aij.*;
import org.junit.Test;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

/**
 * Created by jcatrambone on 9/22/16.
 */
public class NetworkTest {
	public String cleanWord(String w) {
		String s = "";
		for(char c : w.toLowerCase().toCharArray()) {
			if(Character.isAlphabetic(c)) {
				s += c;
			}
		}
		return s;
	}

	@Test
	public void autoencoderTest() {
		final int WINDOW_SIZE = 5; // Should be odd.
		final int ITERATIONS = 10000;
		final int REPRESENTATION_SIZE = 100;
		final float WEIGHT_SCALE = 0.1f;
		final float LEARNING_RATE_DECAY = 0.9999f;
		final int REPORT_INTERVAL = 10;
		Random random = new Random();

		// Build a dictionary.
		System.out.println("Loading sentences and building dictionary.");
		ArrayList<String> sentences = new ArrayList<>();
		HashMap<String, Integer> wordToIndex = new HashMap<>();
		HashMap<String, Integer> frequencyCount = new HashMap<>();
		ArrayList<String> indexToWord = new ArrayList<>();
		//InputStream fin = this.getClass().getResourceAsStream();
		wordToIndex.put("", 0);
		indexToWord.add("");
		Scanner scanner = null;
		try {
			scanner = new Scanner(new File("sentences.txt"));
		} catch(FileNotFoundException fnfe) {
			org.junit.Assert.assertTrue(false);
		}
		while(scanner.hasNext()) {
			String line = scanner.nextLine();
			sentences.add(line);
			String[] words = line.split("\\s");
			for(String w : words) {
				String cleanWord = cleanWord(w);
				if(!wordToIndex.containsKey(cleanWord)) {
					wordToIndex.put(cleanWord, indexToWord.size());
					indexToWord.add(cleanWord);
				}
				frequencyCount.replace(cleanWord, frequencyCount.getOrDefault(cleanWord, 0)+1);
			}
		}
		System.out.println("Found " + wordToIndex.size() + " different words and " + sentences.size() + " sentences.");

		// Build an autoencoder.
		System.out.println("Initializing computational graph.");
		Graph g = new CPUGraph();
		HashMap<Integer, float[]> inputs = new HashMap<>();

		int x = g.addInput("input", new Dimension(indexToWord.size(), 1));
		int y = g.addInput("target", new Dimension(indexToWord.size(), 1));
		int w1 = g.addInput("weight1", new Dimension(REPRESENTATION_SIZE, indexToWord.size()));
		int w2 = g.addInput("weight2", new Dimension(indexToWord.size(), REPRESENTATION_SIZE));
		int b1 = g.addInput("bias1", new Dimension(REPRESENTATION_SIZE, 1));
		int b2 = g.addInput("bias2", new Dimension(indexToWord.size(), 1));

		int h_unbiased = g.addNode("h_unbiased", Graph.NODE_OPERATION.MATRIXMULTIPLY, new int[]{x, w1});
		int h_biased = g.addNode("h_biased", Graph.NODE_OPERATION.ADD, new int[]{h_unbiased, b1});
		int h = g.addNode("h", Graph.NODE_OPERATION.TANH, new int[]{h_biased});
		int out_unbiased = g.addNode("out_unbiased", Graph.NODE_OPERATION.MATRIXMULTIPLY, new int[]{h, w2});
		int out_biased = g.addNode("out_biased", Graph.NODE_OPERATION.ADD, new int[]{out_unbiased, b2});
		int out = g.addNode("out", Graph.NODE_OPERATION.TANH, new int[]{out_biased});
		int error = g.addNode("error", Graph.NODE_OPERATION.SUBTRACT, new int[]{y, out});
		int abs_error = g.addNode("abs_error", Graph.NODE_OPERATION.ABS, new int[]{error});

		// Assign our starting parameters.
		System.out.println("Initializing parameters.");
		inputs.put(x, new float[]{});
		inputs.put(y, new float[]{});
		inputs.put(w1, new float[REPRESENTATION_SIZE*indexToWord.size()]);
		inputs.put(w2, new float[REPRESENTATION_SIZE*indexToWord.size()]);
		inputs.put(b1, new float[REPRESENTATION_SIZE]);
		inputs.put(b2, new float[indexToWord.size()]);

		// Init weights.
		for(int i=0; i < inputs.get(w1).length; i++) {
			inputs.get(w1)[i] = (float)random.nextGaussian()*WEIGHT_SCALE;
		}
		for(int i=0; i < inputs.get(w2).length; i++) {
			inputs.get(w2)[i] = (float)random.nextGaussian()*WEIGHT_SCALE;
		}

		// Train with simple backpropagation, no momentum.
		System.out.println("Training.");
		float learningRate = 0.001f;
		long averageTimePerIteration = 0;
		long startTime = System.currentTimeMillis();
		for(int i=1; i < ITERATIONS; i++) {
			float deltaSum = 0.0f;
			System.out.println("Iteration " + i);
			// Select random sentence and split.  Make sure we have a minimum window size.
			String[] sent = sentences.get(random.nextInt(sentences.size())).split("\\s"); // Split into words.
			while(sent.length < WINDOW_SIZE) {
				sent = sentences.get(random.nextInt(sentences.size())).split("\\s");
			}

			// Scan across with windows, generating examples.
			for(int j=0; j < sent.length-WINDOW_SIZE; j++) {
				int midpoint = j+(WINDOW_SIZE/2);
				// Our 'batch size' is one for now.
				float[] example = new float[indexToWord.size()];
				float[] label = new float[indexToWord.size()];

				//for(String w : Arrays.copyOfRange(sent, j, j+WINDOW_SIZE)) {
				boolean validWindow = true;
				for(int k=j; k < j+WINDOW_SIZE && validWindow; k++) {
					// Copy these words into the example.
					int index = wordToIndex.getOrDefault(cleanWord(sent[k]), -1);
					if(index == -1) { validWindow = false; continue; }
					if(k == midpoint) {
						label[index] = 1.0f;
					} else {
						example[index] = 1.0f;
					}
				}
				if(!validWindow) { continue; } // Jump over the rest of these words.

				// Do training.
				inputs.replace(x, example);
				inputs.replace(y, label);
				// When we call replace on the weights, that ends the batch.  I think.  Unless it's a reference.
				float[][] dw = g.getGradient(inputs, abs_error);
				// Apply weight update.
				for(int k=0; k < dw[w1].length; k++) {
					inputs.get(w1)[k] += dw[w1][k]*learningRate;
					deltaSum += dw[w1][k];
				}
				for(int k=0; k < dw[w2].length; k++) {
					inputs.get(w2)[k] += dw[w2][k]*learningRate;
					deltaSum += dw[w2][k];
				}
				for(int k=0; k < dw[b1].length; k++) {
					inputs.get(b1)[k] += dw[b1][k]*learningRate;
				}
				for(int k=0; k < dw[b2].length; k++) {
					inputs.get(b2)[k] += dw[b2][k]*learningRate;
				}
			}
			learningRate *= LEARNING_RATE_DECAY;

			// Report time.
			long timeDelta = System.currentTimeMillis() - startTime;
			averageTimePerIteration =  (long)(0.9*averageTimePerIteration + 0.1*timeDelta);
			System.out.println("Last iteration: " + timeDelta + "ms.  Average of last ten: " + averageTimePerIteration);
			System.out.println("Delta sum: " + deltaSum);

			if(i % REPORT_INTERVAL == 0) {
				float[] expected = inputs.get(y);
				float[] wordDOneHot = g.getOutput(inputs, out);
				// Pick the highest values.
				int topK = 0;
				int top1 = 0, top2 = 1, top3 = 2;
				for (int k = 0; k < wordDOneHot.length; k++) {
					if (wordDOneHot[k] > wordDOneHot[top1]) {
						top3 = top2;
						top2 = top1;
						top1 = k;
					} else if (wordDOneHot[k] > wordDOneHot[top2]) {
						top3 = top2;
						top2 = k;
					} else if (wordDOneHot[k] > wordDOneHot[top3]) {
						top3 = k;
					}

					if (expected[k] > expected[topK]) {
						topK = k;
					}
				}
				System.out.println("Expected: " + indexToWord.get(topK));
				System.out.println(indexToWord.get(top1) + "/" + indexToWord.get(top2) + "/" + indexToWord.get(top3));
			}

			startTime = System.currentTimeMillis();
		}
	}
}

