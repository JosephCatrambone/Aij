import com.josephcatrambone.aij.CPUGraph;
import com.josephcatrambone.aij.Dimension;
import com.josephcatrambone.aij.Graph;
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

	public void autoencoderTest() {
		final int WINDOW_SIZE = 5; // Should be odd.
		final int ITERATIONS = 10000;
		final int REPRESENTATION_SIZE = 500;
		final float WEIGHT_SCALE = 0.1f;
		Random random = new Random();

		// Build a dictionary.
		System.out.println("Loading sentences and building dictionary.");
		ArrayList<String> sentences = new ArrayList<>();
		HashMap<String, Integer> wordToIndex = new HashMap<>();
		ArrayList<String> indexToWord = new ArrayList<>();
		//InputStream fin = this.getClass().getResourceAsStream();
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
		int b = g.addInput("bias", new Dimension(REPRESENTATION_SIZE, 1));

		int h_unbiased = g.addNode("h_unbiased", Graph.NODE_OPERATION.MATRIXMULTIPLY, new int[]{x, w1});
		int h_biased = g.addNode("h_biased", Graph.NODE_OPERATION.ADD, new int[]{h_unbiased, b});
		int h = g.addNode("h", Graph.NODE_OPERATION.TANH, new int[]{h_biased});
		int out = g.addNode("out_unbiased", Graph.NODE_OPERATION.MATRIXMULTIPLY, new int[]{h, w2});
		int error = g.addNode("error", Graph.NODE_OPERATION.SUBTRACT, new int[]{y, out});
		int abs_error = g.addNode("abs_error", Graph.NODE_OPERATION.ABS, new int[]{error});

		// Assign our starting parameters.
		System.out.println("Initializing parameters.");
		inputs.put(x, new float[]{});
		inputs.put(y, new float[]{});
		inputs.put(w1, new float[REPRESENTATION_SIZE*indexToWord.size()]);
		inputs.put(w2, new float[REPRESENTATION_SIZE*indexToWord.size()]);
		inputs.put(b, new float[REPRESENTATION_SIZE]);

		// Init weights.
		for(int i=0; i < inputs.get(w1).length; i++) {
			inputs.get(w1)[i] = (float)random.nextGaussian()*WEIGHT_SCALE;
		}
		for(int i=0; i < inputs.get(w2).length; i++) {
			inputs.get(w2)[i] = (float)random.nextGaussian()*WEIGHT_SCALE;
		}

		// Train with simple backpropagation, no momentum.
		System.out.println("Training.");
		float learningRate = 0.01f;
		for(int i=0; i < ITERATIONS; i++) {
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
				for(int k=j; k < WINDOW_SIZE; k++) {
					// Copy these words into the example.
					int index = wordToIndex.getOrDefault(cleanWord(sent[k]), 0);
					if(k == midpoint) {
						label[index] = 1.0f;
					} else {
						example[index] = 1.0f;
					}
				}

				// Do training.
				inputs.replace(x, example);
				inputs.replace(y, label);
				// When we call replace on the weights, that ends the batch.  I think.  Unless it's a reference.
				float[][] dw = g.getGradient(inputs, abs_error);
				// Apply weight update.
				for(int k=0; k < dw[w1].length; k++) {
					inputs.get(w1)[k] += dw[w1][k]*learningRate;
				}
				for(int k=0; k < dw[w2].length; k++) {
					inputs.get(w2)[k] += dw[w2][k]*learningRate;
				}
			}

			// Generate a sample analogy.
			// Pick two random words.  Find the transform between their representations.  Pick another word.  Apply.
			int wordA = random.nextInt(wordToIndex.size());
			int wordB = random.nextInt(wordToIndex.size());
			int wordC = random.nextInt(wordToIndex.size());
			System.out.print("'" + indexToWord.get(wordA) + "' is to '" + indexToWord.get(wordB) + "' as '" + indexToWord.get(wordC) + "' is to '");
			inputs.put(x, new float[wordToIndex.size()]);
			inputs.get(x)[wordA] = 1.0f;
			float[] wordARep = g.getOutput(inputs, h);
			inputs.get(x)[wordA] = 0.0f;
			inputs.get(x)[wordB] = 1.0f;
			float[] wordBRep = g.getOutput(inputs, h);
			inputs.get(x)[wordB] = 0.0f;
			inputs.get(x)[wordC] = 1.0f;
			float[] wordCRep = g.getOutput(inputs, h);

			// Can reuse our g for encoding, but easier to build a new graph for decoding.
			Graph decoder = new CPUGraph();
			int decoder_in = decoder.addInput("input", new Dimension(REPRESENTATION_SIZE, 1));
			int decoder_w = decoder.addInput("weight", new Dimension(indexToWord.size(), REPRESENTATION_SIZE));
			int decoder_o = decoder.addNode("out", Graph.NODE_OPERATION.MATRIXMULTIPLY, new int[]{decoder_in, decoder_w});
			HashMap<Integer, float[]> decoderInputs = new HashMap<>();
			decoderInputs.put(decoder_w, inputs.get(w2)); // Copy weights from other network.
			decoderInputs.put(decoder_in, new float[wordARep.length]);
			for(int j=0; j < wordARep.length; j++) {
				decoderInputs.get(decoder_in)[j] = wordCRep[j] + (wordBRep[j] - wordARep[j]);
			}
			float[] wordDOneHot = decoder.getOutput(decoderInputs, decoder_o);
			// Pick the highest values.
			int top1 = 0, top2 = 1, top3 = 2;
			for(int k=0; k < wordDOneHot.length; k++) {
				if(wordDOneHot[k] > top1) { top3 = top2; top2 = top1; top1 = k; }
				else if(wordDOneHot[k] > top2) { top3 = top2; top2 = k; }
				else if(wordDOneHot[k] > top3) { top3 = k; }
			}
			System.out.println(indexToWord.get(top1) + "/" + indexToWord.get(top2) + "/" + indexToWord.get(top3));
		}
	}
}

