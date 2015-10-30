import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.networks.NeuralNetwork;
import com.josephcatrambone.aij.trainers.BackpropTrainer;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Created by Jo on 8/15/2015.
 */
public class TestNeuralNetworkTrainer {

	@Test
	public void testXOR() {
		NeuralNetwork nn = null;

		Matrix x = new Matrix(4, 2, 0.0);
		x.setRow(0, new double[]{0, 0});
		x.setRow(1, new double[]{0, 1});
		x.setRow(2, new double[]{1, 0});
		x.setRow(3, new double[]{1, 1});

		Matrix y = new Matrix(4, 1, 0.0);
		y.set(0, 0, 0.0);
		y.set(1, 0, 1.0);
		y.set(2, 0, 1.0);
		y.set(3, 0, 0.0);

		BackpropTrainer trainer = new BackpropTrainer();
		trainer.momentum = 0.0;
		trainer.learningRate = 0.1;
		trainer.batchSize = 10;
		trainer.maxIterations = 10000;
		trainer.earlyStopError = 0.0;

		// Test XOR
		// Run
		nn = new NeuralNetwork(new int[]{2, 3, 1}, new String[]{"tanh", "tanh", "tanh"});
		trainer.train(nn, x, y, null);

		Matrix predictions = nn.predict(x);
		assertTrue(predictions.get(0, 0) < 0.5);
		assertTrue(predictions.get(1, 0) > 0.5);
		assertTrue(predictions.get(2, 0) > 0.5);
		assertTrue(predictions.get(3, 0) < 0.5);

		nn = new NeuralNetwork(new int[]{2, 3, 1}, new String[]{"logistic", "logistic", "logistic"});
		trainer.train(nn, x, y, null);
		assertTrue(predictions.get(0, 0) < 0.5);
		assertTrue(predictions.get(1, 0) > 0.5);
		assertTrue(predictions.get(2, 0) > 0.5);
		assertTrue(predictions.get(3, 0) < 0.5);

		nn = new NeuralNetwork(new int[]{2, 3, 1}, new String[]{"linear", "logistic", "linear"});
		trainer.train(nn, x, y, null);
		assertTrue(predictions.get(0, 0) < 0.5);
		assertTrue(predictions.get(1, 0) > 0.5);
		assertTrue(predictions.get(2, 0) > 0.5);
		assertTrue(predictions.get(3, 0) < 0.5);
	}

	@Test
	public void testWord2Vec() {
		final int WINDOW_SIZE = 5;
		final int HIDDEN_LAYER_SIZE = 5;
		String paragraph =
			"i have a cat and a dog\n" +
			"i own a computer\n" +
			"my cat chases my dog\n" +
			"i enjoy machine learning\n" +
			"my computer runs machine learning problems\n" +
			"my pets are a cat and a dog\n" +
			"this machine learning problem is expecting unique words\n" +
			"i use github to store my machine learning code\n" +
			"i hope this code will produce good results\n" +
			"this test problem is fun and boring at the same time\n" +
			"my cat and dog are fun\n" +
			"my cat and dog are named cat and dog\n" +
			"i am bad at names\n";

		String[] words = paragraph.split("\\s");
		String[] sentences = paragraph.split("\\n");

		System.out.println("Words array: " + java.util.Arrays.toString(words));
		System.out.println("Sentences array: " + java.util.Arrays.toString(sentences));

		// We have to be able to map from words to our semantic vector.
		// Build a mapping without stemming or chunking.
		List<String> indexToWord = new ArrayList<>();
		Map<String,Integer> wordToIndex = new HashMap<>();

		for(String w : words) {
			// Expensive unique lookup, but this is just a test.
			// If the word is in the list already, don't bother.
			int foundAt = -1;
			for(int i=0; i < indexToWord.size() && foundAt == -1; i++) {
				if(indexToWord.get(i).equals(w)) {
					foundAt = i;
				}
			}
			// Not in the list so far, so add it to the end and keep the index so we can find it again.
			if(foundAt == -1) {
				indexToWord.add(w);
				wordToIndex.put(w, indexToWord.size()-1);
			}
		}

		// Make our training examples.
		// This is lazy. We have a sliding window which goes over sentences, so I'm just assuming there will be at most #sents * 8.
		int exampleIndex = 0;
		Matrix examples = new Matrix(sentences.length*8, indexToWord.size(), 0.0);
		for(String sentence : sentences) {
			//examples.setRow(0, new double[]{0, 0});
			String[] sentenceWords = sentence.split("\\s");
			for(int window = 0; window < sentenceWords.length-WINDOW_SIZE; window++) {
				for(int j=0; j < WINDOW_SIZE; j++) {
					String w = sentenceWords[window+j];
					examples.set(exampleIndex, wordToIndex.get(w), 1.0);
				}
				exampleIndex++; // Each window of three words is an example.
			}
		}

		BackpropTrainer trainer = new BackpropTrainer();
		trainer.momentum = 0.0;
		trainer.learningRate = 0.02;
		trainer.batchSize = 10;
		trainer.maxIterations = 10000;
		trainer.earlyStopError = 0.0;

		NeuralNetwork nn = new NeuralNetwork(new int[]{indexToWord.size(), HIDDEN_LAYER_SIZE, indexToWord.size()}, new String[]{"linear", "logistic", "linear"});
		trainer.train(nn, examples, examples, null);

		// Test

		// How similar are dog and cat?
		// More similar than dog and computer I hope.
		Matrix input = new Matrix(1, indexToWord.size());

		input.set(0, wordToIndex.get("cat"), 1.0);
		Matrix catActivation = nn.forwardPropagate(input)[1];
		input.elementMultiply_i(0); // Reset

		input.set(0, wordToIndex.get("dog"), 1.0);
		Matrix dogActivation = nn.forwardPropagate(input)[1];
		input.elementMultiply_i(0); // Reset

		input.set(0, wordToIndex.get("computer"), 1.0);
		Matrix computerActivation = nn.forwardPropagate(input)[1];
		input.elementMultiply_i(0); // Reset

		Matrix catDogDiff = catActivation.subtract(dogActivation);
		double catDogDistance = catDogDiff.elementMultiply(catDogDiff).sum(); // Squared distance.
		Matrix dogComputerDiff = dogActivation.subtract(computerActivation);
		double dogComputerDistance = dogComputerDiff.elementMultiply(dogComputerDiff).sum();

		System.out.println("cat vec: " + catActivation);
		System.out.println("dog vec: " + dogActivation);
		System.out.println("comp vec: " + computerActivation);
		System.out.println("catDog dist: " + catDogDistance);
		System.out.println("dogComp dist: " + dogComputerDistance);
		assertTrue(catDogDistance < dogComputerDistance);
	}
}
