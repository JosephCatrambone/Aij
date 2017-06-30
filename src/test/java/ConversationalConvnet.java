import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.Model;
import com.josephcatrambone.aij.nodes.PadCropNode;
import org.junit.Assume;
import org.junit.Test;

import java.io.*;
import java.util.*;

/**
 * Created by jcatrambone on 9/13/16.
 */
public class ConversationalConvnet {
	int sentenceLength = 256;
	char[] printableCharacters = " '.0123456789?abcdefghijklmnopqrstuvwxyz\n".toCharArray(); // MUST BE IN THIS ORDER for binsearch to work.
	// Have to populate map on start.
	Map<Character, Integer> characterToIndex = new HashMap<Character, Integer>(printableCharacters.length);
	char[] indexToCharacter = printableCharacters;

	private void populateCTI() {
		characterToIndex.clear();
		for(int i=0; i < printableCharacters.length; i++) {
			characterToIndex.put(printableCharacters[i], i);
		}
	}

	public Matrix sentenceToMatrix(String sent) {
		sent = sent.toLowerCase();
		Matrix res = new Matrix(sentenceLength, printableCharacters.length);
		for(int i=0; i < sentenceLength; i++) {
			int chrIndex = characterToIndex.get('\n');
			if(i < sent.length()) {
				chrIndex = characterToIndex.getOrDefault(sent.charAt(i), 0);
			}
			res.set(i, chrIndex, 1.0);
		}
		return res;
	}

	public String matrixToSentence(Matrix mat) {
		Random random = new Random();
		StringBuilder sb = new StringBuilder();
		// To get a column, we do mat transpose and grab a row.
		// Not as efficient, but it's easier for me.
		boolean done = false;
		for(int r=0; r < mat.rows && !done; r++) {
			double[] probs = mat.getRow(r);

			// Probabalistically select a character.
			// First calculate sum so we can normalize it.
			double min = 10;
			double max = -10;
			double sum = 0;
			for(int k=0; k < probs.length; k++) {
				min = Math.min(min, probs[k]);
				max = Math.max(max, probs[k]);
			}
			if(min == max) { max += 1.0e-6; }
			for(int k=0; k < probs.length; k++) {
				sum += (probs[k]-min)/(max-min);
			}

			// Then generate a random number.
			double energy = random.nextDouble()*sum; // From 0 to 1.
			for(int charPosition=0; charPosition < printableCharacters.length; charPosition++) {
				// The amount of energy required to get to the next spot...
				double barrierEnergy = (probs[charPosition]-min)/(max-min);
				if(energy < barrierEnergy) {
					sb.append(printableCharacters[charPosition]);
					if(printableCharacters[charPosition] == '\n') {
						done = true;
					}
					// Break from inner loop without 'break'.
					energy = 0;
					charPosition = probs.length;
				} else {
					energy -= barrierEnergy;
				}
			}
		}
		return sb.toString();
	}

	@Test
	public void trainConversationalModel() {
		populateCTI();
		Random random = new Random();

		System.out.println(matrixToSentence(sentenceToMatrix("This is a test.")));

		// Build Model
		Model m = new Model(sentenceLength, printableCharacters.length);
		// Encoder vv
		m.addConvLayer(32, 3, printableCharacters.length, 2, printableCharacters.length/2, Model.Activation.RELU); //128x64 out.
		m.addConvLayer(64, 32, 64, 16, 32, Model.Activation.TANH); // 64x1280
		int preFlattenedRows = m.getOutputNode().rows;
		int preFlattenedColumns = m.getOutputNode().columns;
		m.addFlattenLayer();
		m.addDenseLayer(128, Model.Activation.TANH);
		// Encoder output ^^

		m.addDenseLayer(512, Model.Activation.TANH); // Representation

		// Decoder vv
		m.addDenseLayer(preFlattenedColumns*preFlattenedRows, Model.Activation.TANH);
		m.addReshapeLayer(preFlattenedRows, preFlattenedColumns);
		m.addDeconvLayer(64, 32, 64, 16, 32, Model.Activation.TANH);
		m.addDeconvLayer(32, 3, printableCharacters.length, 2, printableCharacters.length/2, Model.Activation.SIGMOID);
		//m.addNode(new PadCropNode(sentenceLength, printableCharacters.length, m.getOutputNode()));
		// Decoder ^^

		System.out.println(m.getOutputNode().rows + ", " + m.getOutputNode().columns);

		// Load training data.
		String[] sentences = null;
		try{
			BufferedReader fin = new BufferedReader(new FileReader(new File("sentences.txt")));
			//sentences = fin.lines(String[]::new);
			sentences = fin.lines().toArray(size -> new String[size]);
			fin.close();
		} catch(IOException ioe) {
			Assume.assumeNoException("Unable to read sample sentence.txt", ioe);
		}

		for(int i=0; i < 10000; i++) {
			//Matrix mat = sentenceToMatrix(sentences[random.nextInt(sentences.length)]);
			Matrix mat = sentenceToMatrix("What the fuck?");
			m.fit(mat.data, mat.data, 0.001, Model.Loss.ABS);
			System.out.println(matrixToSentence(new Matrix(printableCharacters.length, sentenceLength, m.predict(mat.data))));
		}
	}
}
