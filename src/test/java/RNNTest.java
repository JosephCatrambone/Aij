import com.josephcatrambone.aij.*;

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
	public int hiddenSize = 10;
	public Matrix w_ih = new Matrix(inputSize, hiddenSize, (i,j) -> random.nextGaussian()*0.01);
	public Matrix w_hh = new Matrix(hiddenSize, hiddenSize, (i,j) -> random.nextGaussian()*0.01);
	public Matrix w_ho = new Matrix(hiddenSize, inputSize, (i,j) -> random.nextGaussian()*0.01);
	public Matrix b_h = new Matrix(1, hiddenSize);
	public Matrix b_o = new Matrix(1, inputSize);

	// Training network, used only to train the model.
	private int numTrainingStepsUnrolled = 25;
	private Graph trainingGraph = null;
	private Node[] trainingVariables;
	private Node[] trainingInputNodes;
	private Node[] trainingTargetNodes;
	private Node trainingStartHiddenNode;
	private Node trainingOutputHiddenNode;
	private Node lossNode;

	// Sampling section.
	private Graph runGraph = null;
	private Node sampleInputNode;
	private Node sampleHiddenInputNode;
	private Node sampleHiddenOutputNode;
	private Node sampleOutputNode;

	private String echoStringWhileLessThanMinLength(String s, int minLength) {
		do {
			s = s + "~" + s; // At least one duplicate.
		} while(s.length() < minLength);
		return s;
	}

	/*** makeRNN
	 * Given the previous hidden, constructs an RNN step.
	 * @param g The Graph to which we should add these nodes.
	 * @param hidden
	 * @return Returns an array with the inputNode, nextHidden, outputNode, and (if enabled) the target and loss nodes.
	 */
	public Node[] makeRNN(Graph g, VariableNode weight_ih, VariableNode weight_hh, VariableNode weight_ho, VariableNode bias_h, VariableNode bias_o, Node hidden, boolean includeLossNode) {
		Node[] result = null;
		Node inputNode = new InputNode(1, inputSize);

		// Do the actual wiring.
		Node hiddenState = new TanhNode(new AddNode(
			new TanhNode(new AddNode(new MatrixMultiplyNode(hidden, weight_hh), bias_h)),
			new TanhNode(new MatrixMultiplyNode(inputNode, weight_ih))
		));
		//Node outputState = new SigmoidNode(new AddNode(new MatrixMultiplyNode(hiddenState, weight_ho), bias_o));
		Node preactOutput = new AddNode(new MatrixMultiplyNode(hiddenState, weight_ho), bias_o);
		Node outputNode = new SigmoidNode(preactOutput); // This may need to change, depending on the loss.

		// If we want to train, we need to wire up the loss.
		if(includeLossNode) {
			result = new Node[5];

			result[0] = inputNode;
			result[1] = hiddenState;
			result[2] = outputNode;
			Node target = new InputNode(preactOutput.rows, inputSize); // Target node.
			result[3] = target;
			// ABS error
			//result[4] = new AbsNode(new SubtractNode(target, preactOutput));

			// Squared error.
			//Node diff = new SubtractNode(result[3], outputState);
			//result[4] = new PowerNode(diff, 2);

			// KL Divergence = sum y_i log y_i/out_i
			//result[4] = new MultiplyNode(target, new LogNode(new MultiplyNode(target, new InverseNode(result[2]))));

			// XEnt Loss
			// - sum{ y*log(pred) + (1-y)log(1-pred) }
			result[4] = new NegateNode(new AddNode(
				new MultiplyNode(target, new LogNode(new AddNode(new ConstantNode(1e-6, outputNode), outputNode))), // y*log(pred)
				new MultiplyNode(
					new SubtractNode(new ConstantNode(1.0, target), target), // 1-y
					new LogNode(new AddNode(new ConstantNode(1e-6, outputNode), new SubtractNode(new ConstantNode(1.0, outputNode), outputNode))) // log(1-pred)
				)
			));

			// Softmax loss.
			// If our output is not normalized (or is a ReLU), dLoss/dOut = softmax(out) - [all zeros, except a 1 for the correct label]
			// Note that we are _NOT_ activating the output layer.
			// If we activate before softmax, bad things happen.  If we use the softmax node as a 'softmax loss', bad things happen, too.
			// Instead, calculate the softmax of the practivated state by hand and back-propagate a nice loss.
			//result[4] = new SoftmaxLossNode(preactOutput, target);

			g.addNode(result[4]);
		} else {
			result = new Node[3];

			result[0] = inputNode;
			result[1] = hiddenState;
			result[2] = outputNode;
			g.addNode(result[2]);
		}

		return result;
	}

	/*** unrollAndTrain
	 * Unroll i.length loops into a graph, learn the grads, sum them, and return them.
	 * @param inputData
	 * @param outputData
	 * @return
	 */
	public Matrix[] unrollAndTrain(double[][] inputData, double[][] outputData, DoubleConsumer reportFunction) {

		HashMap <Node, Matrix> inputMap = new HashMap<>();

		if(trainingGraph == null) {
			VariableNode weight_ih = new VariableNode(w_ih);
			VariableNode weight_hh = new VariableNode(w_hh);
			VariableNode weight_ho = new VariableNode(w_ho);
			VariableNode bias_h = new VariableNode(b_h);
			VariableNode bias_o = new VariableNode(b_o);

			trainingVariables = new VariableNode[5];
			trainingVariables[0] = weight_ih;
			trainingVariables[1] = weight_hh;
			trainingVariables[2] = weight_ho;
			trainingVariables[3] = bias_h;
			trainingVariables[4] = bias_o;

			// Build the whole graph.
			trainingGraph = new Graph();
			lossNode = null;
			trainingInputNodes = new InputNode[numTrainingStepsUnrolled];
			trainingTargetNodes = new InputNode[numTrainingStepsUnrolled];
			trainingStartHiddenNode = new InputNode(1, hiddenSize);
			Node previousHidden = trainingStartHiddenNode;
			for (int i = 0; i < numTrainingStepsUnrolled; i++) {
				Node[] step = makeRNN(trainingGraph, weight_ih, weight_hh, weight_ho, bias_h, bias_o, previousHidden, true);
				trainingInputNodes[i] = step[0];
				previousHidden = step[1];
				trainingTargetNodes[i] = step[3];
				Node loss = step[4];

				if (lossNode == null) {
					lossNode = loss;
				} else {
					lossNode = new AddNode(lossNode, loss);
				}
			}
			lossNode = new RowSumNode(lossNode);
			trainingGraph.addNode(lossNode);
			trainingOutputHiddenNode = previousHidden;
		}

		Matrix[] results = new Matrix[trainingVariables.length];
		Matrix hiddenState = new Matrix(1, hiddenSize);

		int numTrainingChunks = inputData.length/numTrainingStepsUnrolled;
		for(int chunkIndex=0; chunkIndex < numTrainingChunks; chunkIndex++) {
			for (int i=0; i < numTrainingStepsUnrolled; i++) {
				inputMap.put(trainingStartHiddenNode, hiddenState);
				inputMap.put(trainingInputNodes[i], new Matrix(1, inputSize, inputData[i+(chunkIndex*numTrainingStepsUnrolled)]));
				inputMap.put(trainingTargetNodes[i], new Matrix(1, inputSize, outputData[i+(chunkIndex*numTrainingStepsUnrolled)]));
			}

			// Calculate the gradient for each of those values.
			// The funkiness with using ID is because we want to allocate at least that many spaces for gradients.
			Matrix[] fwd = trainingGraph.forward(inputMap);
			Matrix[] grads = trainingGraph.getGradient(inputMap, fwd, lossNode);

			if (reportFunction != null) {
				reportFunction.accept(fwd[lossNode.id].get(0, 0));
			}

			// Smooth/average the gradients.
			/*
			for(int i=0; i < totalGradient.length; i++) {
				totalGradient[i].elementOp_i(v -> v/(double)inputData.length);
			}
			*/
			for(int i = 0; i < trainingVariables.length; i++) {
				if(results[i] == null) {
					results[i] = grads[trainingVariables[i].id];
				} else {
					results[i].elementOp_i(grads[trainingVariables[i].id], (a, b) -> a + b);
				}
			}
			hiddenState = fwd[trainingOutputHiddenNode.id];
		}
		return results;
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

	public String sampleOutput(int seed) {
		// Build graph.
		if(runGraph == null) {
			runGraph = new Graph();

			VariableNode weight_ih = new VariableNode(w_ih);
			VariableNode weight_hh = new VariableNode(w_hh);
			VariableNode weight_ho = new VariableNode(w_ho);
			VariableNode bias_h = new VariableNode(b_h);
			VariableNode bias_o = new VariableNode(b_o);

			sampleHiddenInputNode = new InputNode(1, hiddenSize);

			Node[] step = makeRNN(runGraph, weight_ih, weight_hh, weight_ho, bias_h, bias_o, sampleHiddenInputNode, false);

			sampleInputNode = step[0];
			sampleHiddenOutputNode = step[1];
			sampleOutputNode = step[2];
		}

		// Step n steps.
		String s = "";
		Matrix lastOut = new Matrix(1, inputSize);
		Matrix lastHidden = new Matrix(1, hiddenSize);
		lastOut.data[seed] = 1.0;
		for(int i=0; i < 100; i++) {
			// Set our input states.
			HashMap<Node, Matrix> inputFeed = new HashMap<>();
			inputFeed.put(sampleInputNode, lastOut);
			inputFeed.put(sampleHiddenInputNode, lastHidden);

			// Push through.
			Matrix[] states = runGraph.forward(inputFeed);

			// Convert output to a character and use it as input.
			String nextChar = selectCharacterFromProbabilityDistribution(states[sampleOutputNode.id].data);
			if(nextChar == "\n") {
				nextChar = "[\\n]";
			}
			s += nextChar;
			lastOut = states[sampleOutputNode.id];
			lastHidden = states[sampleHiddenOutputNode.id];
		}

		// Also, save the graph since we have it.
		try(BufferedWriter fout = new BufferedWriter(new FileWriter("languageModal.dat"))) {
			fout.write(runGraph.serializeToString());
		} catch (IOException ioe) {

		}

		return s;
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
	public void train() {
		// Build the model's key pieces.

		// Grab all the sentences.
		ArrayList <String> sentences = new ArrayList<>();
		try {
			//Scanner scanner = new Scanner(new File("SICK_train.txt"));
			Scanner scanner = new Scanner(new File("sentences.txt"));
			while(scanner.hasNext()) {
				/*
				String line = scanner.nextLine();
				String[] chunks = line.split("\t");
				sentences.add(chunks[1]);
				*/
				sentences.add(scanner.nextLine());
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		// For each batch,
		final int REPORT_INTERVAL = 100;
		double learningRate = 0.5;
		Matrix mem_W_ih = new Matrix(w_ih.rows, w_ih.columns);
		Matrix mem_W_hh = new Matrix(w_hh.rows, w_hh.columns);
		Matrix mem_W_ho = new Matrix(w_ho.rows, w_ho.columns);
		Matrix mem_B_h = new Matrix(b_h.rows, b_h.columns);
		Matrix mem_B_o = new Matrix(b_o.rows, b_o.columns);

		int iteration = 0;
		while(learningRate > 0 && iteration < 10000000) {
			iteration += 1;

			String sent = sentences.get(random.nextInt(sentences.size()));
			//String sent = "Let's see how long it takes to fit this."; // About 2k iters.
			sent = sent.replace('~', ' ');
			sent = echoStringWhileLessThanMinLength(sent, numTrainingStepsUnrolled);
			TrainingPair p = sentenceToTrainingPair(sent);

			// Apply the training and, every N iterations, get a loss report.
			DoubleConsumer report = null;
			if(iteration % REPORT_INTERVAL == 0) {
				report = i -> System.out.println("Loss : " + i);
			}
			Matrix[] grads = unrollAndTrain(p.x, p.y, report);

			// Cap the gradients.
			for(Matrix m : grads) {
				capGradientAtL1Norm(m, 5.0f);
			}

			// Apply gradients with training loss.
			adaGradUpdate(mem_W_ih, grads[0], w_ih, learningRate);
			adaGradUpdate(mem_W_hh, grads[1], w_hh, learningRate);
			adaGradUpdate(mem_W_ho, grads[2], w_ho, learningRate);
			adaGradUpdate(mem_B_h, grads[3], b_h, learningRate);
			adaGradUpdate(mem_B_o, grads[4], b_o, learningRate);

			// For some reason, reassignment appears to be faster than in-place operations here.
			//b_o = b_o.elementOp(grads[4], (w, dw) -> w - dw*learningRate);

			if(iteration % REPORT_INTERVAL == 0) {
				learningRate -= Math.pow(learningRate, 4);
				//learningRate = Math.pow(1.0/Math.sqrt(iteration), 0.1);
				System.out.println("Iter: " + iteration + "\tLR: " + learningRate +"\tSample: " + sampleOutput(random.nextInt(127-32)+32));
			}
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
