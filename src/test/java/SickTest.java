import com.josephcatrambone.aij.*;

import com.josephcatrambone.aij.nodes.*;
import org.junit.Test;

import java.io.*;
import java.util.*;

/**
 * Created by jcatrambone on 2/6/17.
 */
public class SICKTest {

	public Random random = new Random();
	public int inputSize = 127;
	public int hiddenSize = 256;
	public Matrix w_i = new Matrix(inputSize, hiddenSize, (i,j) -> random.nextGaussian()*0.1);
	public Matrix w_f = new Matrix(inputSize, hiddenSize, (i,j) -> random.nextGaussian()*0.1);
	public Matrix w_c = new Matrix(inputSize, hiddenSize, (i,j) -> random.nextGaussian()*0.1);
	public Matrix w_o = new Matrix(hiddenSize, inputSize, (i,j) -> random.nextGaussian()*0.1);
	public Matrix u_i = new Matrix(hiddenSize, hiddenSize, (i,j) -> random.nextGaussian()*0.1);
	public Matrix u_f = new Matrix(hiddenSize, hiddenSize, (i,j) -> random.nextGaussian()*0.1);
	public Matrix u_c = new Matrix(hiddenSize, hiddenSize, (i,j) -> random.nextGaussian()*0.1);
	public Matrix u_o = new Matrix(hiddenSize, inputSize, (i,j) -> random.nextGaussian()*0.1);
	public Matrix b_i = new Matrix(1, hiddenSize);
	public Matrix b_f = new Matrix(1, hiddenSize);
	public Matrix b_c = new Matrix(1, hiddenSize);
	public Matrix b_o = new Matrix(1, inputSize);

	/* LSTM formulation
	i_t = sigmoid(W_i*x_t + U_i*h_t-1 + b_i)
	f_t = sigmoid(W_f*x_t + U_f*h_t-1 + b_f)
	~c_t = tanh(W_c*x_t + U_c*h_t-1 + b_c)
	c_t = i_t (dot) ~c_t + f_t (dot)c_t-1
	o_t = sigmoid(W_o*x_t + U_o*h_t-1 + b_o)
	h_t = o_t (dot) tanh(c_t)
	*/

	/*** unrollAndTrain
	 * Unroll i.length loops into a graph, learn the grads, sum them, and return them.
	 * @param inputData
	 * @param outputData
	 * @return
	 */
	public Matrix[] unrollAndTrain(double[][] inputData, double[][] outputData) {

		HashMap <Node, Matrix> inputMap = new HashMap<>();

		VariableNode n_w_i = new VariableNode(w_i); // We create these from the public matrices because we want to preserve the values between executions.
		VariableNode n_w_f = new VariableNode(w_f); 
		VariableNode n_w_c = new VariableNode(w_c); 
		VariableNode n_w_o = new VariableNode(w_o); 
		VariableNode n_u_i = new VariableNode(u_i);
		VariableNode n_u_f = new VariableNode(u_f); 
		VariableNode n_u_c = new VariableNode(u_c); 
		VariableNode n_u_o = new VariableNode(u_o); 
		VariableNode n_b_i = new VariableNode(b_i);
		VariableNode n_b_f = new VariableNode(b_f); 
		VariableNode n_b_c = new VariableNode(b_c); 
		VariableNode n_b_o = new VariableNode(b_o); 

		// Build the whole graph.
		Graph g = new Graph();
		Node lossAccumulator = null;
		Node previousHidden = new InputNode(1, hiddenSize);
		Node previousCell = new InputNode(1, hiddenSize);
		inputMap.put(previousHidden, new Matrix(1, hiddenSize));
		inputMap.put(previousCell, new Matrix(1, hiddenSize));
		for(int i=0; i < inputData.length; i++) {
			Node inputNode = new InputNode(1, inputSize);
			inputMap.put(inputNode, new Matrix(1, inputSize, inputData[i]));
			// h = tanh(prev_in + h_prev)
			Node newInput = new SigmoidNode(new AddNode(
				new AddNode(
					new MatrixMultiplyNode(inputNode, n_w_i),
					new MatrixMultiplyNode(previousHidden, n_u_i)
				),
				n_b_i
			));
			
			Node newForgetGate = new SigmoidNode(new AddNode(
				new AddNode(
					new MatrixMultiplyNode(inputNode, n_w_f),
					new MatrixMultiplyNode(previousHidden, n_u_f)
				),
				n_b_f
			));

			Node candidateCellState = new TanhNode(new AddNode(
				new AddNode(
					new MatrixMultiplyNode(inputNode, n_w_c),
					new MatrixMultiplyNode(previousHidden, n_u_c)
				),
				n_b_c
			));

			Node newOutputState = new SigmoidNode(new AddNode(
				new AddNode(
					new MatrixMultiplyNode(inputNode, n_w_o),
					new MatrixMultiplyNode(previousHidden, n_u_o)
				),
				n_b_o
			));

			Node newCellState = new AddNode(
				new MultiplyNode(newInput, candidateCellState),
				new MultiplyNode(newForgetGate, previousCell)
			);

			hiddenState = new MultiplyNode(newOutputState, new TanhNode(newCellState));

			previousHidden = hiddenState;
			previousCell = newCellState;

			Node outputState = newOutputState; // SoftmaxRowNode(new AddNode(new MatrixMultiplyNode(hiddenState, weight_ho), bias_o));
			Node target = new InputNode(1, inputSize);
			inputMap.put(target, new Matrix(1, inputSize, outputData[i]));
			Node diff = new SubtractNode(target, outputState);
			Node loss = new PowerNode(diff, 2.0);
			if(lossAccumulator == null) {
				lossAccumulator = loss;
			} else {
				lossAccumulator = new AddNode(lossAccumulator, loss);
			}
		}
		lossAccumulator = new RowSumNode(lossAccumulator);
		g.addNode(lossAccumulator);

		// Calculate the gradient for each of those values.
		// The funkiness with using ID is because we want to allocate at least that many spaces for gradients.
		Matrix[] fwd = g.forward(inputMap);
		Matrix[] grads = g.getGradient(inputMap, fwd, lossAccumulator);
		//System.out.println("Loss: " + fwd[lossAccumulator.id]);

		// Smooth/average the gradients.
		/*
		for(int i=0; i < totalGradient.length; i++) {
			totalGradient[i].elementOp_i(v -> v/(double)inputData.length);
		}
		*/
		return new Matrix[]{
			grads[n_w_i.id],
			grads[n_w_f.id],
			grads[n_w_c.id],
			grads[n_w_o.id],
			grads[n_u_i.id],
			grads[n_u_f.id],
			grads[n_u_c.id],
			grads[n_u_o.id],
			grads[n_b_i.id],
			grads[n_b_f.id],
			grads[n_b_c.id],
			grads[n_b_o.id]
		};
	}

	public double[] encodeCharacterToProbabilityDistribution(String s) {
		double[] enc = new double[127];
		enc[(int)s.charAt(0)] = 1.0;
		return enc;
	}

	public String selectCharacterFromProbabilityDistribution(double[] probs) {
		// First calculate sum so we can normalize it.
		double min = 0;
		double max = 1;
		double sum = 0;
		/*
		for(int i=32; i < probs.length; i++) {
			min = Math.min(min, probs[i]);
			max = Math.max(max, probs[i]);
		}
		if(min == max) { max += 1.0e-6; }
		*/
		for(int i=32; i < probs.length; i++) {
			sum += (probs[i]-min)/(max-min);
		}

		// Then generate a random number.
		double energy = random.nextDouble()*sum; // From 0 to 1.
		for(int charPosition=32; charPosition < probs.length; charPosition++) {
			// The amount of energy required to get to the next spot...
			double barrierEnergy = (probs[charPosition]-min)/(max-min);
			if(energy < barrierEnergy) {
				return ""+(char)(charPosition);
			} else {
				energy -= barrierEnergy;
			}
		}
		return "?";
	}

	public String sampleOutput(int seed) {
		// Build graph.
		Graph g = new Graph();
		Node inputNode = new InputNode(1, inputSize);

		VariableNode n_w_i = new VariableNode(w_i); // We create these from the public matrices because we want to preserve the values between executions.
		VariableNode n_w_f = new VariableNode(w_f); 
		VariableNode n_w_c = new VariableNode(w_c); 
		VariableNode n_w_o = new VariableNode(w_o); 
		VariableNode n_u_i = new VariableNode(u_i);
		VariableNode n_u_f = new VariableNode(u_f); 
		VariableNode n_u_c = new VariableNode(u_c); 
		VariableNode n_u_o = new VariableNode(u_o); 
		VariableNode n_b_i = new VariableNode(b_i);
		VariableNode n_b_f = new VariableNode(b_f); 
		VariableNode n_b_c = new VariableNode(b_c); 
		VariableNode n_b_o = new VariableNode(b_o); 

		Node previousHidden = new InputNode(1, hiddenSize);
		Node hiddenState = new AddNode(
				new TanhNode(new AddNode(new MatrixMultiplyNode(previousHidden, weight_hh), bias_h)),
				new TanhNode(new MatrixMultiplyNode(inputNode, weight_ih))
		);
		Node outputNode = new SoftmaxRowNode(new AddNode(new MatrixMultiplyNode(hiddenState, weight_ho), bias_o));
		g.addNode(outputNode);

		// Step n steps.
		String s = "";
		Matrix lastOut = new Matrix(1, inputSize);
		Matrix lastHidden = new Matrix(1, hiddenSize);
		lastOut.data[seed] = 1.0;
		for(int i=0; i < 100; i++) {
			// Set our input states.
			HashMap<Node, Matrix> inputFeed = new HashMap<>();
			inputFeed.put(inputNode, lastOut);
			inputFeed.put(previousHidden, lastHidden);

			// Push through.
			Matrix[] states = g.forward(inputFeed);

			// Convert output to a character and use it as input.
			s += selectCharacterFromProbabilityDistribution(states[outputNode.id].data);
			lastOut = states[outputNode.id];
			lastHidden = states[hiddenState.id];
		}

		// Also, save the graph since we have it.
		try(BufferedWriter fout = new BufferedWriter(new FileWriter("languageModal.dat"))) {
			fout.write(g.serializeToString());
		} catch (IOException ioe) {

		}

		return s;
	}

	public TrainingPair sentenceToTrainingPair(String s) {
		TrainingPair t = new TrainingPair();
		t.x = new double[s.length()-1][];
		t.y = new double[s.length()-1][];
		for(int i=0; i < s.length()-1; i++) {
			t.x[i] = encodeCharacterToProbabilityDistribution(""+s.charAt(i));
			t.y[i] = encodeCharacterToProbabilityDistribution(""+s.charAt(i + 1));
		}
		return t;
	}

	@Test
	public void trainSick() {
		// Build the model's key pieces.

		// Grab all the sentences.
		ArrayList <String> sentences = new ArrayList<>();
		try {
			Scanner scanner = new Scanner(new File("SICK_train.txt"));
			while(scanner.hasNext()) {
				String line = scanner.nextLine();
				String[] chunks = line.split("\t");
				sentences.add(chunks[1]);
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		// For each batch,
		double learningRate = 0.1;
		int iteration = 0;
		while(learningRate > 0) {
			iteration += 1;

			//String sent = sentences.get(random.nextInt(sentences.size())) + ".";
			String sent = "Free beer";
			TrainingPair p = sentenceToTrainingPair(sent);
			Matrix[] grads = unrollAndTrain(p.x, p.y);

			w_i = w_hh.elementOp(grads[0], (w, dw) -> w - dw*learningRate);
			w_f = w_hh.elementOp(grads[1], (w, dw) -> w - dw*learningRate);
			w_c = w_hh.elementOp(grads[2], (w, dw) -> w - dw*learningRate);
			w_o = w_hh.elementOp(grads[3], (w, dw) -> w - dw*learningRate);
			u_i = w_hh.elementOp(grads[4], (w, dw) -> w - dw*learningRate);
			u_f = w_hh.elementOp(grads[5], (w, dw) -> w - dw*learningRate);
			u_c = w_hh.elementOp(grads[6], (w, dw) -> w - dw*learningRate);
			u_o = w_hh.elementOp(grads[7], (w, dw) -> w - dw*learningRate);
			b_i = w_hh.elementOp(grads[8], (w, dw) -> w - dw*learningRate);
			b_f = w_hh.elementOp(grads[9], (w, dw) -> w - dw*learningRate);
			b_c = w_hh.elementOp(grads[10], (w, dw) -> w - dw*learningRate);
			b_o = w_hh.elementOp(grads[11], (w, dw) -> w - dw*learningRate);
			//b_o.elementOp_i(grads[4], (w, dw) -> w - dw*learningRate);

			if(iteration % 100 == 0) {
				int seed = (int)'A' + random.nextInt(26);
				System.out.println("Sample: " + (char)seed + sampleOutput(seed));
			}
		}
	}

	class TrainingPair {
		public double[][] x;
		public double[][] y;
	}
}

