package com.josephcatrambone.aij.models;

import com.josephcatrambone.aij.Graph;
import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.nodes.*;
import com.josephcatrambone.aij.optimizers.Optimizer;
import com.josephcatrambone.aij.optimizers.SGD;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

// TODO: Need XENT instead of MSE or ABS.
// TODO: Need Softmax output.

/**
 * Created by jcatrambone on 7/12/17.
 */
public class LSTM {
	private Random random = new Random();

	int inputSize = -1;
	int hiddenSize = -1;
	VariableNode weight_ig; // Wi*x_t  // X is our input.  i means input gate.
	VariableNode weight_hg; // Ui*h_t-1
	VariableNode weight_if; // W_f*x_t
	VariableNode weight_hf; // U_f*h_t-1
	VariableNode weight_ic; // W_c*x_t
	VariableNode weight_hc; // U_c*h_t-1
	VariableNode weight_io;
	VariableNode weight_ho;
	VariableNode bias_g;
	VariableNode bias_f;
	VariableNode bias_c;
	VariableNode bias_o;

	Node initialHidden = null;
	Node initialMemory = null;

	// For training.
	Optimizer optimizer = null;
	Graph trainingGraph = null;
	LSTMStep[] steps = null;
	Node trainingLoss = null;

	// For running.
	LSTMStep nextStep = null; // For running.
	Graph runGraph = null;

	public LSTM(int inputSize, int hiddenSize) {
		this.inputSize = inputSize;
		this.hiddenSize = hiddenSize;

		// 1/n_in from paper.  2/n_in from Google's experiemnts, but we're using Tanh & Sigmoid instead of RELU, so 1.0.
		// The last value * (...) is the Xavier initialization.
		weight_ig = new VariableNode(new Matrix(inputSize, hiddenSize, (i,j)->random.nextGaussian()*(1.0/inputSize)));
		weight_if = new VariableNode(new Matrix(inputSize, hiddenSize, (i,j)->random.nextGaussian()*(1.0/inputSize)));
		weight_ic = new VariableNode(new Matrix(inputSize, hiddenSize, (i,j)->random.nextGaussian()*(1.0/inputSize)));
		weight_io = new VariableNode(new Matrix(inputSize, hiddenSize, (i,j)->random.nextGaussian()*(1.0/inputSize)));
		weight_hg = new VariableNode(new Matrix(hiddenSize, hiddenSize, (i,j)->random.nextGaussian()*(1.0/hiddenSize)));
		weight_hf = new VariableNode(new Matrix(hiddenSize, hiddenSize, (i,j)->random.nextGaussian()*(1.0/hiddenSize)));
		weight_hc = new VariableNode(new Matrix(hiddenSize, hiddenSize, (i,j)->random.nextGaussian()*(1.0/hiddenSize)));
		weight_ho = new VariableNode(new Matrix(hiddenSize, hiddenSize, (i,j)->random.nextGaussian()*(1.0/hiddenSize)));
		bias_g = new VariableNode(1, hiddenSize);
		bias_f = new VariableNode(1, hiddenSize);
		bias_c = new VariableNode(1, hiddenSize);
		bias_o = new VariableNode(1, hiddenSize);

		// Assign some names so we're not debugging blind.
		weight_ig.name = "LSTM_Weight_input_to_gate";
		weight_if.name = "LSTM_Weight_input_to_forget";
		weight_ic.name = "LSTM_Weight_input_to_memory";
		weight_io.name = "LSTM_Weight_input_to_output";

		weight_hg.name = "LSTM_Weight_hidden_to_gate";
		weight_hf.name = "LSTM_Weight_hidden_to_forget";
		weight_hc.name = "LSTM_Weight_hidden_to_memory_cell";
		weight_ho.name = "LSTM_Weight_hidden_to_output";

		bias_g.name = "LSTM_Bias_gate";
		bias_f.name = "LSTM_Bias_forget";
		bias_c.name = "LSTM_Bias_memory";
		bias_o.name = "LSTM_Bias_output";
	}

	private VariableNode[] collectTrainingVariables() {
		return new VariableNode[]{
			weight_ig, // Wi*x_t  // X is our input.  i means input gate.
			weight_hg, // Ui*h_t-1
			weight_if, // W_f*x_t
			weight_hf, // U_f*h_t-1
			weight_ic, // W_c*x_t
			weight_hc, // U_c*h_t-1
			weight_io,
			weight_ho,
			bias_g,
			bias_f,
			bias_c,
			bias_o
		};
	}

	/*** makeRunOnlyLSTM
	 * Freeze and duplicate all the weights and return an LSTM with ONLY the run graph built.
	 * @return
	 */
	public LSTM makeRunOnlyLSTM() {
		LSTM ret = new LSTM(inputSize, hiddenSize);
		ret.weight_ic = new VariableNode(this.weight_ic.getVariable());
		ret.weight_if = new VariableNode(this.weight_if.getVariable());
		ret.weight_ig = new VariableNode(this.weight_ig.getVariable());
		ret.weight_io = new VariableNode(this.weight_io.getVariable());
		ret.weight_hc = new VariableNode(this.weight_hc.getVariable());
		ret.weight_hf = new VariableNode(this.weight_hf.getVariable());
		ret.weight_hg = new VariableNode(this.weight_hg.getVariable());
		ret.weight_ho = new VariableNode(this.weight_ho.getVariable());
		ret.bias_c = new VariableNode(this.bias_c.getVariable());
		ret.bias_f = new VariableNode(this.bias_f.getVariable());
		ret.bias_g = new VariableNode(this.bias_g.getVariable());
		ret.bias_o = new VariableNode(this.bias_o.getVariable());
		return ret;
	}

	private void makeSingleRunStep() {
		runGraph = new Graph();

		initialHidden = new InputNode(1, hiddenSize);
		initialMemory = new InputNode(1, hiddenSize);
		LSTMStep previousStep = new LSTMStep(this);
		previousStep.hidden = initialHidden;
		previousStep.memory = initialMemory;

		Node nextInput = new InputNode(1, inputSize);
		nextStep = previousStep.wireNextStep(nextInput);
		runGraph.addNode(nextStep.hidden);
	}

	private void unroll(int batchSize, int stepsToUnwind, boolean singleLoss) {
		// See unrollAndTrain for the real shape of these things.
		// singleLoss means our loss function is ONLY the last output.
		// singleLoss = false means we add up all the differences between the outputs and the targets.
		steps = new LSTMStep[stepsToUnwind];
		trainingGraph = new Graph();
		initialMemory = new InputNode(1, hiddenSize);
		initialHidden = new InputNode(1, hiddenSize);

		// Start with some default input item.
		LSTMStep previousStep = new LSTMStep(this);
		previousStep.input = new InputNode(batchSize, inputSize);
		previousStep.hidden = initialHidden;
		previousStep.memory = initialMemory;

		for(int i=0; i < stepsToUnwind; i++) {
			Node input = new InputNode(batchSize, inputSize);
			input.name = "LSTMStep_INPUT_" + i;
			steps[i] = previousStep.wireNextStep(input);
			if(!singleLoss || i == stepsToUnwind-1) {
				Node target = new InputNode(batchSize, hiddenSize);
				/*
				Node diff = new SubtractNode(steps[i].out, target);
				Node abs = new AbsNode(diff);
				*/
				Node loss = new SoftmaxLossNode(steps[i].out, target);

				target.name = "LSTMStep_TARGET_" + i;
				//abs.name = "LSTMStep_LOSS_" + i;

				steps[i].target = target;
				steps[i].loss = loss;
			}
			previousStep = steps[i];
		}

		// Add up the graph losses.
		Node lossAccumulator = null;
		if(singleLoss) {
			lossAccumulator = steps[steps.length-1].loss;
		} else {
			for(LSTMStep step : steps) {
				if (lossAccumulator == null) {
					lossAccumulator = step.loss;
				} else {
					// trainingGraph.addNode(step.hidden);
					lossAccumulator = new AddNode(lossAccumulator, step.loss);
				}
			}
		}
		trainingGraph.addNode(lossAccumulator);
		trainingLoss = lossAccumulator;
	}

	/*** unrollAndTrain
	 * Makes an unrolled LSTM which will calculate a bunch of inputs at the same time.
	 * We assume the input matrices are in the following format.
	 * inputs[0] = a bunch of examples = [number of examples in this batch, the size of one step].
	 * outputs[0] = the expected outputs.
	 * For example, if we had a sequence of length 5, "my cat is an asshole", our inputs[] would be length 4 & outputs length 4.
	 * inputs[0] would be 1 row x <number of words> columns
	 * If we had a bunch of sentences to compute in parallel,
	 * inputs[0] would be k rows x <number of keywords> columnts.
	 * "my dog is very sweet" and "What a terrible, stupid encoding scheme" would yield an input matrix array of size
	 * [5][2,dictionary length]
	 * input[0] = ["my" ]  output[0] = ["dog"]  input[1] = ["dog"]  output[1] = ["is"]
	 *            ["what"]             ["a" ]   input[1] = ["a"]    output[1] = ["terrible"]
	 * If our dict were [my dog is very sweet what a terrible stupid encoding scheme], then
	 * input[0] = [1 0 0 0 0 0 0 0 0 0 0]
	 *            [0 0 0 0 0 1 0 0 0 0 0]
	 * @param inputs
	 * @param outputs
	 * @param stepsToUnwind
	 * @return
	 */
	public void unrollAndTrain(Matrix[] inputs, Matrix[] outputs, int stepsToUnwind, double learningRate) {
		if(trainingGraph == null) {
			unroll(inputs[0].rows, stepsToUnwind, false);
		}
		optimizer = new SGD(trainingGraph, collectTrainingVariables(), learningRate);

		// If inputs is length k, we have to do k/stepsToUnwind 'leaps'.
		for(int leap=0; leap < inputs.length-stepsToUnwind; leap += stepsToUnwind) {
			// Assign a value to every input and every output.
			Map<Node, Matrix> feedDict = new HashMap<>();
			feedDict.put(initialMemory, new Matrix(outputs[0].rows, outputs[0].columns));
			feedDict.put(initialHidden, new Matrix(outputs[0].rows, outputs[0].columns));
			for (int i = 0; i < stepsToUnwind; i++) {
				feedDict.put(steps[i].input, inputs[i + leap]);
				feedDict.put(steps[i].target, outputs[i + leap]);
			}

			// Apply every step?
			//optimizer.minimize(trainingLoss, feedDict);
			optimizer.accumulateGradients(trainingLoss, feedDict);


		}
		optimizer.applyGradients();
		optimizer.clearGradients();
	}

	/*** unrollAndTrain
	 * If you only care about the final state of the LSTM, this is the item to use, as compared to the above.
	 * @param inputs
	 * @param output
	 * @param stepsToUnwind
	 * @return
	 */
	void unrollAndTrain(Matrix[] inputs, Matrix output, int stepsToUnwind) {

	}

	// Gives the output at the last step.
	// Each input should be a 1x<unit size> item.
	public Matrix predict(Matrix[] inputs) {
		Matrix[] res = predictStream(inputs);
		return res[res.length-1];
	}

	// Perform a series of predictions, returning the output value from each step.  Uses the input value given.
	public Matrix[] predictStream(Matrix[] inputs) {
		Matrix[] outputs = new Matrix[inputs.length];

		Matrix previousHiddenValue = new Matrix(1, hiddenSize);
		Matrix previousMemoryValue = new Matrix(1, hiddenSize);
		Map<Node, Matrix> feedDict = new HashMap<>();

		for(int i=0; i < inputs.length; i++) {
			Matrix input = inputs[i];
			feedDict.put(nextStep.input, input);
			feedDict.put(initialHidden, previousHiddenValue);
			feedDict.put(initialMemory, previousMemoryValue);
			Matrix[] results = runGraph.forward(feedDict);

			previousHiddenValue = results[nextStep.hidden.id];
			previousMemoryValue = results[nextStep.memory.id];
			outputs[i] = results[nextStep.out.id];
		}

		return outputs;
	}

	// Performs a series of predictions, returning the output value from each and using that as the input value in the next step.
	public Matrix[] generate(Matrix start, int steps) {
		if(runGraph == null) {
			makeSingleRunStep();
		}

		Matrix[] outputs = new Matrix[steps];

		Matrix previousInput = start;
		Matrix previousHiddenValue = new Matrix(1, hiddenSize);
		Matrix previousMemoryValue = new Matrix(1, hiddenSize);
		Map<Node, Matrix> feedDict = new HashMap<>();


		for(int i=0; i < steps; i++) {
			feedDict.put(nextStep.input, previousInput);
			feedDict.put(initialHidden, previousHiddenValue);
			feedDict.put(initialMemory, previousMemoryValue);

			Matrix[] results = runGraph.forward(feedDict);

			previousHiddenValue = results[nextStep.hidden.id];
			previousMemoryValue = results[nextStep.memory.id];
			outputs[i] = results[nextStep.out.id];
			previousInput = results[nextStep.out.id];
		}

		return outputs;
	}

	class LSTMStep {
		LSTM parent;

		// These aren't actually part of the network, but they're useful to track here.
		Node input;
		Node target = null;
		Node loss = null;

		// Taken from Siamese Recurrent Architectures by Muller & Thyagarajan (2016).
		Node gate;
		Node forget;
		Node memory;
		Node out;
		Node hidden;

		LSTMStep(LSTM parentLSTM) {
			// i_c = sigmoid(Wixt + uiht-1 + bi)
			this.parent = parentLSTM;
		}

		LSTMStep wireNextStep(Node input) {
			LSTMStep next = new LSTMStep(this.parent);

			// We need to broadcast the biases so they match the shape of the input count.
			Node bg = new BroadcastNode(parent.bias_g, 1, input.rows);
			Node bf = new BroadcastNode(parent.bias_f, 1, input.rows);
			Node bc = new BroadcastNode(parent.bias_c, 1, input.rows);
			Node bo = new BroadcastNode(parent.bias_o, 1, input.rows);

			next.input = input;
			next.gate = new SigmoidNode(new AddNode(new AddNode(new MatrixMultiplyNode(input, parent.weight_ig), new MatrixMultiplyNode(this.hidden, parent.weight_hg)), bg));
			next.forget = new SigmoidNode(new AddNode(new AddNode(new MatrixMultiplyNode(input, parent.weight_if), new MatrixMultiplyNode(this.hidden, parent.weight_hf)), bf));
			Node memoryHat = new TanhNode(new AddNode(new AddNode(new MatrixMultiplyNode(input, parent.weight_ic), new MatrixMultiplyNode(this.hidden, parent.weight_hc)), bc));
			next.memory = new AddNode(new MultiplyNode(next.gate, memoryHat), new MultiplyNode(next.forget, this.memory));
			next.out = new SigmoidNode(new AddNode(new AddNode(new MatrixMultiplyNode(input, parent.weight_io), new MatrixMultiplyNode(this.hidden, parent.weight_ho)), bo));
			next.hidden = new MultiplyNode(next.out, new TanhNode(next.memory));

			next.gate.name = "LSTMStep_GATE";
			next.forget.name = "LSTMStep_FORGET";
			next.memory.name = "LSTMStep_MEMORY";
			next.out.name = "LSTMStep_OUT";
			next.hidden.name = "LSTMStep_HIDDEN";

			return next;
		}
	}
}
