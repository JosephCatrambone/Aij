package com.josephcatrambone.aij.models;

import com.josephcatrambone.aij.Graph;
import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.nodes.*;
import com.josephcatrambone.aij.optimizers.Optimizer;
import com.josephcatrambone.aij.optimizers.SGD;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

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

	// For training.
	Optimizer optimizer = null;
	Graph trainingGraph = null;
	LSTMStep[] steps = null;
	Node trainingLoss = null;

	// For running.
	Node initialHidden = null;
	Node initialMemory = null;
	LSTMStep nextStep = null; // For running.
	Graph runGraph = null;

	public LSTM(int inputSize, int hiddenSize) {
		this.inputSize = inputSize;
		this.hiddenSize = hiddenSize;

		final double WEIGHT_SCALE = 0.1;

		weight_ig = new VariableNode(new Matrix(inputSize, hiddenSize, (i, j)->random.nextGaussian()*WEIGHT_SCALE));
		weight_if = new VariableNode(new Matrix(inputSize, hiddenSize, (i,j)->random.nextGaussian()*WEIGHT_SCALE));
		weight_ic = new VariableNode(new Matrix(inputSize, hiddenSize, (i,j)->random.nextGaussian()*WEIGHT_SCALE));
		weight_io = new VariableNode(new Matrix(inputSize, hiddenSize, (i,j)->random.nextGaussian()*WEIGHT_SCALE));
		weight_hg = new VariableNode(new Matrix(hiddenSize, hiddenSize, (i,j)->random.nextGaussian()*WEIGHT_SCALE));
		weight_hf = new VariableNode(new Matrix(hiddenSize, hiddenSize, (i,j)->random.nextGaussian()*WEIGHT_SCALE));
		weight_hc = new VariableNode(new Matrix(hiddenSize, hiddenSize, (i,j)->random.nextGaussian()*WEIGHT_SCALE));
		weight_ho = new VariableNode(new Matrix(hiddenSize, hiddenSize, (i,j)->random.nextGaussian()*WEIGHT_SCALE));
		bias_g = new VariableNode(1, hiddenSize);
		bias_f = new VariableNode(1, hiddenSize);
		bias_c = new VariableNode(1, hiddenSize);
		bias_o = new VariableNode(1, hiddenSize);
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

	private void makeSingleRunStep() {
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

		// Start with some default input item.
		LSTMStep startStep = new LSTMStep(this);
		startStep.input = new InputNode(batchSize, inputSize);
		startStep.hidden = new ConstantNode(batchSize, hiddenSize, 0.0);
		startStep.memory = new ConstantNode(batchSize, hiddenSize, 0.0);

		LSTMStep previousStep = startStep;
		for(int i=0; i < stepsToUnwind; i++) {
			Node input = new InputNode(batchSize, inputSize);
			input.name = "LSTMStep_INPUT_" + i;
			steps[i] = previousStep.wireNextStep(input);
			if(!singleLoss || i == stepsToUnwind-1) {
				Node target = new InputNode(batchSize, inputSize);
				Node diff = new SubtractNode(steps[i].out, target);
				Node abs = new AbsNode(diff);

				target.name = "LSTMStep_TARGET_" + i;
				abs.name = "LSTMStep_LOSS_" + i;

				steps[i].target = target;
				steps[i].loss = abs;
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
	public void unrollAndTrain(Matrix[] inputs, Matrix[] outputs, int stepsToUnwind) {
		if(trainingGraph == null) {
			unroll(inputs[0].rows, stepsToUnwind, false);
			optimizer = new SGD(trainingGraph, collectTrainingVariables(), 0.01);
		}

		// If inputs is length k, we have to do k/stepsToUnwind 'leaps'.
		for(int leap=0; leap < inputs.length/stepsToUnwind; leap++) {
			// Assign a value to every input and every output.
			Map<Node, Matrix> feedDict = new HashMap<>();
			for (int i = 0; i < stepsToUnwind; i++) {
				feedDict.put(steps[i].input, inputs[i + leap*stepsToUnwind]);
				feedDict.put(steps[i].target, outputs[i + leap*stepsToUnwind]);
			}

			optimizer.accumulateGradients(trainingLoss, feedDict);
		}
		// TODO: Should we apply every step?
		optimizer.applyGradients();
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
		if(runGraph == null) {
			makeSingleRunStep();
		}

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
