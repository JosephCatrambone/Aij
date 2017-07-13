package com.josephcatrambone.aij.models;

import com.josephcatrambone.aij.Graph;
import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.nodes.*;
import com.josephcatrambone.aij.optimizers.Optimizer;

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

	Optimizer optimizer = null;
	Graph trainingGraph = null;
	LSTMStep[] steps = null;
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
	float unrollAndTrain(Matrix[] inputs, Matrix[] outputs, int stepsToUnwind) {
		if(trainingGraph == null) {
			steps = new LSTMStep[stepsToUnwind];
			trainingGraph = new Graph();

			// Start with some default input item.
			LSTMStep startStep = new LSTMStep(this);
			startStep.input = new InputNode(inputs[0].rows, inputSize);
			startStep.hidden = new ConstantNode(inputs[0].rows, hiddenSize, 0.0);
			startStep.memory = new ConstantNode(inputs[0].rows, hiddenSize, 0.0);

			steps[0] = startStep;
			for(int i=1; i < stepsToUnwind; i++) {
				Node input = new InputNode(inputs[0].rows, inputSize);
				steps[i] = steps[i-1].wireNextStep(input);
			}

			// Wire up a loss for all of them.

			// Add all of them to the training graph.
			for(LSTMStep step : steps) {
				trainingGraph.addNode(step.hidden);
				// TODO: Start here.
			}
		}

		return 0f;
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

			next.input = input;
			next.gate = new SigmoidNode(new AddNode(new AddNode(new MatrixMultiplyNode(input, parent.weight_ig), new MatrixMultiplyNode(this.hidden, parent.weight_hg)), parent.bias_g));
			next.forget = new SigmoidNode(new AddNode(new AddNode(new MatrixMultiplyNode(input, parent.weight_if), new MatrixMultiplyNode(this.hidden, parent.weight_hf)), parent.bias_f));
			Node memoryHat = new TanhNode(new AddNode(new AddNode(new MatrixMultiplyNode(input, parent.weight_ic), new MatrixMultiplyNode(this.hidden, parent.weight_hc)), parent.bias_c));
			next.memory = new AddNode(new MultiplyNode(next.gate, memoryHat), new MultiplyNode(next.forget, this.memory));
			next.out = new SigmoidNode(new AddNode(new AddNode(new MatrixMultiplyNode(input, parent.weight_io), new MatrixMultiplyNode(this.hidden, parent.weight_ho)), parent.bias_o));
			next.hidden = new MultiplyNode(next.out, new TanhNode(next.memory));

			return next;
		}
	}
}























































/*
I feel useless.
I've felt useless for a while.  At work, my sole purpose is to do machine learning stuff, but I'm not very good at it.
I've not made anything in production that's being used.  The stuff I have made is being replaced.
First it was LDA topics.  I couldn't calculate them and we had to bring in an outside team to do it.
Next was doing the similarity DB.  My solution to store them in the database is being replaced by the data team.
Next is the home front.  Someone without AI experience has an offer to work at OpenAI.  :(  Good for him, I mean,
but darn for me.  I can't seem to make anything of value.
 */


























































