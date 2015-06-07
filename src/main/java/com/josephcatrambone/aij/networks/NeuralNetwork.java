package com.josephcatrambone.aij.networks;

import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.layers.*;

import java.io.Serializable;

/**
 * Created by jcatrambone on 5/28/15.
 */
public class NeuralNetwork implements Network, Serializable {
	public static double WEIGHT_SCALE = 0.1;

	private Matrix[] weights;
	private Layer[] layers;

	public NeuralNetwork(int[] layerSizes, String[] activationFunctions) {
		layers = new Layer[layerSizes.length];
		weights = new Matrix[layerSizes.length-1];

		for(int i=0; i < layerSizes.length; i++) {
			switch(activationFunctions[i].toLowerCase()) {
				case "linear":
					layers[i] = new LinearLayer(layerSizes[i]);
					break;
				case "tanh":
					layers[i] = new TanhLayer(layerSizes[i]);
					break;
				case "sigmoid":
					layers[i] = new SigmoidLayer(layerSizes[i]);
					break;
				case "softplus":
					layers[i] = new SoftplusLayer(layerSizes[i]);
					break;
				default:
					throw new java.lang.IllegalArgumentException("Invalid layer type.");
			}
		}

		for(int i=0; i < weights.length; i++) {
			weights[i] = Matrix.random(layerSizes[i], layerSizes[i + 1]);
			weights[i].elementMultiply_i(WEIGHT_SCALE);
		}
	}

	public void forwardPropagate(Matrix input) {
		layers[0].setActivities(input);
		for(int i=0; i < weights.length; i++) {
			layers[i+1].setActivities(layers[i].getActivations().multiply(weights[i]));
		}
	}

	public void backPropagate(Matrix output) {
		layers[layers.length-1].setActivities(output); // TODO: Do we set activities or activations?
		layers[layers.length-1].setActivations(output);
		for(int i=layers.length-1; i > 0; i--) {
			layers[i-1].setActivities(layers[i].getActivities().multiply(weights[i-1].transpose()));
		}
	}

	@Override
	public Matrix predict(Matrix input) {
		forwardPropagate(input);
		return layers[layers.length-1].getActivations();
	}

	@Override
	public Matrix reconstruct(Matrix output) {
		return null;
	}

	@Override
	public int getNumInputs() {
		return layers[0].getSize();
	}

	@Override
	public int getNumOutputs() {
		return layers[layers.length-1].getSize();
	}

	@Override
	public Matrix getWeights(int layer) {
		return weights[layer];
	}

	@Override
	public void setWeights(int layer, Matrix weights) {
		this.weights[layer] = weights;
	}

	@Override
	public int getNumLayers() {
		return layers.length;
	}

	@Override
	public Layer getLayer(int index) {
		return layers[index];
	}

	@Override
	public void setLayer(int index, Layer layer) {
		this.layers[index] = layer;
	}
}
