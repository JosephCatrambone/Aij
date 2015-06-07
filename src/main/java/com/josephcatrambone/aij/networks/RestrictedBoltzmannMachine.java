package com.josephcatrambone.aij.networks;

import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.layers.*;

import java.io.Serializable;

/**
 * Created by jcatrambone on 5/28/15.
 */
public class RestrictedBoltzmannMachine implements Network, Serializable {

	Layer visible, hidden;
	Matrix weights;

	public RestrictedBoltzmannMachine(int numVisible, int numHidden) {
		visible = new LinearLayer(numVisible);
		hidden = new SigmoidLayer(numHidden);
		weights = Matrix.random(numVisible, numHidden);
	}

	public RestrictedBoltzmannMachine(int numVisible, int numHidden, String layerType) {
		visible = new LinearLayer(numVisible);
		switch(layerType) {
			case "sigmoid":
				hidden = new SigmoidLayer(numHidden);
				break;
			case "tanh":
				hidden = new TanhLayer(numHidden);
				break;
			case "softplus":
				hidden = new SoftplusLayer(numHidden);
				break;
			case "linear":
				hidden = new LinearLayer(numHidden); // Why would you do this?
				break;
		}
		weights = Matrix.random(numVisible, numHidden);
	}

	@Override
	public Matrix predict(Matrix input) {
		visible.setActivities(input);
		hidden.setActivities(visible.getActivations().multiply(weights));
		return hidden.getActivations();
	}

	@Override
	public Matrix reconstruct(Matrix output) {
		hidden.setActivations(output);
		visible.setActivities(hidden.getActivations().multiply(weights.transpose()));
		//return visible.getActivities();
		return visible.getActivations(); // NOTE: This is NON-STANDARD!  Normally we just get activities.
	}

	@Override
	public int getNumInputs() {
		return visible.getSize();
	}

	@Override
	public int getNumOutputs() {
		return hidden.getSize();
	}

	@Override
	public int getNumLayers() {
		return 2;
	}

	@Override
	public Layer getLayer(int i) {
		if(i == 0) {
			return visible;
		} else if(i == 1) {
			return hidden;
		}
		throw new ArrayIndexOutOfBoundsException(i);
	}

	@Override
	public void setLayer(int i, Layer layer) {
		if(i == 0) {
			visible = layer;
		} else if(i == 1) {
			hidden = layer;
		} else {
			throw new ArrayIndexOutOfBoundsException(i);
		}
	}

	@Override
	public Matrix getWeights(int i) {
		return weights;
	}

	@Override
	public void setWeights(int i, Matrix weights) {
		this.weights = weights;
	}

	// Non-standard methods.

	public Layer getVisible() {
		return visible;
	}

	public Layer getHidden() {
		return hidden;
	}

	public Matrix daydream(int numSamples, int numCycles) {
		// Do numCycles gibbs samples to produce numSample sampels.
		Matrix output = Matrix.random(numSamples, visible.getSize());
		for(int i=0; i < numCycles; i++) {
			output = reconstruct(predict(output));
		}
		return output;
	}
}
