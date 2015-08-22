package com.josephcatrambone.aij.networks;

import com.josephcatrambone.aij.Matrix;

import java.io.Serializable;

/**
 * Created by jcatrambone on 5/28/15.
 */
public class DeepNetwork implements Network, Serializable {
	private Network[] networks;

	public DeepNetwork(Network... subnets) {
		this.networks = subnets;

		// Verify that each network fits into the one above it.
		for(int i=0; i < subnets.length-1; i++) {
			if(networks[i].getNumOutputs() != networks[i+1].getNumInputs()) {
				throw new IllegalArgumentException("Subnetwork " + i + " output size (" + networks[i].getNumOutputs() +
				") does not match next layer's input size (" + networks[i+1].getNumInputs() + ").");
			}
		}
	}

	@Override
	public Matrix predict(Matrix input) {
		Matrix output = input;
		for(Network n : networks) {
			output = n.predict(output);
		}
		return output;
	}

	@Override
	public Matrix reconstruct(Matrix output) {
		Matrix input = output;
		for(int i=networks.length-1; i >= 0; i--) {
			input = networks[i].reconstruct(input);
		}
		return input;
	}

	@Override
	public int getNumInputs() {
		return networks[0].getNumInputs();
	}

	@Override
	public int getNumOutputs() {
		return networks[networks.length-1].getNumOutputs();
	}

	/*** getNumLayers
	 * Returns the number of sub-networks, not the total number of layers in all networks.
	 * @return
	 */
	@Override
	public int getNumLayers() {
		return networks.length;
	}

	@Override
	public Matrix getWeights(int i) {
		return null;
	}

	@Override
	public void setWeights(int i, Matrix weights) {

	}
}
