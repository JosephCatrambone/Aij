package com.josephcatrambone.aij.networks;

import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.layers.Layer;

/**
 * A debugging network which has 1:1 connections from in to out.
 * Created by Jo on 6/7/2015.
 */
public class OneToOneNetwork implements Network {
	private int size;
	public Monitor predictionMonitor = null, reconstructionMonitor = null;

	public interface Monitor {
		void run(Matrix intermediate);
	}

	public OneToOneNetwork(int size) {
		this.size = size;
	}

	@Override
	public Matrix predict(Matrix input) {
		if(predictionMonitor != null) {
			predictionMonitor.run(input);
		}
		return input;
	}

	@Override
	public Matrix reconstruct(Matrix output) {
		if(reconstructionMonitor != null) {
			reconstructionMonitor.run(output);
		}
		return output;
	}

	@Override
	public int getNumInputs() {
		return size;
	}

	@Override
	public int getNumOutputs() {
		return size;
	}

	@Override
	public int getNumLayers() {
		return 0;
	}

	@Override
	public Layer getLayer(int i) {
		return null;
	}

	@Override
	public void setLayer(int i, Layer layer) {
	}

	@Override
	public Matrix getWeights(int i) {
		return null;
	}

	@Override
	public void setWeights(int i, Matrix weights) {
	}
}
