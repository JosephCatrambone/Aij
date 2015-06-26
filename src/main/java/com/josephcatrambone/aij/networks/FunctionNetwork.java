package com.josephcatrambone.aij.networks;

import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.layers.Layer;

import java.util.function.Consumer;
import java.util.function.UnaryOperator;

/**
 * A debugging network which has 1:1 connections from in to out.
 * If output size is zero or the preduction function is null, reutrns null.
 * Created by Jo on 6/7/2015.
 */
public class FunctionNetwork implements Network {
	private int inputSize, outputSize;

	public Consumer <Matrix> predictionMonitor = null;
	public Consumer <Matrix> reconstructionMonitor = null;
	public UnaryOperator <Matrix> predictionFunction = null;
	public UnaryOperator <Matrix> reconstructionFunction = UnaryOperator.identity();

	public FunctionNetwork(int inputSize, int outputSize) {
		this.inputSize = inputSize;
		this.outputSize = outputSize;
	}

	@Override
	public Matrix predict(Matrix input) {
		if(predictionMonitor != null) {
			predictionMonitor.accept(input);
		}

		if(predictionFunction != null && outputSize > 0) {
			return predictionFunction.apply(input);
		} else {
			return null;
		}
	}

	@Override
	public Matrix reconstruct(Matrix output) {
		if(reconstructionMonitor != null) {
			reconstructionMonitor.accept(output);
		}

		if(reconstructionFunction != null) {
			return reconstructionFunction.apply(output);
		} else {
			return null;
		}
	}

	@Override
	public int getNumInputs() {
		return inputSize;
	}

	@Override
	public int getNumOutputs() {
		return outputSize;
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
