package com.josephcatrambone.aij.networks;

import com.josephcatrambone.aij.Matrix;

/**
 * Created by jcatrambone on 5/28/15.
 */
public interface Network {
	Matrix predict(Matrix input);
	Matrix reconstruct(Matrix output);
	int getNumInputs();
	int getNumOutputs();
	int getNumLayers();
	Matrix getWeights(int i);
	void setWeights(int i, Matrix weights);
}
