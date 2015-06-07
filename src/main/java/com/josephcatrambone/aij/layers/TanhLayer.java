package com.josephcatrambone.aij.layers;

import com.josephcatrambone.aij.Matrix;

import java.io.Serializable;

/**
 * Created by jcatrambone on 5/29/15.
 */
public class TanhLayer extends LinearLayer {
	public TanhLayer(int size) {
		super(size);
	}

	@Override
	public Matrix getActivations() {
		if(dirty) {
			setActivations(getActivities().add(getBias().repmat(getActivities().numRows(), 1)).tanh_i()); // Can do _i because add return a new mat.
		}
		return super.getActivations();
	}

	@Override
	public Matrix getGradient() {
		// tanh(mat) -> tanh
		// d/dx tanh -> 1 - tanh(x)^2
		// But if x is already tanh(x), d/dx -> 1 - x^2
		return getActivations().dtanhFromActivation();
	}
}
