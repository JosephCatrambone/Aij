package com.josephcatrambone.aij.layers;

import com.josephcatrambone.aij.Matrix;

/**
 * Created by jcatrambone on 5/29/15.
 */
public class SigmoidLayer extends LinearLayer {
	public SigmoidLayer(int size) {
		super(size);
	}

	@Override
	public Matrix getActivations() {
		if(dirty) {
			setActivations(getActivities().add(getBias().repmat(getActivities().numRows(), 1)).sigmoid_i());
		}
		return super.getActivations();
	}

	@Override
	public Matrix getGradient() {
		return getActivations().dsigmoidFromActivation();
	}
}
