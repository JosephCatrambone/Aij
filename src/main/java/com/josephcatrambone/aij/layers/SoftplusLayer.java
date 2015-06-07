package com.josephcatrambone.aij.layers;

import com.josephcatrambone.aij.Matrix;

/**
 * Created by jcatrambone on 5/29/15.
 */
public class SoftplusLayer extends LinearLayer {
	public SoftplusLayer(int size) {
		super(size);
	}

	@Override
	public Matrix getActivations() {
		if(dirty) {
			setActivations(getActivities().add(getBias().repmat(getActivities().numRows(), 1)).softplus_i());
		}
		return super.getActivations();
	}

	@Override
	public Matrix getGradient() {
		return getActivations().dsoftplusFromActivation();
	}
}
