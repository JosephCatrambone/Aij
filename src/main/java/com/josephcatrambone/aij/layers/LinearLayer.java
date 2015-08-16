package com.josephcatrambone.aij.layers;

import com.josephcatrambone.aij.Matrix;

import java.io.Serializable;

/**
 * Created by jcatrambone on 5/29/15.
 */
public class LinearLayer implements Layer, Serializable {
	protected boolean dirty = true; // Lazy evaluate activation.
	protected int size;
	private Matrix bias;
	private Matrix activity;
	private Matrix activation; // Keep a copy of activation for efficiency, but be mindful if bias or activity are changed.

	public LinearLayer(int size) {
		dirty = true;
		this.size = size;
		bias = Matrix.zeros(1, size);
		activity = Matrix.zeros(1, size);
	}

	@Override
	public void setBias(Matrix bias) {
		assert(bias.numColumns() == size);
		this.bias = bias;
		dirty = true;
	}

	@Override
	public void setActivities(Matrix preactivation) {
		assert(preactivation.numColumns() == size);
		this.activity = preactivation;
		dirty = true;
	}

	@Override
	public void setActivations(Matrix activations) {
		assert(activations.numColumns() == size);
		this.activation = activations;
		if(activations != null) {
			dirty = false; // If our activation is recalculated, we're good.
		}
	}

	@Override
	public int getSize() {
		return this.size;
	}

	@Override
	public Matrix getBias() {
		return this.bias;
	}

	@Override
	public Matrix getActivities() {
		return activity;
	}

	@Override
	public Matrix getActivations() {
		if(dirty) {
			// Linear, so our activation won't really change, but it might have been changed.
			Matrix acts = getActivities();
			setActivations(acts.add(getBias().repmat(acts.numRows(), 1)));
		}
		return activation;
	}

	@Override
	public Matrix getGradient() {
		// TODO: Null check?
		return Matrix.ones(this.activity.numRows(), this.activity.numColumns());
	}
}
