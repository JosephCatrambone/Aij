package com.josephcatrambone.aij.layers;

import com.josephcatrambone.aij.Matrix;

/**
 * Created by jcatrambone on 5/29/15.
 */
public interface Layer {
	public void setBias(Matrix bias);
	public void setActivities(Matrix preactivation); // activitiy == preactivation
	public void setActivations(Matrix activations);
	public int getSize();
	public Matrix getBias();
	public Matrix getActivities(); // Get activity WITHOUT BIAS.
	public Matrix getActivations(); // Get f(activity + bias)
	public Matrix getGradient();
}
