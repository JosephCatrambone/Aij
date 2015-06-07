package com.josephcatrambone.aij.trainers;

import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.networks.Network;

/**
 * Created by jcatrambone on 5/29/15.
 */
public interface Trainer {
	void train(Network network, Matrix examples, Matrix labels, Runnable notification);
}
