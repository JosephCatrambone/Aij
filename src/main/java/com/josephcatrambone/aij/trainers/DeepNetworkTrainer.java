package com.josephcatrambone.aij.trainers;

import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.networks.DeepNetwork;
import com.josephcatrambone.aij.networks.Network;

/**
 * Created by josephcatrambone on 12/30/15.
 */
public class DeepNetworkTrainer implements Trainer {
	Trainer[] trainers;

	public DeepNetworkTrainer(DeepNetwork target) {
		trainers = new Trainer[target.getNumLayers()];
	}

	public void setTrainer(int layer, Trainer t) {
		trainers[layer] = t;
	}

	public void train(Network net, Matrix inputs, Matrix labels, Runnable notification) {
		DeepNetwork dn = (DeepNetwork)net;

		Matrix trainingData = inputs;

		// TODO: Check sizes.

		// Unsupervised training for all layers but last.
		for(int i=0; i < dn.getNumLayers()-1; i++) {
			// Sometimes we will have null trainers, like for normalization layers.
			if(trainers[i] != null) {
				trainers[i].train(dn.getSubnet(i), trainingData, null, notification);
			}

			// Abstract out the training data for the next layer.
			trainingData = dn.getSubnet(i).predict(trainingData);
		}

		// Supervised training for last layer.
		trainers[dn.getNumLayers()-1].train(dn.getSubnet(dn.getNumLayers()-1), trainingData, labels, notification);
	}


}
