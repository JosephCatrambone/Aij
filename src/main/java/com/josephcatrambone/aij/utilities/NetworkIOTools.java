package com.josephcatrambone.aij.utilities;

import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.layers.Layer;
import com.josephcatrambone.aij.networks.Network;
import com.josephcatrambone.aij.networks.RestrictedBoltzmannMachine;

import java.io.*;
import java.text.ParseException;
import java.util.Scanner;

/**
 * Created by Jo on 8/17/2015.
 */
public class NetworkIOTools {
	public static boolean SaveNetworkToDisk(Network net, String filename) {
		try {
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(filename)));
			oos.writeObject(net);
			oos.close();
		} catch(IOException ioe) {
			return false;
		}
		return true;
	}

	public static Network LoadNetworkFromDisk(String filename) {
		Network net = null;
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(new File(filename)));
			net = (Network)ois.readObject();
			ois.close();
		} catch(IOException ioe) {
			return null;
		} catch(ClassNotFoundException cnfe) {
			return null;
		}
		return net;
	}

	public static String NetworkToString(Network net) {
		StringBuilder result = new StringBuilder();
		result.append(net.getClass().toString() + "\n");
		result.append("numlayers " + net.getNumLayers() + "\n");
		for(int i=0; i < net.getNumLayers(); i++) {
			result.append("layer " + i + "\n");
			Matrix weights = net.getWeights(i);
			result.append("inputs " + weights.numRows() + "\n");
			result.append("outputs " + weights.numColumns() + "\n");
			result.append("activation " + "NONE" + "\n"); // TODO: Sigmoid? Linear?
			result.append("weights ");
			for(int j=0; j < weights.numRows(); j++) {
				for(int k=0; k < weights.numColumns(); k++) {
					result.append(weights.get(j, k));
					result.append(" ");
				}
			}
			result.append("\n");
		}
		return result.toString();
	}

	private static void expectString(Scanner scanner, String identifier) {
		// Doesn't return anything, but lets us be consistent with the exceptions we're raising below
		String s = scanner.next();
		if(!identifier.equals(s)) {
			String error = "Expectation mismatch: " + identifier + " != " + s;
			System.err.println(error);
		}
	}

	public static RestrictedBoltzmannMachine StringToRBM(String string) {
		Scanner scanner = new Scanner(string);
		RestrictedBoltzmannMachine rbm = null;

		// Instance the network.
		try {
			expectString(scanner, "class");
			Class networkType = Class.forName(scanner.next());
			//rbm = (RestrictedBoltzmannMachine)networkType.newInstance();
		} catch(ClassNotFoundException cnfe) {
			System.err.println(cnfe);
			return null;
		}/* catch(IllegalAccessException iae) {
			System.err.println(iae);
			return null;
		} catch (InstantiationException e) {
			System.err.println(e);
			//e.printStackTrace();
			return null;
		}*/

		// Get network size.
		expectString(scanner, "numlayers");
		int numLayers = scanner.nextInt();
		// Load all layers.
		for(int i=0; i < numLayers; i++) {
			expectString(scanner, "layer");
			int layer = scanner.nextInt();
			expectString(scanner, "inputs");
			int inputs = scanner.nextInt();
			expectString(scanner, "outputs");
			int outputs = scanner.nextInt();
			expectString(scanner, "activation");
			scanner.next(); // Read eventually.
			expectString(scanner, "weights");
			Matrix weights = Matrix.zeros(inputs, outputs);
			for(int j=0; j < inputs; j++) {
				for(int k=0; k < outputs; k++) {
					weights.set(j, k, scanner.nextDouble());
				}
			}
			rbm = new RestrictedBoltzmannMachine(inputs, outputs);
			rbm.setWeights(i, weights);
		}
		return rbm;
	}
}
