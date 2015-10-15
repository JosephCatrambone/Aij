package com.josephcatrambone.aij.utilities;

import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.networks.Network;

import java.io.*;
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
		} catch (IOException ioe) {
			return false;
		}
		return true;
	}

	public static Network LoadNetworkFromDisk(String filename) {
		Network net = null;
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(new File(filename)));
			net = (Network) ois.readObject();
			ois.close();
		} catch (IOException ioe) {
			return null;
		} catch (ClassNotFoundException cnfe) {
			return null;
		}
		return net;
	}

	public static String MatrixToString(Matrix weights) {
		StringBuilder result = new StringBuilder();
		result.append("rows " + weights.numRows() + " ");
		result.append("columns " + weights.numColumns() + " ");
		result.append("matrix ");
		for (int j = 0; j < weights.numRows(); j++) {
			for (int k = 0; k < weights.numColumns(); k++) {
				result.append(weights.get(j, k));
				result.append(" ");
			}
		}
		result.append("\n");
		return result.toString();
	}

	private static void expectString(Scanner scanner, String identifier) {
		// Doesn't return anything, but lets us be consistent with the exceptions we're raising below
		String s = scanner.next();
		if (!identifier.equals(s)) {
			System.err.println("Expectation mismatch: " + identifier + " != " + s);
		}
	}

	public static Matrix StringToMatrix(String string) {
		Scanner scanner = new Scanner(string);

		// Instance the network.

		expectString(scanner, "rows");
		int inputs = scanner.nextInt();
		expectString(scanner, "columns");
		int outputs = scanner.nextInt();
		expectString(scanner, "matrix");
		Matrix weights = new Matrix(inputs, outputs);
		for (int j = 0; j < inputs; j++) {
			for (int k = 0; k < outputs; k++) {
				weights.set(j, k, scanner.nextDouble());
			}
		}
		return weights;
	}
}
