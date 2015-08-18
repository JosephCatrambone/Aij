package com.josephcatrambone.aij.utilities;

import com.josephcatrambone.aij.networks.Network;

import java.io.*;

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
}
