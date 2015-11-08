import com.josephcatrambone.aij.networks.*;
import com.josephcatrambone.aij.utilities.NetworkIOTools;
import org.junit.Test;

import java.util.logging.Logger;

/**
 * Created by josephcatrambone on 8/19/15.
 */
public class IOToolTest {
	private static final Logger LOGGER = Logger.getLogger(TestRBMTrainer.class.getName());

	@Test
	public void testStringIO() {
	}

	@Test
	public void testSerializerSanity() {
		//LOGGER.log(Level.INFO, "Testing Serialization...");
		final String TEST_MODEL_FILENAME = "test.model";
		Network test = null;
		// This does not do a full test of the serializers.  It only checks to make sure they can save/load without probs.

		// RBM
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(10, 10);
		org.junit.Assert.assertTrue(NetworkIOTools.saveNetworkToDisk(rbm, TEST_MODEL_FILENAME));
		test = NetworkIOTools.loadNetworkFromDisk(TEST_MODEL_FILENAME);
		org.junit.Assert.assertNotNull(test);

		// Verify CN can serialize/deserialize.
		ConvolutionalNetwork cn = new ConvolutionalNetwork(rbm, 1, 1, 1, 1, 1, 1, 1, 1, ConvolutionalNetwork.EdgeBehavior.ZEROS);
		org.junit.Assert.assertTrue(NetworkIOTools.saveNetworkToDisk(cn, TEST_MODEL_FILENAME));
		test = NetworkIOTools.loadNetworkFromDisk(TEST_MODEL_FILENAME);
		org.junit.Assert.assertNotNull(test);

		// Verify DeepNetwork
		DeepNetwork dn = new DeepNetwork(rbm, rbm, rbm);
		org.junit.Assert.assertTrue(NetworkIOTools.saveNetworkToDisk(dn, TEST_MODEL_FILENAME));
		test = NetworkIOTools.loadNetworkFromDisk(TEST_MODEL_FILENAME);
		org.junit.Assert.assertNotNull(test);

		// Verify function network
		FunctionNetwork fn = new FunctionNetwork(10, 10);
		org.junit.Assert.assertTrue(NetworkIOTools.saveNetworkToDisk(fn, TEST_MODEL_FILENAME));
		test = NetworkIOTools.loadNetworkFromDisk(TEST_MODEL_FILENAME);
		org.junit.Assert.assertNotNull(test);

		// If function network works, MaxPol and MeanFilter should work, too, but...
		MaxPoolingNetwork mpn = new MaxPoolingNetwork(10);
		org.junit.Assert.assertTrue(NetworkIOTools.saveNetworkToDisk(mpn, TEST_MODEL_FILENAME));
		test = NetworkIOTools.loadNetworkFromDisk(TEST_MODEL_FILENAME);
		org.junit.Assert.assertNotNull(test);

		// Neural Network
		NeuralNetwork nn = new NeuralNetwork(new int[]{1, 2, 3}, new String[]{"tanh", "tanh", "tanh"});
		org.junit.Assert.assertTrue(NetworkIOTools.saveNetworkToDisk(nn, TEST_MODEL_FILENAME));
		test = NetworkIOTools.loadNetworkFromDisk(TEST_MODEL_FILENAME);
		org.junit.Assert.assertNotNull(test);
	}
}
