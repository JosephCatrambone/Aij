import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.networks.Network;
import com.josephcatrambone.aij.networks.RestrictedBoltzmannMachine;
import com.josephcatrambone.aij.utilities.NetworkIOTools;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

/**
 * Created by josephcatrambone on 8/19/15.
 */
public class IOToolTest {
	@Test
	public void testStringIO() {
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(20, 10);
		String rbmSerialized = NetworkIOTools.NetworkToString(rbm);
		Network net = NetworkIOTools.StringToNetwork(rbmSerialized);
		RestrictedBoltzmannMachine rbm2 = (RestrictedBoltzmannMachine)net;
		System.out.println(rbmSerialized);
		assertArrayEquals("String IO Mismatch", rbm.getWeights(0).getRowArray(0), rbm2.getWeights(0).getRowArray(0), 1e-5);
	}
}
