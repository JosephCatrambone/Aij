package com.josephcatrambone.aij.networks;

import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.layers.Layer;

import java.io.Serializable;

/**
 * Created by jcatrambone on 5/28/15.
 *
 * Will train a network on data which is sampled and resized on a row-by-row basis.
 * Example: If we set windowWidth to 3 and window height to 2, we will assume an input with 6 elements
 * need to be
 * like this:
 * [1 2 3 4 5 6]
 * [
 */
public class ConvolutionalNetwork implements Network, Serializable{
	public enum EdgeBehavior {ZEROS, REPEAT, MIRROR, WRAP};

	private Network operator;
	private int windowWidth, windowHeight;
	private int exampleWidth, exampleHeight;
	private int convolutionOutputWidth, convolutionOutputHeight;
	private int xStep, yStep;
	private EdgeBehavior edgeBehavior;

	/*** ConvolutionalNetwork
	 *
	 * @param op
	 * @param exampleWidth The number of columns in an input.  Inputs are provided as row-examples, so we reshape them to this many columns when training.
	 * @param exampleHeight Number of rows in an input example.
	 * @param windowWidth The number of columns that should be sampled from an input example.
	 * @param windowHeight
	 * @param convolutionOutputWidth The expected output width of a single window operation.
	 * @param convolutionOutputHeight
	 * @param xStep How many columns we should step between input samples.
	 * @param yStep
	 */
	public ConvolutionalNetwork(Network op,
								int exampleWidth, int exampleHeight,
								int windowWidth, int windowHeight,
								int convolutionOutputWidth, int convolutionOutputHeight,
								int xStep, int yStep,
								EdgeBehavior boundaryBehavior) {
		this.operator = op;
		this.windowWidth = windowWidth;
		this.windowHeight = windowHeight;
		this.exampleWidth = exampleWidth;
		this.exampleHeight = exampleHeight;
		this.convolutionOutputWidth = convolutionOutputWidth;
		this.convolutionOutputHeight = convolutionOutputHeight;
		this.xStep = xStep;
		this.yStep = yStep;
		this.edgeBehavior = boundaryBehavior;
	}

	/*** convolve2D
	 * Run a convolution on each of the windows in an example.
	 * We accept a ton of parameters so we can use this for convolution and deconvolution.
	 * You probably don't want to do this unless you REALLY know what you're doing.  You should probably use
	 * predict and reconstruct instead.
	 * @return
	 */
	public Matrix convolve2D(Matrix input, ConvolutionOperation convOp, boolean hasOutput, int eW, int eH, int eXStep, int eYStep, int wW, int wH, int cW, int cH, int cXStep, int cYStep) {
		Matrix output = null;

		if(hasOutput) { output = new Matrix(input.numRows(), getNumOutputs()); }

		for(int exampleIndex = 0; exampleIndex < input.numRows(); exampleIndex++) {
			// For each example row, reshape to a 2D 'image'.
			Matrix exampleIn = input.getRow(exampleIndex).reshape_i(eH, eW);
			Matrix exampleOut = null;
			if(hasOutput) { exampleOut = new Matrix(cH *(eH/eYStep), cW *(eW/eXStep)); }

			for(int y=0; y < eH/eYStep; y++) { // y is the sample multiplier.
				for(int x=0; x < eW/eXStep; x++) {
					// We have a global x/y offset, now.
					// Copy the example matrix into the subsample, being mindful of the edge condition.
					Matrix subsample = Matrix.zeros(wH, wW);
					for(int windowY = 0; windowY < wH; windowY++) {
						for (int windowX = 0; windowX < wW; windowX++) {
							int exampleY = (y*eYStep)-(wH/2) + windowY;
							int exampleX = (x*eXStep)-(wW/2) + windowX;
							if(exampleY < 0 || exampleY >= exampleIn.numRows() || exampleX < 0 || exampleX >= exampleIn.numColumns()) {
								switch(edgeBehavior) {
									case ZEROS:
										subsample.set(windowY, windowX, 0.0);
										break;
									case REPEAT:
										//subsample.set(exampleY, exampleX, exampleIn.get(Math.min(exampleIn.numRows(), Math.max(0, exampleYPosition)), Math.min(exampleIn.numColumns(), Math.max(0, exampleXPosition))));
										break;
									case MIRROR:
										break;
									case WRAP:
										break;
								}
							} else {
								subsample.set(windowY, windowX, exampleIn.get(exampleY, exampleX));
							}
						}
					}
					// Reshape the subsample into something linear.
					subsample.reshape_i(1, wW*wH);
					// Run the operation.
					Matrix result = convOp.op(subsample);
					// If a result is given by the op, reshape it and apply to the output.
					if(hasOutput) {
						result.reshape_i(cH, cW);
						exampleOut.addSubmatrix_i(result, y*cYStep, x*cXStep); // We may want to overlay the 'conv' in some direction.
						//exampleOut.setSubmatrix_i(result, y*cH, x*cW);
					}
				}
			}
			if(hasOutput) {
				output.setRow(exampleIndex, exampleOut.reshape_i(1, exampleOut.numColumns() * exampleOut.numRows()));
			}
		}
		return output;
	}

	private Matrix convolve1D(Matrix input, ConvolutionOperation convOp) {
		return null;
	}

	@Override
	public Matrix predict(Matrix input) {
		return convolve2D(input, new ConvolutionOperation() {
			@Override
			public Matrix op(Matrix input) {
				return operator.predict(input);
			}
		}, true,
			exampleWidth, exampleHeight,
			xStep, yStep,
			windowWidth, windowHeight,
			convolutionOutputWidth, convolutionOutputHeight,
			convolutionOutputWidth, convolutionOutputHeight
		);
	}

	@Override
	public Matrix reconstruct(Matrix output) {
		return convolve2D(output, new ConvolutionOperation() {
			@Override
			public Matrix op(Matrix input) {
				return operator.reconstruct(input);
			}
		}, true,
			convolutionOutputWidth*(exampleWidth/xStep), convolutionOutputHeight*(exampleHeight/yStep), // Our convolution layer is this big
			convolutionOutputWidth, convolutionOutputHeight, // We assume there's no overlap on the conv layer and step across examples like this.
			convolutionOutputWidth, convolutionOutputHeight, // Our window size is one convolution op.
			windowWidth, windowHeight, // Out convolution output, in this case, is the size of the original sample.
			xStep, yStep // And we step across it with the same step size.
		);
	}

	@Override
	public int getNumInputs() {
		return exampleWidth*exampleHeight;
	}

	@Override
	public int getNumOutputs() {
		return getOperations()*operator.getNumOutputs();
	}

	/*** getOperations
	 * Return the number of times 'operator' is run on the input of the specified size.
	 * Our output will be flat, but we resample in column-prime (x first, C-style) order.
	 * If we have a 15x14 matrix (210 elements), and a window size of 3x7,
	 * that means we have 5 vertical (row-wise) steps and 2 horizontal (column-wise) steps,
	 * for a total output of 10 outputs.
	 * @return
	 */
	public int getOperations() {
		return (exampleWidth/xStep) * (exampleHeight/yStep);
	}

	public Network getOperator() { return this.operator; }

	public void setOperator(Network op) { this.operator = op; }

	@Override
	public int getNumLayers() {
		return operator.getNumLayers();
	}

	@Override
	public Layer getLayer(int i) {
		return operator.getLayer(i);
	}

	@Override
	public void setLayer(int i, Layer layer) {
		operator.setLayer(i, layer);
	}

	@Override
	public Matrix getWeights(int i) {
		return operator.getWeights(i);
	}

	@Override
	public void setWeights(int i, Matrix weights) {
		operator.setWeights(i, weights);
	}

	private interface ConvolutionOperation {
		public Matrix op(Matrix input);
	}
}
