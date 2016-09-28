package com.josephcatrambone.aij;

import java.util.HashMap;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;

/**
 * Created by jcatrambone on 9/19/16.
 */
public class CPUGraph extends Graph {

	float[][] forward;
	float[][] adjoint;

	@Override
	public float[] getOutput(HashMap<Integer, float[]> inputs, int node) {
		forward = new float[this.names.size()][];
		for(int i=0; i <= node; i++) {
			evaluateForward(inputs, i);
		}
		return forward[node];
	}

	@Override
	public float[][] getGradient(HashMap<Integer, float[]> inputs, int node) {
		getOutput(inputs, node); // Run forward.

		adjoint = new float[this.names.size()][];
		// Starting adjoint is ones.
		adjoint[node] = new float[shapes.get(node).size()];
		for(int i=0; i < adjoint[node].length; i++) { adjoint[node][i] = 1.0f; }
		// Trace evaluation in reverse order.
		evaluateAdjointChildren(inputs, node);
		return adjoint;
	}

	private void elementBinaryOp(float[] srcA, float[] srcB, float[] dst, BinaryOperator<Float> op) {
		for(int i=0; i < srcA.length; i++) {
			dst[i] = op.apply(srcA[i], srcB[i]);
		}
	}

	private void elementUnaryOp(float[] src, float[] dst, UnaryOperator<Float> op) {
		for (int i = 0; i < src.length; i++) {
			dst[i] = op.apply(src[i]);
		}
	}

	private void evaluateAdjointChildren(HashMap<Integer, float[]> inputs, int node) {
		// Each of these operations must operate on its child value, setting the adjoints.
		// From "GPU-accelerated adjoint algorithmic differentiation" by Gremse et al. (2016)
		// ??? Adjoint(X) = Adjoint(Parent(X)) * f'(x)
		for(int arg : arguments.get(node)) {
			// Allocate buffers.
			if(adjoint[arg] == null) {
				adjoint[arg] = new float[getShape(arg).size()];
			}
		}

		// y := EW(x, op).  x_adj += EW(y_adj, EW(x, d_op), dot)

		int left = -1;
		int right = -1;
		Dimension leftShape;
		Dimension rightShape;

		switch(this.ops.get(node)) {
			case ADD: // z := x + y -> x_adj += z, y_adj += z.
				for(int arg : arguments.get(node)) {
					for (int i = 0; i < adjoint[node].length; i++) {
						adjoint[arg][i] += adjoint[node][i];
					}
				}
				break;
			case EXP:
				// y := EW(x, op) -> y = e^(x)
				// x_adj += y_adj*e^x
				left = arguments.get(node)[0];
				leftShape = getShape(left);
				for(int i=0; i < leftShape.size(); i++) {
					adjoint[left][i] = adjoint[node][i]*(float)Math.exp(forward[left][i]);
				}
				break;
			case SUBTRACT: // z := x + y -> x_adj += z, y_adj += z.
				left = arguments.get(node)[0];
				right = arguments.get(node)[1];
				for (int i = 0; i < adjoint[node].length; i++) {
					adjoint[left][i] += adjoint[node][i];
					adjoint[right][i] += -adjoint[node][i];
				}
				break;
			case MULTIPLY: // z = x*y. dz/da = d/da x*y + x * d/day  x_adj += z_adj*y.  y_adj += z_adj*x.
				// z = x1*y1, x2*y2, x3*y3
				// x1_adj += z_adj*y1
				// x2_adj += z_adj*y2
				left = arguments.get(node)[0];
				right = arguments.get(node)[1];
				for (int i = 0; i < adjoint[node].length; i++) {
					adjoint[left][i] += adjoint[node][i] * forward[right][i];
					adjoint[right][i] += adjoint[node][i] * forward[left][i];
				}

				break;
			case MATRIXMULTIPLY:
				// C = AB -> A^ = C^B.T.  B^ = A.TC^
				left = arguments.get(node)[0];
				right = arguments.get(node)[1];
				leftShape = getShape(left);
				rightShape = getShape(right);
				Dimension thisShape = getShape(node);

				// First, left = C_adj (x) B_transpose.
				for(int y=0; y < leftShape.getHeight(); y++) {
					for(int x=0; x < leftShape.getWidth(); x++) {
						// C_adj * B_t
						float accumulator = 0;
						for(int k=0; k < thisShape.getWidth(); k++) {
							//accumulator += forward[left][i + y*leftShape.getWidth()] * forward[right][x + i*rightShape.getWidth()];
							accumulator += adjoint[node][k + y*thisShape.getWidth()] * forward[right][k*rightShape.getHeight() + y]; // Need to transpose forward.
							// ____  0 1 2 3 4 5
							// R:   [a b c d e f]
							// R_t: [a c b e d f]
							// R: 2x3 (2 row, 3 col.  w = 3, h = 2.)
							// R: x + y*w -> x + y*3 -> 0, 1, 2, 3, 4, 5
							// R_t: x + y*w -> x + y*2 -> [0, 2, 1, 4, 3, 5]
							// _____ 0  1  2  3  4  5  6  7  8
							// R:   [a, b, c, d, e, f, g, h, i]
							// R_t: [a, d, g, b, e, h, c, f, i]
							// 0, 3, 6, 1, 4, 7, 2, 5, 8
							// x*w_new + y
						}
						adjoint[left][x + y*leftShape.getWidth()] += accumulator;
					}
				}

				// right = A_transpose * C_adj. -> leftShape rows x this cols -> left height x this width
				// A_transpose * C_adj -> A(mxn) * C(mxo) -> A(nxm) * C(mxo)
				// Result is nxo.  A_columns by C_columns.
				for(int y=0; y < leftShape.getColumns(); y++) {
					for(int x=0; x < thisShape.getColumns(); x++) {
						float accumulator = 0.0f;
						for(int k=0; k < leftShape.getRows(); k++) {
							// First row * first column.
							// Except left is transpose, so we do the first column * first column.
							float fwd = forward[left][y + k*leftShape.getWidth()];
							float adj = adjoint[node][x + k*thisShape.getWidth()];
							accumulator += fwd * adj;
						}
						adjoint[right][x + y*rightShape.getWidth()] = accumulator;
					}
				}

				break;
			case INVERT:
				// y := EW(x, op) -> y = 1/x
				// f(x) = x^-1.  df(x) = -x^2
				// x_adj += y_adj * -(x*x)
				left = arguments.get(node)[0];
				leftShape = getShape(left);
				for(int i=0; i < leftShape.size(); i++) {
					adjoint[left][i] = adjoint[node][i] * -(forward[left][i]*forward[left][i]);
				}
				break;
			case LOG:
				// y := EW(x, op) -> y = log(x)
				// f(x) = log(x).  df(x) = 1/x
				// x_adj += y_adj/x
				left = arguments.get(node)[0];
				leftShape = getShape(left);
				for(int i=0; i < leftShape.size(); i++) {
					adjoint[left][i] = adjoint[node][i]/forward[left][i];
				}
				break;
			case NEGATE:
				// y := EW(x, op).  x_adj += EW(y_adj, EW(x, d_op), dot)
				// y = -x.  x_adj = -y_adj .
				left = arguments.get(node)[0];
				leftShape = getShape(left);
				for(int i=0; i < leftShape.size(); i++) {
					adjoint[left][i] = -adjoint[node][i];
				}
				break;
			case TRANSPOSE:
				throw new RuntimeException("Not yet implemented.");
			case POWER:
				// y := EW(x, op).  x_adj += EW(y_adj, EW(x, d_op), dot)
				// y = x^n.  x_adj = y_adj * (2*x^(n-1)) for all.
				left = arguments.get(node)[0];
				right = arguments.get(node)[1];
				leftShape = getShape(left);
				for(int i=0; i < leftShape.size(); i++) {
					adjoint[left][i] = adjoint[node][i]*(forward[right][0] * (float)Math.pow(forward[left][i], forward[right][0]-1));
				}
				break;
			case POWER2:
				left = arguments.get(node)[0];
				leftShape = getShape(left);
				for(int i=0; i < leftShape.size(); i++) {
					adjoint[left][i] = adjoint[node][i]*(2.0f * forward[left][i]);
				}
				break;
			case TANH:
				left = arguments.get(node)[0];
				leftShape = getShape(left);
				// y := EW(x, op).  x_adj += EW(y_adj, EW(x, d_op), dot)
				// 1 - tanh^2
				for(int i=0; i < leftShape.size(); i++) {
					float th = (float)Math.tanh(forward[left][i]);
					adjoint[left][i] += adjoint[node][i]*(1.0f - (th*th));
				}
				break;
			case SIGMOID:
				left = arguments.get(node)[0];
				leftShape = getShape(left);
				// y := EW(x, op).  x_adj += EW(y_adj, EW(x, d_op), dot)
				// sig(x) * (1-sig(x))
				for(int i=0; i < leftShape.size(); i++) {
					float sigX = (float)1.0f/(1.0f+(float)Math.exp(-forward[left][i]));
					adjoint[left][i] += adjoint[node][i]*(sigX * (1.0f - sigX));
				}
				break;
			case ABS:
				left = arguments.get(node)[0];
				leftShape = getShape(left);
				// y := EW(x, op).  x_adj += EW(y_adj, EW(x, d_op), dot)
				for(int i=0; i < leftShape.size(); i++) {
					float dAbs = forward[left][i];
					if(dAbs < 0) { dAbs = -1f; }
					else if(dAbs == 0) { dAbs = 0f; }
					else { dAbs = 1f; }
					adjoint[left][i] += adjoint[node][i]*dAbs;
				}
			case INPUT:
				// Do nothing.
				break;
			default:
				throw new RuntimeException("Invalid operation in graph: " + this.ops.get(node));
		}

		// Evaluate the children's children.
		for(int arg : arguments.get(node)) {
			evaluateAdjointChildren(inputs, arg);
		}
	}

	private void evaluateForward(HashMap<Integer, float[]> inputs, int node) {
		forward[node] = new float[shapes.get(node).size()];

		int left = -1;
		int right = -1;
		Dimension leftShape = null;
		Dimension rightShape = null;

		switch(this.ops.get(node)) {
			case ABS:
				elementUnaryOp(forward[arguments.get(node)[0]], forward[node], x -> Math.abs(x));
				break;
			case ADD:
				for(int arg : arguments.get(node)) {
					elementBinaryOp(forward[arg], forward[node], forward[node], (x,y) -> x+y); // fwd[this] = fwd[this] + fwd[arg]
				}
				break;
			case EXP:
				elementUnaryOp(forward[arguments.get(node)[0]], forward[node], x -> (float)Math.exp(x));
				break;
			case SUBTRACT:
				left = arguments.get(node)[0];
				right = arguments.get(node)[1];
				elementBinaryOp(forward[left], forward[right], forward[node], (x,y) -> x-y);
				break;
			case MULTIPLY:
				left = arguments.get(node)[0];
				right = arguments.get(node)[1];
				elementBinaryOp(forward[left], forward[right], forward[node], (x,y) -> x*y); // fwd[this] = fwd[this] + fwd[arg]
				break;
			case MATRIXMULTIPLY:
				// MxN -> M rows N columns -> N width, M height.
				left = arguments.get(node)[0];
				right = arguments.get(node)[1];
				leftShape = this.shapes.get(left);
				rightShape = this.shapes.get(right);
				assert(leftShape.getColumns() == rightShape.getRows());

				int resultHeight = leftShape.getHeight(); // r = h
				int resultWidth = rightShape.getWidth(); // c = w
				for(int y=0; y < resultHeight; y++) {
					for(int x=0; x < resultWidth; x++) {
						float accumulator = 0;
						for(int i=0; i < leftShape.getWidth(); i++) {
							accumulator += forward[left][i + y*leftShape.getWidth()] * forward[right][x + i*rightShape.getWidth()];
						}
						forward[node][x + y*resultWidth] = accumulator;
					}
				}
				break;
			case INVERT:
				for(int arg : arguments.get(node)) {
					elementUnaryOp(forward[arg], forward[node], (x) -> 1.0f/x);
				}
				break;
			case NEGATE:
				for(int arg : arguments.get(node)) {
					elementUnaryOp(forward[arg], forward[node], (x) -> -x);
				}
				break;
			case INPUT:
				elementUnaryOp(inputs.get(node), forward[node], (x) -> x);
				break;
			case TRANSPOSE:
				// MxN -> M rows N columns -> N width, M height.
				int srcArg = this.arguments.get(node)[0];
				Dimension srcShape = this.shapes.get(srcArg);
				Dimension newShape = this.shapes.get(node);
				for(int y=0; y < newShape.getHeight(); y++) {
					for(int x=0; x < newShape.getWidth(); x++) {
						forward[node][x + y*newShape.getWidth()] = forward[srcArg][y + x*srcShape.getWidth()];
					}
				}
				break;
			case LOG:
				elementUnaryOp(forward[arguments.get(node)[0]], forward[node], x -> (float)Math.log(x));
				break;
			case TANH:
				elementUnaryOp(forward[arguments.get(node)[0]], forward[node], x -> (float)Math.tanh(x));
				break;
			case SIGMOID:
				elementUnaryOp(forward[arguments.get(node)[0]], forward[node], x -> 1.0f/(1.0f+(float)Math.exp(-x)));
				break;
			case POWER:
				float[] base = forward[arguments.get(node)[0]];
				float exp = forward[arguments.get(node)[1]][0];
				elementUnaryOp(base, forward[node], (x) -> (float)Math.pow(x, exp));
				break;
			case POWER2:
				elementUnaryOp(forward[arguments.get(node)[0]], forward[node], (x) -> (float)Math.pow(x, 2.0f));
				break;
			case TRACE:
				left = this.arguments.get(node)[0];
				leftShape = this.shapes.get(left);
				for(int i=0; i < Math.min(leftShape.getWidth(), leftShape.getHeight()); i++) {
					forward[node][i] = forward[left][i+i*leftShape.getWidth()];
				}
				break;
			default:
				throw new RuntimeException("Invalid operation in graph.");
		}
	}
}