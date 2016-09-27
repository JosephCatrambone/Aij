package com.josephcatrambone.aij;

import org.jocl.*;

import java.io.*;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;

import static org.jocl.CL.*;

/**
 * Created by jcatrambone on 9/19/16.
 */
public class GPUGraph extends Graph {
	public static final String KERNEL_OPERATION_PREFIX = "op_";

	private cl_context context;
	private cl_command_queue commandQueue;
	private cl_program program;
	private cl_kernel[] kernels; // Indexed into the graph node types.
	
	float[][] forward; // On our device.
	float[][] adjoint; // On CPU.
	cl_mem[] gpuMemoryObjects; // We copy the CPU memory inputs to these.
	Pointer[] forwardPointers;
	Pointer[] adjointPointers;

	public GPUGraph() {
		super();

		final int platformIndex = 0;
		final int deviceIndex = 0;
		final long deviceType = CL_DEVICE_TYPE_ALL;

		CL.setExceptionsEnabled(true);

		// Get platform count.
		int[] platformCountArray = new int[1];
		clGetPlatformIDs(0, null, platformCountArray);
		int platformCount = platformCountArray[0];

		// Get a platform ID.
		cl_platform_id[] platforms = new cl_platform_id[platformCount];
		clGetPlatformIDs(platforms.length, platforms, null);
		cl_platform_id platform = platforms[platformIndex];

		// Initialize context.
		cl_context_properties contextProperties = new cl_context_properties();
		contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform); // Platform any?

		// Get devices.
		int[] deviceCountArray = new int[1];
		clGetDeviceIDs(platform, deviceType, 0, null, deviceCountArray);
		int deviceCount = deviceCountArray[0];
		cl_device_id[] devices = new cl_device_id[deviceCount];
		clGetDeviceIDs(platform, deviceType, deviceCount, devices, null);
		cl_device_id device = devices[deviceIndex];

		// Get a context for the device.
		this.context = clCreateContext(contextProperties, 1, new cl_device_id[]{ device }, null, null, null);

		// Create a command queue.
		cl_queue_properties commandQueueProperties = new cl_queue_properties();
		try {
			this.commandQueue = clCreateCommandQueueWithProperties(this.context, device, commandQueueProperties, null);
		} catch(UnsupportedOperationException uoe) {
			this.commandQueue = clCreateCommandQueue(this.context, device, 0, null); // Deprecated.
		}

		// Create the program from the source in the parent directory.
		String programSource = GPUGraph.buildProgram();
		this.program = clCreateProgramWithSource(context, 1, new String[]{ programSource }, null, null);

		// Create the kernels for each function.
		this.kernels = new cl_kernel[Graph.NODE_OPERATION.values().length];
		for(Graph.NODE_OPERATION n : Graph.NODE_OPERATION.values()) {
			if(n == NODE_OPERATION.INPUT) { continue; }
			// Expecting OpenCL code to define op_ABS, op_MATMUL, etc.
			this.kernels[n.ordinal()] = clCreateKernel(program, GPUGraph.KERNEL_OPERATION_PREFIX + n.name(), null);
		}

		// Allocate memory for the objects.
		// Set the kernel arguments in run.
		// Set the work-item dimensions.
		// Execute kernels.
		// Read output data.
	}

	public static String buildProgram() {
		StringBuilder prog = new StringBuilder();

		// Unary ops.
		// ABS, EXP, INVERT, LOG, NEGATE, POWER2, SIGMOID, TANH
		prog.append("#define ELEMENT_UNARY_OP(NAME, A) __kernel void NAME (__global float* target, __global float* src) { int gid = get_global_id(0); target[gid] = A(src[gid]); }\n");
		prog.append("ELEMENT_UNARY_OP(op_ABS, abs)\n");
		prog.append("ELEMENT_UNARY_OP(op_EXP, exp)\n");
		prog.append("ELEMENT_UNARY_OP(op_INVERT, 1.0/)\n");
		prog.append("ELEMENT_UNARY_OP(op_LOG, log)\n");
		prog.append("ELEMENT_UNARY_OP(op_NEGATE, -)\n");
		prog.append("ELEMENT_UNARY_OP(op_POWER2, exp2)\n");
		prog.append("ELEMENT_UNARY_OP(op_TANH, tanh)\n");

		// Binary ops.
		// ADD, MULTIPLY, MATRIXMULTIPLY, POWER, SUBTRACT,
		prog.append("__kernel void op_ADD(__global float* target, __global float* srcA, __global float* srcB) { int gid = get_global_id(0); target[gid] = srcA[gid]+srcB[gid]; }\n");
		prog.append("__kernel void op_MULTIPLY(__global float* target, __global float* srcA, __global float* srcB) { int gid = get_global_id(0); target[gid] = srcA[gid]*srcB[gid]; }\n");
		prog.append("__kernel void op_POWER(__global float* target, __global float* srcA, __global float* srcB) { int gid = get_global_id(0); target[gid] = pow(srcA[gid], srcB[0]); }\n");
		prog.append("__kernel void op_SUBTRACT(__global float* target, __global float* srcA, __global float* srcB) { int gid = get_global_id(0); target[gid] = srcA[gid]-srcB[gid]; }\n");

		prog.append("__kernel void op_MATRIXMULTIPLY(__global float* target, __global float* srcA, __global float* srcB) { int gid = get_global_id(0); target[gid] = 0; }\n");

		// Special ops.
		// INPUT, TRANSPOSE, TRACE
		prog.append("__kernel void op_TRANSPOSE(__global float* target, __global float* srcA, __global float* srcB) { int gid = get_global_id(0); target[gid] = 0; }\n");
		prog.append("__kernel void op_TRACE(__global float* target, __global float* srcA, __global float* srcB) { int gid = get_global_id(0); target[gid] = 0; }\n");

		return prog.toString();
	}

	public static String loadProgramSource() {
		// TODO: This isn't working well.
		BufferedReader br = null;
		String finalSource = "";

		try {
			File sourceFile = new File(GPUGraph.class.getResource("/cl/gpu_ops.cl").toURI());
			StringBuilder result = new StringBuilder();
			br = new BufferedReader(new FileReader(sourceFile));
			String line = br.readLine();
			while(line != null) {
				result.append(line + "\n");
				line = br.readLine();
			}
			finalSource = result.toString();
			br.close();
		} catch(NullPointerException npe) {
			System.err.println("Problem loading CL Code: " + npe);
		} catch(URISyntaxException urise) {
			System.err.println("Invalid URL: " + urise);
		} catch(FileNotFoundException fnfe) {
			System.err.println("File not found: " + fnfe);
		} catch(IOException ioe) {
			System.err.println("IO Exception: " + ioe);
		}

		return finalSource;
	}

	protected void finalize() throws Throwable {
		try {
			// Release all the memory objects.

			// Release kernels.
			for(cl_kernel k : this.kernels) {
				clReleaseKernel(k);
			}
			this.kernels = null;

			// Release program.
			clReleaseProgram(this.program);

			// Destroy queue.
			clReleaseCommandQueue(this.commandQueue);

			// Finally, release context.
			clReleaseContext(this.context);
		} finally {
			super.finalize();
		}
	}

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

		// If there is memory on the GPU already allocated, free it.
		// For each of the adjoints, reset to zero..

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
				for(int arg : arguments.get(node)) {
					elementBinaryOp(forward[arg], forward[node], forward[node], (x,y) -> x*y); // fwd[this] = fwd[this] + fwd[arg]
				}
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
