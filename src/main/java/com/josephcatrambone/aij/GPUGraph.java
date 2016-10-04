package com.josephcatrambone.aij;

import org.jocl.*;

import java.util.HashMap;

import static com.josephcatrambone.aij.Graph.NODE_OPERATION.*;
import static org.jocl.CL.*;

/**
 * Created by jcatrambone on 9/19/16.
 */
public class GPUGraph extends Graph {
	public static final int BLOCK_SIZE = 1;
	public static final String KERNEL_FORWARD_PREFIX = "fwd_";
	public static final String KERNEL_ADJOINT_PREFIX = "adj_";

	private cl_context context;
	private cl_command_queue commandQueue;
	private cl_program program;
	private HashMap<String, cl_kernel> kernels; // Indexed into the graph node types.  Need to index based on op name.
	
	cl_mem[] forward; // We copy the CPU memory inputs to these.
	cl_mem[] adjoint;

	public GPUGraph() {
		super();

		final int platformIndex = 0;
		final int deviceIndex = 0;
		final long deviceType = CL_DEVICE_TYPE_GPU; //CL_DEVICE_TYPE_ALL;

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
		try {
			cl_queue_properties commandQueueProperties = new cl_queue_properties();
			this.commandQueue = clCreateCommandQueueWithProperties(this.context, device, commandQueueProperties, null);
		} catch(UnsupportedOperationException uoe) {
			this.commandQueue = clCreateCommandQueue(this.context, device, 0, null); // Deprecated.
		}

		// Create the program from the source in the parent directory.
		String programSource = GPUGraph.buildProgram();
		int[] programBuildErrorCodes = new int[1];
		this.program = clCreateProgramWithSource(context, 1, new String[]{ programSource }, null, programBuildErrorCodes);
		clBuildProgram(this.program, 0, null, null, null, null);
		assert(programBuildErrorCodes[0] == CL_SUCCESS);
		//clGetProgramBuildInfo(this.program, device, CL_PROGRAM_BUILD_LOG, 1, 0, )

		// Create the kernels for each function.
		this.kernels = new HashMap<>();
		for(Graph.NODE_OPERATION n : Graph.NODE_OPERATION.values()) {
			if(n == NODE_OPERATION.INPUT) { continue; }
			// Expecting OpenCL code to define op_ABS, op_MATMUL, etc.
			int[] kernelBuildErrorCodes = new int[1];
			this.kernels.put(GPUGraph.KERNEL_FORWARD_PREFIX + n.name(), clCreateKernel(program, GPUGraph.KERNEL_FORWARD_PREFIX+ n.name(), kernelBuildErrorCodes));
			this.kernels.put(GPUGraph.KERNEL_ADJOINT_PREFIX + n.name(), clCreateKernel(program, GPUGraph.KERNEL_ADJOINT_PREFIX+ n.name(), kernelBuildErrorCodes));
		}

		// Allocate memory for the objects.
		// Set the kernel arguments in run.
		// Set the work-item dimensions.
		// Execute kernels.
		// Read output data.
	}

	private void runForward(HashMap<Integer, float[]> inputs, int node, boolean freeMemory) {
		// Allocate forward pointers based on the sizes.
		if(freeMemory || forward == null) {
			freeMemoryObjects(forward);
			forward = new cl_mem[this.names.size()];
		}

		// Copy the inputs to input buffers.
		for(int i=0; i <= node; i++) {
			if(ops.get(i) == NODE_OPERATION.INPUT) {
				Pointer inputPtr = Pointer.to(inputs.get(i));
				forward[i] = clCreateBuffer(this.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float*getShape(i).size(), inputPtr, null);
			} else {
				forward[i] = clCreateBuffer(this.context, CL_MEM_READ_WRITE, Sizeof.cl_float*getShape(i).size(), null, null);
			}
		}

		// Calculate forward.
		for(int i=0; i <= node; i++) {
			evaluateForward(i);
		}

		// Flush ops.
		clFinish(commandQueue);
	}

	private void runBackward(HashMap<Integer, float[]> inputs, int node, boolean freeMemory) {
		runForward(inputs, node, true); // Run forward.

		// If there is memory on the GPU already allocated, free it.
		// For each of the adjoints, reset to zero..
		freeMemoryObjects(adjoint);
		adjoint = new cl_mem[this.names.size()];

		// Pre-allocate adjoint memory blocks.
		float[] ones = new float[getShape(node).size()]; for(int i=0; i < getShape(node).size(); i++) { ones[i] = 1.0f; }
		adjoint[node] = clCreateBuffer(this.context, CL_MEM_COPY_HOST_PTR, Sizeof.cl_float*ones.length, Pointer.to(ones), null);
		float[] zeros = new float[]{ 0.0f };
		for(int i=node-1; i >= 0; i--) {
			adjoint[i] = clCreateBuffer(this.context, CL_MEM_READ_WRITE, Sizeof.cl_float*getShape(i).size(), null, null);
			// Init adjoints of this node to ones. Init children to zero.
			clEnqueueFillBuffer(this.commandQueue, adjoint[i], Pointer.to(zeros), 1, 0, Sizeof.cl_float*getShape(i).size(), 0, null, null);
		}
		clEnqueueBarrier(commandQueue);

		// Trace evaluation in reverse order.
		evaluateAdjointChildren(node);

		// Execute all the queued operations.
		clFinish(commandQueue);
	}

	private static String makeForwardUnaryOpWrapper(String name, String sourceOp) {
		// Used to avoid naming conflicts.
		return "__kernel void " + KERNEL_FORWARD_PREFIX + name + "(__global float* restrict target, __global const float* restrict src) { int gid = get_global_id(0); target[gid] = "+ sourceOp + "}\n";
	}

	private static String makeForwardBinaryOpWrapper(String name, String sourceOp) {
		return "__kernel void " + KERNEL_FORWARD_PREFIX + name + "(__global float* restrict target, __global const float* srcA, __global const float* srcB) { int gid = get_global_id(0); target[gid] = "+ sourceOp + "}\n";
	}

	private static String makeAdjointUnaryOpWrapper(String name, String derivativeOperation) {
		// Let the first argument by the adjoin values we're going to run, the second value be the parent's ajoint, and the third be the value forward.
		// y := EW(x, op).  x_adj += EW(y_adj, EW(x, d_op), dot)
		// Forward is the value going forward.  Don't have inputs, though.
		// derivativeOperation should be something like parentAdjoint[gid]*derivative(foward[gid])
		return "__kernel void " + KERNEL_ADJOINT_PREFIX + name + "(__global float* restrict targetAdjoint, __global const float* restrict parentAdjoint, __global const float* forward) { int gid = get_global_id(0); targetAdjoint[gid] += "+ derivativeOperation + "}\n";
	}

	private static String makeAdjointBinaryOpWrapper(String name, String operation) {
		// First argument is left adjoint.  Second argument is right adjoint.  Third arg is parent adjoint.  Fourth argument is left forward.  Fifth is right forward.  Sixth is op.
		// Available variables: g, leftAdjoint, rightAdjoint, parentAdjoint, leftForward, rightForward
		// Should be something like leftAdjoint[g] += foo(bar(parentAdjoint));
		return "__kernel void " + KERNEL_ADJOINT_PREFIX + name + "(__global float* restrict leftAdjoint, __global float* restrict rightAdjoint, __global const float* parentAdjoint, __global const float* leftForward, __global const float* rightForward) { int g = get_global_id(0); "+ operation + "}\n";
	}

	public static String buildProgram() {
		StringBuilder prog = new StringBuilder();

		prog.append("#define BLOCK_SIZE " + GPUGraph.BLOCK_SIZE + "\n");
		// TODO: Vectorize ops.
		// kernel void op_ABS4(global float4 *target, global const float4 *src)

		// FORWARD OPERATIONS

		// Unary ops.
		// ABS, EXP, INVERT, LOG, NEGATE, POWER2, SIGMOID, TANH
		prog.append(makeForwardUnaryOpWrapper("ABS", "fabs(src[gid]);"));
		prog.append(makeForwardUnaryOpWrapper("EXP", "exp(src[gid]);"));
		prog.append(makeForwardUnaryOpWrapper("INVERT", "1.0f/(src[gid]);"));
		prog.append(makeForwardUnaryOpWrapper("LOG", "log(src[gid]);"));
		prog.append(makeForwardUnaryOpWrapper("NEGATE", "-(src[gid]);"));
		prog.append(makeForwardUnaryOpWrapper("POWER2", "src[gid]*src[gid];"));
		prog.append(makeForwardUnaryOpWrapper("SIGMOID", "1.0f/(1.0f+exp(-src[gid]));"));
		prog.append(makeForwardUnaryOpWrapper("TANH", "tanh(src[gid]);"));

		// Binary ops.
		// ADD, MULTIPLY, MATRIXMULTIPLY, POWER, SUBTRACT,
		prog.append(makeForwardBinaryOpWrapper("ADD", "srcA[gid]+srcB[gid];"));
		prog.append(makeForwardBinaryOpWrapper("MULTIPLY", "srcA[gid]*srcB[gid];"));
		prog.append(makeForwardBinaryOpWrapper("POWER", "pow(srcA[gid], srcB[0]);"));
		prog.append(makeForwardBinaryOpWrapper("SUBTRACT", "srcA[gid]-srcB[gid];"));

		prog.append("__kernel void fwd_MATRIXMULTIPLY(__global float* restrict target, __global const float* restrict srcA, __global const float* restrict srcB, int widthA, int widthB) { " +
			"int blockIndexX = get_global_id(0); " +
			"int blockIndexY = get_global_id(1); " +
			//"int threadIndexX = get_local_id(0); " +
			//"int threadIndexY = get_local_id(1); " +
			"float accumulator = 0;" +
			"for(int k=0; k < widthA; ++k) {" +
				//"barrier(CLK_LOCAL_MEM_FENCE);" +
				"accumulator += srcA[k + blockIndexY*widthA]*srcB[blockIndexX + k*widthB];" +
			"}" +
			"target[blockIndexX + blockIndexY*widthB] = accumulator; }\n");

		// Special ops.
		// INPUT, TRANSPOSE, TRACE
		prog.append("__kernel void fwd_TRANSPOSE(__global float* restrict target, __global float* src, int srcWidth, int srcHeight) { int g = get_global_id(0); target[g/srcWidth + (g%srcWidth)*srcHeight] = src[(g/srcWidth) + g]; }\n");
		prog.append("__kernel void fwd_TRACE(__global float* restrict target, __global float* srcA) { int gid = get_global_id(0); target[gid] = 0; }\n");

		/*
		 * Quick notes on TRANSPOSE.
		 * If g is in float[data], x = g%width, y = g/width.
		 * Want: b[y + x*b_width] := a[x + y*a_width].  b[i,j] = a[j,i].
		 * b_width = a_height
		 * y = g/a_width.  x = g%a_width.
		 * b[g/a_width + (g%a_width)*a_height] = a[(g%a_width) + (g/a_width)*a_width] = a[(g%a_width) + g]
		 */

		// ADJOINT/BACKWARDS OPERATIONS.

		// ADD: adjoint[arg][i] += adjoint[node][i]; // z := x + y -> x_adj += z, y_adj += z.
		prog.append(makeAdjointBinaryOpWrapper("ADD", "leftAdjoint[g] += parentAdjoint[g]; rightAdjoint[g] += parentAdjoint[g];"));

		// EXP: adjoint[left][i] = adjoint[node][i]*(float)Math.exp(forward[left][i]);
		prog.append(makeAdjointUnaryOpWrapper("EXP", "parentAdjoint[gid]*exp(forward[gid]);"));

		// SUB: adjoint[left][i] += adjoint[node][i];
		// SUB: adjoint[right][i] += -adjoint[node][i];
		prog.append(makeAdjointBinaryOpWrapper("SUBTRACT", "leftAdjoint[g] += parentAdjoint[g]; rightAdjoint[g] += -parentAdjoint[g];"));

		// MULTIPLY: // z = x*y. dz/da = d/da x*y + x * d/day  x_adj += z_adj*y.  y_adj += z_adj*x.
		// z = x1*y1, x2*y2, x3*y3
		// x1_adj += z_adj*y1
		// x2_adj += z_adj*y2
		//adjoint[left][i] += adjoint[node][i] * forward[right][i];
		//adjoint[right][i] += adjoint[node][i] * forward[left][i];
		prog.append(makeAdjointBinaryOpWrapper("MULTIPLY", "leftAdjoint[g] += parentAdjoint[g]*rightForward[g]; rightAdjoint[g] += parentAdjoint[g]*leftForward[g];"));

		// Matmul:
		// C = AB -> A^ = C^B.T.  B^ = A.TC^
		// First, left = C_adj (x) B_transpose.
		//accumulator += adjoint[node][k + y*thisShape.getWidth()] * forward[right][k*rightShape.getHeight() + y]; // Need to transpose forward.
		//adjoint[left][x + y*leftShape.getWidth()] += accumulator;
		// right = A_transpose * C_adj. -> leftShape rows x this cols -> left height x this width
		// A_transpose * C_adj -> A(mxn) * C(mxo) -> A(nxm) * C(mxo)
		//float fwd = forward[left][y + k*leftShape.getWidth()];
		//float adj = adjoint[node][x + k*thisShape.getWidth()];
		//adjoint[right][x + y*rightShape.getWidth()] = accumulator;
		// First argument is left adjoint.  Second argument is right adjoint.  Third arg is parent adjoint.  Fourth argument is left forward.  Fifth is right forward.  Sixth is op.
		prog.append("__kernel void adj_MATRIXMULTIPLY(__global float* restrict leftAdjoint, __global float* rightAdjoint, __global float* parentAdjoint, __global const float* leftForward, __global const float* rightForward, int leftWidth, int leftHeight, int rightWidth, int rightHeight) {" +
			"int leftX = get_global_id(0)%leftWidth;" +
			"int leftY = get_global_id(0)/leftWidth;" +
			"int rightX = get_global_id(1)%rightWidth;" +
			"int rightY = get_global_id(1)/rightWidth;" +
			"float leftAdjointAccumulator = 0, rightAdjointAccumulator = 0;" +
			"for(int k=0; k < rightWidth; k++) { leftAdjointAccumulator += parentAdjoint[k + leftY*leftWidth]*rightForward[leftY + k*rightHeight]; }" + // k < parentWidth -> A*B -> mxn * nxo -> mxo h*wL h*wR wR
			"for(int k=0; k < leftWidth; k++) { rightAdjointAccumulator += parentAdjoint[k + leftY*leftWidth]*leftForward[rightX + k*rightWidth]; }" +
			"leftAdjoint[leftX + leftY*leftWidth] += leftAdjointAccumulator;" +
			"rightAdjoint[rightX + rightY*rightWidth] += rightAdjointAccumulator;"+
		"}");

		// INVERT:
		// y := EW(x, op) -> y = 1/x
		// f(x) = x^-1.  df(x) = -x^2
		//adjoint[left][i] = adjoint[node][i] * -(forward[left][i]*forward[left][i]);
		prog.append(makeAdjointUnaryOpWrapper("INVERT", "parentAdjoint[gid]* (-1.0f/(forward[gid]*forward[gid]));"));

		// LOG:
		// y := EW(x, op) -> y = log(x)
		// f(x) = log(x).  df(x) = 1/x
		// x_adj += y_adj/x
		//adjoint[left][i] = adjoint[node][i]/forward[left][i];
		prog.append(makeAdjointUnaryOpWrapper("LOG", "parentAdjoint[gid]/forward[gid];"));

		//NEGATE:
		// y := EW(x, op).  x_adj += EW(y_adj, EW(x, d_op), dot)
		// y = -x.  x_adj = -y_adj .
		//adjoint[left][i] = -adjoint[node][i];
		prog.append(makeAdjointUnaryOpWrapper("NEGATE", "-parentAdjoint[gid];"));

		//POWER:
		// y := EW(x, op).  x_adj += EW(y_adj, EW(x, d_op), dot)
		// y = x^n.  x_adj = y_adj * (2*x^(n-1)) for all.
		//adjoint[left][i] = adjoint[node][i]*(forward[right][0] * (float)Math.pow(forward[left][i], forward[right][0]-1));
		prog.append(makeAdjointBinaryOpWrapper("POWER", "leftAdjoint[g] += parentAdjoint[g]*(rightForward[0] * pow(leftForward[g], rightForward[0]-1));")); // TODO: Generalize from [0] on right.

		//POWER2:
		//adjoint[left][i] = adjoint[node][i]*(2.0f * forward[left][i]);
		prog.append(makeAdjointUnaryOpWrapper("POWER2", "parentAdjoint[gid]*(2.0f * forward[gid]);"));

		//TANH:
		// y := EW(x, op).  x_adj += EW(y_adj, EW(x, d_op), dot)
		// 1 - tanh^2
		//float th = (float)Math.tanh(forward[left][i]);
		//adjoint[left][i] += adjoint[node][i]*(1.0f - (th*th));
		prog.append(makeAdjointUnaryOpWrapper("TANH", "parentAdjoint[gid]*(1.0f - (tanh(forward[gid])*tanh(forward[gid])));"));

		//SIGMOID:
		// y := EW(x, op).  x_adj += EW(y_adj, EW(x, d_op), dot)
		// sig(x) * (1-sig(x))
		//float sigX = (float)1.0f/(1.0f+(float)Math.exp(-forward[left][i]));
		//adjoint[left][i] += adjoint[node][i]*(sigX * (1.0f - sigX));
		prog.append(makeAdjointUnaryOpWrapper("SIGMOID", "parentAdjoint[gid]*((1.0f/(1.0f+exp(-forward[gid]))) * (1.0f - (1.0f/(1.0f+exp(-forward[gid])))));"));

		//ABS:
		// y := EW(x, op).  x_adj += EW(y_adj, EW(x, d_op), dot)
		//adjoint[left][i] += adjoint[node][i]*dAbs;
		prog.append(makeAdjointUnaryOpWrapper("ABS", "parentAdjoint[gid] * copysign(1.0f, forward[gid]);"));

		prog.append("__kernel void adj_TRANSPOSE(__global float* restrict target, __global float* src, int srcWidth, int srcHeight) {}\n");
		prog.append("__kernel void adj_TRACE(__global float* restrict target, __global float* srcA) {}\n");

		prog.append("__kernel void fwd_ADD_BROADCAST(__global float* restrict target, __global float* srcA, __global float* srcB, int srcBWidth) { int g = get_global_id(0); target[g] = srcA[g] + srcB[g%srcBWidth]; }\n");
		prog.append("__kernel void adj_ADD_BROADCAST(__global float* restrict leftAdjoint, __global float* restrict rightAdjoint, __global const float* restrict parentAdjoint, int rightWidth) { int g = get_global_id(0); leftAdjoint[g] += parentAdjoint[g]; rightAdjoint[g%rightWidth] += parentAdjoint[g]/rightWidth; }\n");

		return prog.toString();
	}

	protected void finalize() throws Throwable {
		try {
			// Release all the memory objects.
			freeMemoryObjects(forward);
			freeMemoryObjects(adjoint);

			// Release kernels.
			for(cl_kernel k : this.kernels.values()) {
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

	private void freeMemoryObjects(cl_mem[] objs) {
		if(objs == null) { return; }
		for(cl_mem memObject : objs) {
			try {
				clReleaseMemObject(memObject);
			} catch(org.jocl.CLException clexception) {
				// CL_INVALID_MEM_OBJECT?
				if(clexception.getStatus() == CL_INVALID_MEM_OBJECT) {
					System.err.println("Attempted free of invalid memory object.");
				}
			}
		}
	}

	@Override
	public float[] getOutput(HashMap<Integer, float[]> inputs, int node) {
		runForward(inputs, node, true);

		// Copy the result to an input buffer.
		float[] result = new float[getShape(node).size()];
		clEnqueueReadBuffer(commandQueue, forward[node], CL_TRUE, 0, result.length * Sizeof.cl_float, Pointer.to(result), 0, null, null);
		return result;
	}

	@Override
	public float[][] getGradient(HashMap<Integer, float[]> inputs, int node) {
		runBackward(inputs, node, true);

		// Copy adjoints back from memory.
		float[][] result = new float[names.size()][];
		for(int i=0; i <= node; i++) {
			float[] buffer = new float[getShape(i).size()];
			Pointer bufferPointer = Pointer.to(buffer);
			clEnqueueReadBuffer(commandQueue, adjoint[i], CL_TRUE, 0, getShape(i).size()*Sizeof.cl_float, bufferPointer, 0, null, null);
			result[i] = buffer;
		}

		// Make sure the copies are finished.
		clFinish(commandQueue);
		return result;
	}

	private void evaluateAdjointChildren(int node) {
		// y := EW(x, op).  x_adj += EW(y_adj, EW(x, d_op), dot)
		// Populate this node's children's adjoint values.
		String kernelName = KERNEL_ADJOINT_PREFIX + this.ops.get(node).name();
		long[] globalWorkSize = new long[]{getShape(node).size()};
		long[] localWorkSize = new long[]{1L};
		int workDim = 1;

		// Copy from above: Let the first argument by the adjoin values we're going to run, the second value be the parent's ajoint, and the third be the value forward.
		// Execute the derivative op.
		if(ops.get(node) == INPUT) {
			return;
		} else if(ops.get(node) == MATRIXMULTIPLY) {
			// Special case.
			int[] args = arguments.get(node);
			clSetKernelArg(this.kernels.get(kernelName), 0, Sizeof.cl_mem, Pointer.to(adjoint[args[0]]));
			clSetKernelArg(this.kernels.get(kernelName), 1, Sizeof.cl_mem, Pointer.to(adjoint[args[1]]));
			clSetKernelArg(this.kernels.get(kernelName), 2, Sizeof.cl_mem, Pointer.to(adjoint[node]));
			clSetKernelArg(this.kernels.get(kernelName), 3, Sizeof.cl_mem, Pointer.to(forward[args[0]]));
			clSetKernelArg(this.kernels.get(kernelName), 4, Sizeof.cl_mem, Pointer.to(forward[args[1]]));

			clSetKernelArg(this.kernels.get(kernelName), 5, Sizeof.cl_int, Pointer.to(new int[]{getShape(args[0]).getWidth()}));
			clSetKernelArg(this.kernels.get(kernelName), 6, Sizeof.cl_int, Pointer.to(new int[]{getShape(args[0]).getHeight()}));
			clSetKernelArg(this.kernels.get(kernelName), 7, Sizeof.cl_int, Pointer.to(new int[]{getShape(args[1]).getWidth()}));
			clSetKernelArg(this.kernels.get(kernelName), 8, Sizeof.cl_int, Pointer.to(new int[]{getShape(args[1]).getHeight()}));

			localWorkSize = new long[]{1, 1}; // TODO: Tune this.
			globalWorkSize = new long[]{getShape(args[0]).size(), getShape(args[1]).size()};
			workDim = 2;
		} else if(ops.get(node) == ADD_BROADCAST) {
			int[] args = arguments.get(node);
			clSetKernelArg(this.kernels.get(kernelName), 0, Sizeof.cl_mem, Pointer.to(adjoint[args[0]]));
			clSetKernelArg(this.kernels.get(kernelName), 1, Sizeof.cl_mem, Pointer.to(adjoint[args[1]]));
			clSetKernelArg(this.kernels.get(kernelName), 2, Sizeof.cl_mem, Pointer.to(adjoint[node]));
			clSetKernelArg(this.kernels.get(kernelName), 3, Sizeof.cl_int, Pointer.to(new int[]{getShape(args[1]).getWidth()}));
		} else {
			int currentArg = 0;
			for(int arg : arguments.get(node)) {
				// Let first arguments be the adjoints..
				clSetKernelArg(this.kernels.get(kernelName), currentArg++, Sizeof.cl_mem, Pointer.to(adjoint[arg]));
			}
			// Second arg is the parent adjoint.
			clSetKernelArg(this.kernels.get(kernelName), currentArg++, Sizeof.cl_mem, Pointer.to(adjoint[node]));
			for(int arg : arguments.get(node)) {
				// Third args are the forward values.
				clSetKernelArg(this.kernels.get(kernelName), currentArg++, Sizeof.cl_mem, Pointer.to(forward[arg]));
			}
		}

		clEnqueueNDRangeKernel(commandQueue, this.kernels.get(kernelName), workDim, null, globalWorkSize, localWorkSize, 0, null, null);
		clEnqueueBarrier(commandQueue);

		// Evaluate the children's children.
		for(int arg : arguments.get(node)) {
			evaluateAdjointChildren(arg);
		}
	}

	private void evaluateForward(int node) {
		if(this.ops.get(node) == INPUT) {
			return;
		}

		// First argument is ALWAYS the target.  Other arguments are, well, arguments.
		clSetKernelArg(this.kernels.get(KERNEL_FORWARD_PREFIX + this.ops.get(node).name()), 0, Sizeof.cl_mem, Pointer.to(forward[node]));
		for(int i=0; i < arguments.get(node).length; i++) {
			clSetKernelArg(this.kernels.get(KERNEL_FORWARD_PREFIX + this.ops.get(node).name()), i+1, Sizeof.cl_mem, Pointer.to(forward[arguments.get(node)[i]]));
		}

		// Set the work size.  Global work size is the number of things you want done.  It is always 0-(N-1).
		// Local work size is a value in 0-local_size.
		long[] globalWorkSize = new long[]{getShape(node).size()}; // TODO: This is probably the wrong size.
		long[] localWorkSize = new long[]{1L};
		int workDim = 1;

		if(this.ops.get(node) == MATRIXMULTIPLY) { // Set work size and enqueue the extra parameters.
			clSetKernelArg(this.kernels.get(KERNEL_FORWARD_PREFIX + MATRIXMULTIPLY.name()), 3, Sizeof.cl_int, Pointer.to(new int[]{getShape(arguments.get(node)[0]).getWidth()}));
			clSetKernelArg(this.kernels.get(KERNEL_FORWARD_PREFIX + MATRIXMULTIPLY.name()), 4, Sizeof.cl_int, Pointer.to(new int[]{getShape(arguments.get(node)[1]).getWidth()}));
			localWorkSize = new long[]{ 1, 1 }; // TODO: Tune this.
			globalWorkSize = new long[]{ getShape(node).getWidth(), getShape(node).getHeight() };
			workDim = 2;
		} else if(this.ops.get(node) == TRANSPOSE) {
			clSetKernelArg(this.kernels.get(KERNEL_FORWARD_PREFIX + TRANSPOSE.name()), 2, Sizeof.cl_int, Pointer.to(new int[]{getShape(arguments.get(node)[0]).getWidth()}));
			clSetKernelArg(this.kernels.get(KERNEL_FORWARD_PREFIX + TRANSPOSE.name()), 3, Sizeof.cl_int, Pointer.to(new int[]{getShape(arguments.get(node)[0]).getHeight()}));
		} else if(this.ops.get(node) == ADD_BROADCAST) {
			clSetKernelArg(this.kernels.get(KERNEL_FORWARD_PREFIX + ADD_BROADCAST.name()), 3, Sizeof.cl_int, Pointer.to(new int[]{getShape(arguments.get(node)[1]).getWidth()}));
		}

		clEnqueueNDRangeKernel(commandQueue, this.kernels.get(KERNEL_FORWARD_PREFIX + this.ops.get(node).name()), workDim, null, globalWorkSize, localWorkSize, 0, null, null);
		clEnqueueBarrier(commandQueue);
	}
}


