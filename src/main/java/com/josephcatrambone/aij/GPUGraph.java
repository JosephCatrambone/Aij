package com.josephcatrambone.aij;

import org.jocl.*;

import java.util.HashMap;

import static com.josephcatrambone.aij.Graph.NODE_OPERATION.*;
import static org.jocl.CL.*;

/**
 * Created by jcatrambone on 9/19/16.
 */
public class GPUGraph extends Graph {
	public static int BLOCK_SIZE = 16; // Can be changed to update OpenCL Matmul.
	public static final String KERNEL_FORWARD_PREFIX = "fwd_";
	public static final String KERNEL_ADJOINT_PREFIX = "adj_";

	private cl_context context;
	private cl_command_queue commandQueue;
	private cl_program program;
	private cl_kernel[] kernels; // Indexed into the graph node types.
	
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
		cl_queue_properties commandQueueProperties = new cl_queue_properties();
		try {
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
		this.kernels = new cl_kernel[Graph.NODE_OPERATION.values().length];
		for(Graph.NODE_OPERATION n : Graph.NODE_OPERATION.values()) {
			if(n == NODE_OPERATION.INPUT) { continue; }
			// Expecting OpenCL code to define op_ABS, op_MATMUL, etc.
			int[] kernelBuildErrorCodes = new int[1];
			this.kernels[n.ordinal()] = clCreateKernel(program, GPUGraph.KERNEL_FORWARD_PREFIX+ n.name(), kernelBuildErrorCodes);
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
		prog.append("__kernel void fwd_ABS (__global float* target, __global float* src) { int gid = get_global_id(0); target[gid] = fabs(src[gid]); }\n");
		prog.append("__kernel void fwd_EXP (__global float* target, __global float* src) { int gid = get_global_id(0); target[gid] = exp(src[gid]); }\n");
		prog.append("__kernel void fwd_INVERT (__global float* target, __global float* src) { int gid = get_global_id(0); target[gid] = 1.0/src[gid]; }\n");
		prog.append("__kernel void fwd_LOG (__global float* target, __global float* src) { int gid = get_global_id(0); target[gid] = log(src[gid]); }\n");
		prog.append("__kernel void fwd_NEGATE (__global float* target, __global float* src) { int gid = get_global_id(0); target[gid] = -src[gid]; }\n");
		prog.append("__kernel void fwd_POWER2 (__global float* target, __global float* src) { int gid = get_global_id(0); target[gid] = src[gid]*src[gid]; }\n");
		prog.append("__kernel void fwd_SIGMOID (__global float* target, __global float* src) { int gid = get_global_id(0); target[gid] = 1.0/(1.0+exp(-src[gid])); }\n");
		prog.append("__kernel void fwd_TANH (__global float* target, __global float* src) { int gid = get_global_id(0); target[gid] = tanh(src[gid]); }\n");

		// Binary ops.
		// ADD, MULTIPLY, MATRIXMULTIPLY, POWER, SUBTRACT,
		prog.append("__kernel void fwd_ADD(__global float* target, __global float* srcA, __global float* srcB) { int gid = get_global_id(0); target[gid] = srcA[gid]+srcB[gid]; }\n");
		prog.append("__kernel void fwd_MULTIPLY(__global float* target, __global float* srcA, __global float* srcB) { int gid = get_global_id(0); target[gid] = srcA[gid]*srcB[gid]; }\n");
		prog.append("__kernel void fwd_POWER(__global float* target, __global float* srcA, __global float* srcB) { int gid = get_global_id(0); target[gid] = pow(srcA[gid], srcB[0]); }\n");
		prog.append("__kernel void fwd_SUBTRACT(__global float* target, __global float* srcA, __global float* srcB) { int gid = get_global_id(0); target[gid] = srcA[gid]-srcB[gid]; }\n");

		prog.append("#define BLOCK_SIZE " + GPUGraph.BLOCK_SIZE + "\n");
		prog.append("__kernel void fwd_MATRIXMULTIPLY(__global float* target, __global float* srcA, __global float* srcB, int widthA, int widthB) { " +
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
		prog.append("__kernel void fwd_TRANSPOSE(__global float* target, __global float* srcA) { int gid = get_global_id(0); target[gid] = 0; }\n");
		prog.append("__kernel void fwd_TRACE(__global float* target, __global float* srcA) { int gid = get_global_id(0); target[gid] = 0; }\n");

		return prog.toString();
	}

	protected void finalize() throws Throwable {
		try {
			// Release all the memory objects.
			freeMemoryObjects(forward);
			freeMemoryObjects(adjoint);

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

	private void freeMemoryObjects(cl_mem[] objs) {
		if(objs == null) { return; }
		for(cl_mem memObject : objs) {
			clReleaseMemObject(memObject);
		}
	}

	@Override
	public float[] getOutput(HashMap<Integer, float[]> inputs, int node) {
		// Allocate forward pointers based on the sizes.
		freeMemoryObjects(forward);
		forward = new cl_mem[this.names.size()];

		// Copy the inputs to input buffers.
		for(int i=0; i < node; i++) {
			if(ops.get(i) == NODE_OPERATION.INPUT) {
				Pointer inputPtr = Pointer.to(inputs.get(i));
				forward[i] = clCreateBuffer(this.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float*getShape(i).size(), inputPtr, null);
			}
		}

		// Calculate forward.
		for(int i=0; i <= node; i++) {
			evaluateForward(inputs, i);
		}

		// Copy the result to an input buffer.
		float[] result = new float[getShape(node).size()];
		clEnqueueReadBuffer(commandQueue, forward[node], CL_TRUE, 0, result.length * Sizeof.cl_float, Pointer.to(result), 0, null, null);
		return result;
	}

	@Override
	public float[][] getGradient(HashMap<Integer, float[]> inputs, int node) {
		getOutput(inputs, node); // Run forward.

		// If there is memory on the GPU already allocated, free it.
		// For each of the adjoints, reset to zero..
		freeMemoryObjects(adjoint);
		adjoint = new cl_mem[this.names.size()];

		/*
		// Starting adjoint is ones.
		adjoint[node] = new float[shapes.get(node).size()];
		for(int i=0; i < adjoint[node].length; i++) { adjoint[node][i] = 1.0f; }
		// Trace evaluation in reverse order.
		evaluateAdjointChildren(inputs, node);
		*/
		return null;
	}

	private void evaluateAdjointChildren(HashMap<Integer, float[]> inputs, int node) {
		// Each of these operations must operate on its child value, setting the adjoints.
		// From "GPU-accelerated adjoint algorithmic differentiation" by Gremse et al. (2016)
		// ??? Adjoint(X) = Adjoint(Parent(X)) * f'(x)
		for(int arg : arguments.get(node)) {
			// Allocate buffers.
			if(adjoint[arg] == null) {
				//adjoint[arg] = new float[getShape(arg).size()];
			}
		}
		// y := EW(x, op).  x_adj += EW(y_adj, EW(x, d_op), dot)

		// Evaluate the children's children.
		for(int arg : arguments.get(node)) {
			evaluateAdjointChildren(inputs, arg);
		}
	}

	private void evaluateForward(HashMap<Integer, float[]> inputs, int node) {
		if(this.ops.get(node) == INPUT) {
			// IGNORE!  We already allocated and copied.
			return;
		}

		// Allocate memory for result.  First argument is ALWAYS the target.
		forward[node] = clCreateBuffer(this.context, CL_MEM_READ_WRITE, Sizeof.cl_float*getShape(node).size(), null, null);
		clSetKernelArg(this.kernels[this.ops.get(node).ordinal()], 0, Sizeof.cl_mem, Pointer.to(forward[node]));
		for(int i=0; i < arguments.get(node).length; i++) {
			clSetKernelArg(this.kernels[this.ops.get(node).ordinal()], i+1, Sizeof.cl_mem, Pointer.to(forward[arguments.get(node)[i]]));
		}

		// Set the work size.
		long[] globalWorkSize = new long[]{getShape(node).size()}; // TODO: This is probably the wrong size.
		long[] localWorkSize = new long[]{1L};
		int workDim = 1;

		if(this.ops.get(node) == MATRIXMULTIPLY) { // Set work size and enqueue the extra parameters.
			clSetKernelArg(this.kernels[MATRIXMULTIPLY.ordinal()], 3, Sizeof.cl_int, Pointer.to(new int[]{getShape(arguments.get(node)[0]).getWidth()}));
			clSetKernelArg(this.kernels[MATRIXMULTIPLY.ordinal()], 4, Sizeof.cl_int, Pointer.to(new int[]{getShape(arguments.get(node)[1]).getWidth()}));
			localWorkSize = new long[]{ 1, 1 }; // TODO: Tune this.
			globalWorkSize = new long[]{ getShape(node).getWidth(), getShape(node).getHeight() };
			workDim = 2;
		}

		clEnqueueNDRangeKernel(commandQueue, this.kernels[this.ops.get(node).ordinal()], workDim, null, globalWorkSize, localWorkSize, 0, null, null);
	}
}

