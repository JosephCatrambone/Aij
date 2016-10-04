package com.josephcatrambone.aij;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Created by jcatrambone on 9/12/16.
 */
public abstract class Graph {
	public enum NODE_OPERATION {ABS, ADD, ADD_BROADCAST, EXP, INPUT, INVERT, LOG, MULTIPLY, MATRIXMULTIPLY, NEGATE, POWER, POWER2, SIGMOID, SUBTRACT, TANH, TRANSPOSE, TRACE};
	public List<String> names;
	public List<NODE_OPERATION> ops;
	public List<int[]> arguments;
	public List<Dimension> shapes;

	public Graph() {
		this.names = new ArrayList<>();
		this.ops = new ArrayList<>();
		this.arguments = new ArrayList<>();
		this.shapes = new ArrayList<>();
	}

	public int addNode(String name, NODE_OPERATION op, int[] inputs) {
		int id = this.names.size();
		this.names.add(name);
		this.ops.add(op);
		this.arguments.add(inputs);
		this.shapes.add(getShape(id));
		return id;
	}

	public int addNode(String name, NODE_OPERATION op, int[] inputs, Dimension shape) {
		int id = this.names.size();
		this.names.add(name);
		this.ops.add(op);
		this.arguments.add(inputs);
		this.shapes.add(shape);
		return id;
	}

	public int addInput(String name, Dimension shape) {
		return addNode(name, NODE_OPERATION.INPUT, new int[]{}, shape);
	}

	public Dimension getShape(int node) {
		if(shapes.contains(node)) {
			return shapes.get(node);
		}

		switch(ops.get(node)) {
			case ABS:
			case ADD:
			case ADD_BROADCAST:
			case EXP:
			case INVERT:
			case LOG:
			case MULTIPLY:
			case NEGATE:
			case POWER:
			case POWER2:
			case SUBTRACT:
			case SIGMOID:
			case TANH:
				return getShape(arguments.get(node)[0]); // Get left arg.
			case MATRIXMULTIPLY:
				//leftShape.getColumns() == rightShape.getRows()
				return new Dimension(getShape(arguments.get(node)[1]).getColumns(), getShape(arguments.get(node)[0]).getRows());
			case TRANSPOSE:
				return new Dimension(getShape(arguments.get(node)[0]).getHeight(), getShape(arguments.get(node)[0]).getWidth());
			case TRACE:
				Dimension childArgs = getShape(arguments.get(node)[0]); // Flatten to 1D.
				return new Dimension(Math.min(childArgs.getWidth(), childArgs.getHeight()),1);
			case INPUT:
				return shapes.get(node); // Assume we've already inserted it.
			default:
				throw new RuntimeException();
		}
	}

	public void setShape(int node, Dimension dim) {
		this.shapes.set(node, dim);
	}

	public abstract float[] getOutput(HashMap<Integer, float[]> inputs, int node);

	public abstract float[][] getGradient(HashMap<Integer, float[]> inputs, int node);
}
