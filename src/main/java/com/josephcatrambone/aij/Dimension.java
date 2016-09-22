package com.josephcatrambone.aij;

/**
 * Created by jcatrambone on 9/19/16.
 */
public class Dimension {
	public int width;
	public int height;

	public Dimension() {
		width = 0;
		height = 0;
	}

	public Dimension(int w_c, int h_r) {
		this.width = w_c;
		this.height = h_r;
	}

	public int getRows() {
		return this.height;
	}

	public int getHeight() {
		return this.height;
	}

	public int getColumns() {
		return this.width;
	}

	public int getWidth() {
		return this.width;
	}

	public int size() {
		return this.width*this.height;
	}
}
