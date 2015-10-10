package com.josephcatrambone.aij.utilities;

import com.josephcatrambone.aij.Matrix;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.SnapshotParameters;
import javafx.scene.image.*;
import javafx.scene.paint.Color;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * Created by josephcatrambone on 8/17/15.
 */
public class ImageTools {
	public static Matrix ImageFileToMatrix(String filename, int width, int height) {
		Image img = new Image("file:" + filename, width, height, true, true, false);
		return FXImageToMatrix(img, width, height);
	}

	public static Matrix FXImageToMatrix(Image image, int width, int height) {
		if(width == -1) { width = (int)image.getWidth(); }
		if(height == -1) { height = (int)image.getHeight(); }

		// Return a matrix with the given dimensions, image scaled to the appropriate size.
		// If the aspect ratio of the image is different, fill with black around the edges.
		ImageView imageView = new ImageView();
		imageView.setImage(image);
		imageView.setFitWidth(width);
		imageView.setFitHeight(height);
		imageView.setPreserveRatio(true);
		imageView.setSmooth(true);

		WritableImage scaledImage = new WritableImage(width, height);

		SnapshotParameters parameters = new SnapshotParameters();
		parameters.setFill(Color.TRANSPARENT);
		imageView.snapshot(parameters, scaledImage);

		PixelReader img = scaledImage.getPixelReader();
		Matrix output = new Matrix(height, width);
		for(int y=0; y < height; y++) {
			for(int x=0; x < width; x++) {
				output.set(y, x, img.getColor(x, y).getBrightness());
			}
		}
		return output;
	}

	/*** GreyMatrixToBitMatrix
	 * Convert an image with 8-bit grey values to a matrix with 8x the entries, but with 1/0 values.
	 * @param matrix
	 * @return
	 */
	public static Matrix GreyMatrixToBitMatrix(Matrix matrix, int bitsPerPixel) {
		Matrix output = Matrix.zeros(matrix.numRows(), matrix.numColumns() * bitsPerPixel);
		for(int row=0; row < matrix.numRows(); row++) {
			for(int column=0; column < matrix.numColumns(); column++) {
				int value = (int)((Math.pow(2, bitsPerPixel)-1)*matrix.get(row, column));
				for(int bit=0; bit < bitsPerPixel; bit++) {
					if((value & 0x1) != 0) {
						output.set(row, column*bitsPerPixel + bit, 1.0);
					}
					value = value >> 1;
				}
			}
		}
		return output;
	}

	/*** BitMatrixToGrayMatrix
	 * Takes an mxn matrix and returns an mx(n/8) matrix.
	 * @param matrix
	 * @param threshold The value for a bit to be considered 'high'.  Usually 0.99 is good.
	 * @return
	 */
	public static Matrix BitMatrixToGrayMatrix(Matrix matrix, double threshold, int bitsPerPixel) {
		Matrix output = Matrix.zeros(matrix.numRows(), matrix.numColumns() / bitsPerPixel);
		for(int row=0; row < matrix.numRows(); row++) {
			for(int column=0; column < matrix.numColumns()/bitsPerPixel; column++) {
				double accumulator = 0;
				for(int bit=0; bit < bitsPerPixel; bit++) {
					if(matrix.get(row, column*bitsPerPixel + bit) > threshold) {
						accumulator += Math.pow(2,bit);
					}
				}
				output.set(row, column, accumulator/Math.pow(2, bitsPerPixel));
			}
		}
		return output;
	}

	public static Image MatrixToFXImage(Matrix matrix) {
		return MatrixToFXImage(matrix, true);
	}

	public static Image MatrixToFXImage(Matrix matrix, boolean normalize) {
		WritableImage img = new WritableImage(matrix.numColumns(), matrix.numRows());
		PixelWriter pw = img.getPixelWriter();

		if(normalize) {
			matrix = matrix.clone(); // So we don't tamper.
			matrix.normalize_i();
		}

		for(int y=0; y < matrix.numRows(); y++) {
			for(int x=0; x < matrix.numColumns(); x++) {
				double color = matrix.get(y, x);
				pw.setColor(x, y, Color.gray(color));
			}
		}

		return img;
	}

	public static boolean FXImageToDisk(Image img, String filename) {
		try {
			ImageIO.write(SwingFXUtils.fromFXImage(img, null), "png", new File(filename));
		} catch(IOException ioe) {
			return false;
		}
		return true;
	}

	public static Matrix imageFileToMatrix(String filename, int width, int height) {
		try {
			BufferedImage img = ImageIO.read(new File(filename));
			return AWTImageToMatrix(img, width, height);
		} catch(IOException ioe) {
			return null;
		}
	}

	public static boolean matrixToDiskAsImage(Matrix matrix, String filename) {
		return matrixToDiskAsImage(matrix, filename, true);
	}

	public static boolean matrixToDiskAsImage(Matrix matrix, String filename, boolean normalize) {

		// Normalize contrast to 0-1.
		if(normalize) {
			Matrix preimg = matrix.clone();
			preimg.normalize_i();
			matrix = preimg;
		}

		BufferedImage img = matrixToAWTImage(matrix);
		try {
			ImageIO.write(img, "png", new File(filename));
		} catch(IOException ioe) {
			return false;
		}
		return true;
	}

	public static Matrix AWTImageToMatrix(BufferedImage img, int width, int height) {
		if(width != -1 || height != -1) {
			if(width == -1) { width = img.getWidth(); }
			if(height == -1) { height = img.getHeight(); }
			img = (BufferedImage)img.getScaledInstance(width, height, BufferedImage.SCALE_SMOOTH);
		}

		Matrix matrix = new Matrix(img.getHeight(), img.getWidth());

		for(int y=0; y < img.getHeight(); y++) {
			for(int x=0; x < img.getWidth(); x++) {
				int rgb = img.getRGB(x, y);
				double a = (rgb >> 32 & 0xff)/255.0;
				double r = (rgb >> 16 & 0xff)/255.0;
				double g = (rgb >> 8 & 0xff)/255.0;
				double b = (rgb & 0xff)/255.0;
				double luminance = Math.sqrt(r*r + g*g + b*b);
				matrix.set(y, x, luminance);
			}
		}

		return matrix;
	}

	public static BufferedImage matrixToAWTImage(Matrix matrix) {
		BufferedImage img = new BufferedImage(matrix.numColumns(), matrix.numRows(), BufferedImage.TYPE_BYTE_GRAY);
		for(int y=0; y < matrix.numRows(); y++) {
			for(int x=0; x < matrix.numColumns(); x++) {
				img.setRGB(x, y, (int) (255 * matrix.get(y, x)));
			}
		}
		return img;
	}
}
