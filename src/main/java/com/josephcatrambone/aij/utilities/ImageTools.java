package com.josephcatrambone.aij.utilities;

import com.josephcatrambone.aij.Matrix;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.SnapshotParameters;
import javafx.scene.image.*;
import javafx.scene.paint.Color;

import javax.imageio.ImageIO;
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
		Matrix output = Matrix.zeros(matrix.numRows(), matrix.numColumns()*bitsPerPixel);
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
		Matrix output = Matrix.zeros(matrix.numRows(), matrix.numColumns()/bitsPerPixel);
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
		WritableImage img = new WritableImage(matrix.numColumns(), matrix.numRows());
		PixelWriter pw = img.getPixelWriter();

		for(int y=0; y < matrix.numRows(); y++) {
			for(int x=0; x < matrix.numColumns(); x++) {
				pw.setColor(x, y, Color.gray(matrix.get(y, x)));
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
}
