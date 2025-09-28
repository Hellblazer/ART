package com.hellblazer.art.laminar.core;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Weight matrix for connections in laminar circuits.
 * Supports both dense and sparse representations.
 *
 * @author Hal Hildebrand
 */
public class WeightMatrix implements Serializable {
    private static final long serialVersionUID = 1L;

    private final int rows;
    private final int cols;
    private final double[][] weights;
    private boolean normalized;

    public WeightMatrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.weights = new double[rows][cols];
        this.normalized = false;
    }

    public WeightMatrix(double[][] weights) {
        this.rows = weights.length;
        this.cols = weights[0].length;
        this.weights = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(weights[i], 0, this.weights[i], 0, cols);
        }
        this.normalized = false;
    }

    public int getRows() {
        return rows;
    }

    public int getCols() {
        return cols;
    }

    public double get(int row, int col) {
        return weights[row][col];
    }

    public void set(int row, int col, double value) {
        weights[row][col] = value;
        normalized = false;
    }

    public double[][] getWeights() {
        var copy = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(weights[i], 0, copy[i], 0, cols);
        }
        return copy;
    }

    public void normalize() {
        for (int j = 0; j < cols; j++) {
            double sum = 0.0;
            for (int i = 0; i < rows; i++) {
                sum += weights[i][j];
            }
            if (sum > 0) {
                for (int i = 0; i < rows; i++) {
                    weights[i][j] /= sum;
                }
            }
        }
        normalized = true;
    }

    public boolean isNormalized() {
        return normalized;
    }

    public WeightMatrix transpose() {
        var transposed = new WeightMatrix(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed.weights[j][i] = weights[i][j];
            }
        }
        return transposed;
    }

    public void randomize(double min, double max) {
        var range = max - min;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                weights[i][j] = min + Math.random() * range;
            }
        }
        normalized = false;
    }

    public void clear() {
        for (int i = 0; i < rows; i++) {
            Arrays.fill(weights[i], 0.0);
        }
        normalized = false;
    }

    public WeightMatrix copy() {
        return new WeightMatrix(weights);
    }

    @Override
    public String toString() {
        return String.format("WeightMatrix[%dx%d, normalized=%b]", rows, cols, normalized);
    }
}