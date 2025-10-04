package com.hellblazer.art.laminar.core;

/**
 * Weight matrix for connections between layers.
 *
 * @author Hal Hildebrand
 */
public class WeightMatrix {
    private final double[][] weights;
    private final int rows;
    private final int cols;

    public WeightMatrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.weights = new double[rows][cols];
    }

    public double get(int i, int j) {
        return weights[i][j];
    }

    public void set(int i, int j, double value) {
        weights[i][j] = value;
    }

    public int getRows() {
        return rows;
    }

    public int getCols() {
        return cols;
    }

    public double[][] getData() {
        return weights;
    }
}