package com.hellblazer.art.temporal.core;

import org.ejml.simple.SimpleMatrix;

/**
 * Matrix wrapper for linear algebra operations.
 * Uses EJML for efficient computation.
 */
public class Matrix {
    private final SimpleMatrix matrix;

    public Matrix(int rows, int cols) {
        this.matrix = new SimpleMatrix(rows, cols);
    }

    public Matrix(double[][] data) {
        this.matrix = new SimpleMatrix(data);
    }

    private Matrix(SimpleMatrix matrix) {
        this.matrix = matrix;
    }

    public void set(int row, int col, double value) {
        matrix.set(row, col, value);
    }

    public double get(int row, int col) {
        return matrix.get(row, col);
    }

    public int rows() {
        return matrix.getNumRows();
    }

    public int cols() {
        return matrix.getNumCols();
    }

    /**
     * Compute eigenvalues for stability analysis.
     */
    public double[] eigenvalues() {
        var eig = matrix.eig();
        var count = eig.getNumberOfEigenvalues();
        var eigenvalues = new double[count];

        for (int i = 0; i < count; i++) {
            eigenvalues[i] = eig.getEigenvalue(i).real;
        }

        return eigenvalues;
    }

    /**
     * Compute spectral radius (largest absolute eigenvalue).
     */
    public double spectralRadius() {
        var eigenvalues = eigenvalues();
        double maxAbs = 0.0;

        for (double ev : eigenvalues) {
            maxAbs = Math.max(maxAbs, Math.abs(ev));
        }

        return maxAbs;
    }

    /**
     * Matrix multiplication.
     */
    public Matrix multiply(Matrix other) {
        return new Matrix(matrix.mult(other.matrix));
    }

    /**
     * Matrix-vector multiplication.
     */
    public double[] multiply(double[] vector) {
        var v = new SimpleMatrix(vector.length, 1, true, vector);
        var result = matrix.mult(v);
        return result.getDDRM().getData();
    }

    /**
     * Compute matrix norm.
     */
    public double norm() {
        return matrix.normF();
    }

    /**
     * Create identity matrix.
     */
    public static Matrix identity(int size) {
        return new Matrix(SimpleMatrix.identity(size));
    }

    /**
     * Create diagonal matrix.
     */
    public static Matrix diagonal(double[] values) {
        return new Matrix(SimpleMatrix.diag(values));
    }
}