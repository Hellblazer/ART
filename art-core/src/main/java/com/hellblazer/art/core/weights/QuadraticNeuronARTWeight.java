/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 * 
 * This file is part of Java ART Neural Networks.
 */
package com.hellblazer.art.core.weights;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import java.util.Arrays;
import java.util.Objects;

/**
 * Weight structure for QuadraticNeuronART algorithm.
 * 
 * Stores the hyper-ellipsoid parameters:
 * - Transformation matrix W (n×n)
 * - Centroid (bias) vector b (n)
 * - Quadratic term s (scalar)
 * 
 * The data is stored as: [matrix elements (n²), centroid (n), s (1)]
 * 
 * @author Hal Hildebrand
 */
public record QuadraticNeuronARTWeight(double[] data, int inputDimension) implements WeightVector {
    
    /**
     * Constructor with validation and defensive copying.
     */
    public QuadraticNeuronARTWeight {
        Objects.requireNonNull(data, "QuadraticNeuronARTWeight data cannot be null");
        if (inputDimension <= 0) {
            throw new IllegalArgumentException("Input dimension must be positive, got: " + inputDimension);
        }
        int expectedSize = inputDimension * inputDimension + inputDimension + 1;
        if (data.length != expectedSize) {
            throw new IllegalArgumentException(
                "Expected data size " + expectedSize + " but got " + data.length
            );
        }
        // Copy array to ensure immutability
        data = Arrays.copyOf(data, data.length);
    }
    
    /**
     * Create a weight from components
     * 
     * @param matrix Transformation matrix (n×n)
     * @param centroid Centroid/bias vector (n)
     * @param s Quadratic term
     * @return New weight instance
     */
    public static QuadraticNeuronARTWeight fromComponents(double[][] matrix, double[] centroid, double s) {
        int dim = centroid.length;
        int dim2 = dim * dim;
        var data = new double[dim2 + dim + 1];
        
        // Pack matrix (row-major order)
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                data[i * dim + j] = matrix[i][j];
            }
        }
        
        // Pack centroid
        System.arraycopy(centroid, 0, data, dim2, dim);
        
        // Pack s
        data[dim2 + dim] = s;
        
        return new QuadraticNeuronARTWeight(data, dim);
    }
    
    /**
     * Get the transformation matrix W
     * 
     * @return Matrix as 2D array
     */
    public double[][] getMatrix() {
        int dim2 = inputDimension * inputDimension;
        var matrix = new double[inputDimension][inputDimension];
        
        for (int i = 0; i < inputDimension; i++) {
            for (int j = 0; j < inputDimension; j++) {
                matrix[i][j] = data[i * inputDimension + j];
            }
        }
        
        return matrix;
    }
    
    /**
     * Get the centroid (bias) vector
     * 
     * @return Centroid vector
     */
    public double[] getCentroid() {
        int dim2 = inputDimension * inputDimension;
        var centroid = new double[inputDimension];
        System.arraycopy(data, dim2, centroid, 0, inputDimension);
        return centroid;
    }
    
    /**
     * Get the quadratic term s
     * 
     * @return Quadratic term value
     */
    public double getS() {
        return data[inputDimension * inputDimension + inputDimension];
    }
    
    @Override
    public double get(int index) {
        if (index < 0 || index >= data.length) {
            throw new IndexOutOfBoundsException("Index out of bounds: " + index);
        }
        return data[index];
    }
    
    @Override
    public int dimension() {
        return data.length;
    }
    
    @Override
    public double l1Norm() {
        double sum = 0.0;
        for (double value : data) {
            sum += Math.abs(value);
        }
        return sum;
    }
    
    @Override
    public WeightVector update(Pattern input, Object parameters) {
        // Updates are handled in the algorithm itself
        throw new UnsupportedOperationException("Updates are handled by QuadraticNeuronART algorithm");
    }
    
    /**
     * Get the raw data array (defensive copy)
     */
    public double[] getData() {
        return Arrays.copyOf(data, data.length);
    }
    
    /**
     * Get the input dimension
     */
    public int getDimension() {
        return inputDimension;
    }
    
    @Override
    public String toString() {
        return String.format("QuadraticNeuronARTWeight{dim=%d, s=%.3f}", inputDimension, getS());
    }
}