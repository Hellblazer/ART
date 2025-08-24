package com.hellblazer.art.core;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable pattern interface for ART algorithms.
 * Represents input patterns for neural network processing.
 * Supports both dense and sparse representations with Pattern API optimization.
 * All operations return new instances, preserving immutability.
 */
public sealed interface Pattern permits DenseVector {
    
    /**
     * Get the value at the specified index.
     * @param index the index (0-based)
     * @return the value at the index
     * @throws IndexOutOfBoundsException if index is out of bounds
     */
    double get(int index);
    
    /**
     * Get the dimensionality of this vector.
     * @return the number of dimensions
     */
    int dimension();
    
    /**
     * Calculate the L1 (Manhattan) norm of this vector.
     * @return the L1 norm (sum of absolute values)
     */
    double l1Norm();
    
    /**
     * Calculate the L2 (Euclidean) norm of this vector.
     * @return the L2 norm (square root of sum of squares)
     */
    double l2Norm();
    
    /**
     * Normalize this pattern using the provided data bounds.
     * Each dimension is normalized as: (value - min) / (max - min)
     * Zero ranges result in 0.0 for that dimension.
     * 
     * @param bounds the data bounds for normalization
     * @return a new normalized pattern
     * @throws IllegalArgumentException if dimensions don't match
     * @throws NullPointerException if bounds is null
     */
    Pattern normalize(DataBounds bounds);
    
    /**
     * Element-wise minimum with another pattern.
     * @param other the other pattern
     * @return a new pattern with element-wise minimum values
     * @throws IllegalArgumentException if dimensions don't match
     * @throws NullPointerException if other is null
     */
    Pattern min(Pattern other);
    
    /**
     * Element-wise maximum with another pattern.
     * @param other the other pattern
     * @return a new pattern with element-wise maximum values
     * @throws IllegalArgumentException if dimensions don't match
     * @throws NullPointerException if other is null
     */
    Pattern max(Pattern other);
    
    /**
     * Scale this pattern by a scalar value.
     * @param scalar the scalar multiplier
     * @return a new scaled pattern
     */
    Pattern scale(double scalar);
    
    /**
     * Create a dense pattern from an array of values.
     * The array is copied to ensure immutability.
     * 
     * @param data the array of values
     * @return a new dense pattern
     * @throws NullPointerException if data is null
     * @throws IllegalArgumentException if data is empty
     */
    static Pattern of(double... data) {
        return new DenseVector(data);
    }
}

/**
 * Dense pattern implementation using Pattern API for SIMD optimization.
 * Immutable record that copies input data to prevent external modification.
 */
record DenseVector(double[] data) implements Pattern {
    
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    
    /**
     * Constructor that validates and copies input data.
     * @param data the array of values (will be copied)
     * @throws NullPointerException if data is null
     * @throws IllegalArgumentException if data is empty
     */
    public DenseVector {
        Objects.requireNonNull(data, "Pattern data cannot be null");
        if (data.length == 0) {
            throw new IllegalArgumentException("Pattern cannot be empty");
        }
        // Copy the array to ensure immutability
        data = Arrays.copyOf(data, data.length);
    }
    
    @Override
    public double get(int index) {
        if (index < 0 || index >= data.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for pattern of size " + data.length);
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
        int i = 0;
        
        // Vectorized computation
        for (; i < SPECIES.loopBound(data.length); i += SPECIES.length()) {
            var vec = DoubleVector.fromArray(SPECIES, data, i);
            var abs = vec.abs();
            sum += abs.reduceLanes(VectorOperators.ADD);
        }
        
        // Handle remaining elements
        for (; i < data.length; i++) {
            sum += Math.abs(data[i]);
        }
        
        return sum;
    }
    
    @Override
    public double l2Norm() {
        double sumOfSquares = 0.0;
        int i = 0;
        
        // Vectorized computation
        for (; i < SPECIES.loopBound(data.length); i += SPECIES.length()) {
            var vec = DoubleVector.fromArray(SPECIES, data, i);
            var squares = vec.mul(vec);
            sumOfSquares += squares.reduceLanes(VectorOperators.ADD);
        }
        
        // Handle remaining elements
        for (; i < data.length; i++) {
            sumOfSquares += data[i] * data[i];
        }
        
        return Math.sqrt(sumOfSquares);
    }
    
    @Override
    public Pattern normalize(DataBounds bounds) {
        Objects.requireNonNull(bounds, "DataBounds cannot be null");
        if (bounds.dimension() != data.length) {
            throw new IllegalArgumentException("Pattern dimension " + data.length + 
                " does not match bounds dimension " + bounds.dimension());
        }
        
        var normalized = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            var range = bounds.range(i);
            if (range == 0.0) {
                normalized[i] = 0.0;  // Zero range results in 0
            } else {
                normalized[i] = (data[i] - bounds.min(i)) / range;
            }
        }
        
        return new DenseVector(normalized);
    }
    
    @Override
    public Pattern min(Pattern other) {
        Objects.requireNonNull(other, "Other pattern cannot be null");
        if (!(other instanceof DenseVector(double[] data1))) {
            throw new IllegalArgumentException("Can only compute min with another DenseVector");
        }
        if (data.length != data1.length) {
            throw new IllegalArgumentException("Pattern dimensions must match: " + 
                data.length + " vs " + data1.length);
        }
        
        var result = new double[data.length];
        int i = 0;
        
        // Vectorized computation
        for (; i < SPECIES.loopBound(data.length); i += SPECIES.length()) {
            var vecA = DoubleVector.fromArray(SPECIES, data, i);
            var vecB = DoubleVector.fromArray(SPECIES, data1, i);
            var minVec = vecA.min(vecB);
            minVec.intoArray(result, i);
        }
        
        // Handle remaining elements
        for (; i < data.length; i++) {
            result[i] = Math.min(data[i], data1[i]);
        }
        
        return new DenseVector(result);
    }
    
    @Override
    public Pattern max(Pattern other) {
        Objects.requireNonNull(other, "Other pattern cannot be null");
        if (!(other instanceof DenseVector(double[] data1))) {
            throw new IllegalArgumentException("Can only compute max with another DenseVector");
        }
        if (data.length != data1.length) {
            throw new IllegalArgumentException("Pattern dimensions must match: " + 
                data.length + " vs " + data1.length);
        }
        
        var result = new double[data.length];
        int i = 0;
        
        // Vectorized computation
        for (; i < SPECIES.loopBound(data.length); i += SPECIES.length()) {
            var vecA = DoubleVector.fromArray(SPECIES, data, i);
            var vecB = DoubleVector.fromArray(SPECIES, data1, i);
            var maxVec = vecA.max(vecB);
            maxVec.intoArray(result, i);
        }
        
        // Handle remaining elements
        for (; i < data.length; i++) {
            result[i] = Math.max(data[i], data1[i]);
        }
        
        return new DenseVector(result);
    }
    
    @Override
    public Pattern scale(double scalar) {
        var result = new double[data.length];
        int i = 0;
        
        // Vectorized computation
        for (; i < SPECIES.loopBound(data.length); i += SPECIES.length()) {
            var vec = DoubleVector.fromArray(SPECIES, data, i);
            var scaled = vec.mul(scalar);
            scaled.intoArray(result, i);
        }
        
        // Handle remaining elements
        for (; i < data.length; i++) {
            result[i] = data[i] * scalar;
        }
        
        return new DenseVector(result);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof DenseVector(double[] data1))) return false;
        return Arrays.equals(data, data1);
    }
    
    @Override
    public int hashCode() {
        return Arrays.hashCode(data);
    }
    
    /**
     * Get the underlying array values.
     * @return a copy of the data array
     */
    public double[] values() {
        return Arrays.copyOf(data, data.length);
    }
    
    /**
     * Get dimension for compatibility with WeightVector interface.
     * @return the dimension of this vector
     */
    public int getDimension() {
        return data.length;
    }
    
    @Override
    public String toString() {
        return "DenseVector" + Arrays.toString(data);
    }
}
