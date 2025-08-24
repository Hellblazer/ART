package com.hellblazer.art.core;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable vector interface for ART algorithms.
 * Supports both dense and sparse representations with Vector API optimization.
 * All operations return new instances, preserving immutability.
 */
public sealed interface Vector permits DenseVector {
    
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
     * Normalize this vector using the provided data bounds.
     * Each dimension is normalized as: (value - min) / (max - min)
     * Zero ranges result in 0.0 for that dimension.
     * 
     * @param bounds the data bounds for normalization
     * @return a new normalized vector
     * @throws IllegalArgumentException if dimensions don't match
     * @throws NullPointerException if bounds is null
     */
    Vector normalize(DataBounds bounds);
    
    /**
     * Element-wise minimum with another vector.
     * @param other the other vector
     * @return a new vector with element-wise minimum values
     * @throws IllegalArgumentException if dimensions don't match
     * @throws NullPointerException if other is null
     */
    Vector min(Vector other);
    
    /**
     * Element-wise maximum with another vector.
     * @param other the other vector
     * @return a new vector with element-wise maximum values
     * @throws IllegalArgumentException if dimensions don't match
     * @throws NullPointerException if other is null
     */
    Vector max(Vector other);
    
    /**
     * Scale this vector by a scalar value.
     * @param scalar the scalar multiplier
     * @return a new scaled vector
     */
    Vector scale(double scalar);
    
    /**
     * Create a dense vector from an array of values.
     * The array is copied to ensure immutability.
     * 
     * @param data the array of values
     * @return a new dense vector
     * @throws NullPointerException if data is null
     * @throws IllegalArgumentException if data is empty
     */
    static Vector of(double... data) {
        return new DenseVector(data);
    }
}

/**
 * Dense vector implementation using Vector API for SIMD optimization.
 * Immutable record that copies input data to prevent external modification.
 */
record DenseVector(double[] data) implements Vector {
    
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    
    /**
     * Constructor that validates and copies input data.
     * @param data the array of values (will be copied)
     * @throws NullPointerException if data is null
     * @throws IllegalArgumentException if data is empty
     */
    public DenseVector {
        Objects.requireNonNull(data, "Vector data cannot be null");
        if (data.length == 0) {
            throw new IllegalArgumentException("Vector cannot be empty");
        }
        // Copy the array to ensure immutability
        data = Arrays.copyOf(data, data.length);
    }
    
    @Override
    public double get(int index) {
        if (index < 0 || index >= data.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for vector of size " + data.length);
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
    public Vector normalize(DataBounds bounds) {
        Objects.requireNonNull(bounds, "DataBounds cannot be null");
        if (bounds.dimension() != data.length) {
            throw new IllegalArgumentException("Vector dimension " + data.length + 
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
    public Vector min(Vector other) {
        Objects.requireNonNull(other, "Other vector cannot be null");
        if (!(other instanceof DenseVector denseOther)) {
            throw new IllegalArgumentException("Can only compute min with another DenseVector");
        }
        if (data.length != denseOther.data.length) {
            throw new IllegalArgumentException("Vector dimensions must match: " + 
                data.length + " vs " + denseOther.data.length);
        }
        
        var result = new double[data.length];
        int i = 0;
        
        // Vectorized computation
        for (; i < SPECIES.loopBound(data.length); i += SPECIES.length()) {
            var vecA = DoubleVector.fromArray(SPECIES, data, i);
            var vecB = DoubleVector.fromArray(SPECIES, denseOther.data, i);
            var minVec = vecA.min(vecB);
            minVec.intoArray(result, i);
        }
        
        // Handle remaining elements
        for (; i < data.length; i++) {
            result[i] = Math.min(data[i], denseOther.data[i]);
        }
        
        return new DenseVector(result);
    }
    
    @Override
    public Vector max(Vector other) {
        Objects.requireNonNull(other, "Other vector cannot be null");
        if (!(other instanceof DenseVector denseOther)) {
            throw new IllegalArgumentException("Can only compute max with another DenseVector");
        }
        if (data.length != denseOther.data.length) {
            throw new IllegalArgumentException("Vector dimensions must match: " + 
                data.length + " vs " + denseOther.data.length);
        }
        
        var result = new double[data.length];
        int i = 0;
        
        // Vectorized computation
        for (; i < SPECIES.loopBound(data.length); i += SPECIES.length()) {
            var vecA = DoubleVector.fromArray(SPECIES, data, i);
            var vecB = DoubleVector.fromArray(SPECIES, denseOther.data, i);
            var maxVec = vecA.max(vecB);
            maxVec.intoArray(result, i);
        }
        
        // Handle remaining elements
        for (; i < data.length; i++) {
            result[i] = Math.max(data[i], denseOther.data[i]);
        }
        
        return new DenseVector(result);
    }
    
    @Override
    public Vector scale(double scalar) {
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
        if (!(obj instanceof DenseVector other)) return false;
        return Arrays.equals(data, other.data);
    }
    
    @Override
    public int hashCode() {
        return Arrays.hashCode(data);
    }
    
    @Override
    public String toString() {
        return "DenseVector" + Arrays.toString(data);
    }
}