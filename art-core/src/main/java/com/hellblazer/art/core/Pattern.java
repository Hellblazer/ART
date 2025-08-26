package com.hellblazer.art.core;

import com.hellblazer.art.core.utils.DataBounds;

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