package com.hellblazer.art.core;

/**
 * Interface for weight vectors in ART algorithms.
 * Different ART variants use different weight vector types with specific update rules.
 * All implementations are immutable and updates return new instances.
 */
public interface WeightVector {
    
    /**
     * Get the value at the specified index.
     * @param index the index (0-based)
     * @return the value at the index
     * @throws IndexOutOfBoundsException if index is out of bounds
     */
    double get(int index);
    
    /**
     * Get the dimensionality of this weight vector.
     * @return the number of dimensions
     */
    int dimension();
    
    /**
     * Calculate the L1 (Manhattan) norm of this weight vector.
     * @return the L1 norm
     */
    double l1Norm();
    
    /**
     * Update this weight vector with a new input sample.
     * The update rule depends on the specific weight vector type.
     * 
     * @param input the input vector
     * @param parameters the algorithm parameters
     * @return a new updated weight vector
     * @throws IllegalArgumentException if input dimension doesn't match
     * @throws NullPointerException if input or parameters is null
     */
    WeightVector update(Pattern input, Object parameters);
}