package com.hellblazer.art.core.utils;

import java.util.Objects;

/**
 * Mathematical operations utility class for TopoART algorithm.
 * Implements all mathematical functions required by the TopoART algorithm
 * as specified in Tscherepanow (2010).
 */
public final class MathOperations {
    
    private MathOperations() {
        // Utility class - prevent instantiation
    }
    
    /**
     * Apply complement coding to input vector.
     * Transforms d-dimensional input to 2d-dimensional complement coded vector.
     * 
     * Complement coding: x → [x, 1-x]
     * For input [x1, x2, ..., xd] produces [x1, x2, ..., xd, 1-x1, 1-x2, ..., 1-xd]
     * 
     * @param input the input vector with values in [0, 1]
     * @return complement coded vector of length 2 * input.length
     * @throws NullPointerException if input is null
     * @throws IllegalArgumentException if input values are outside [0, 1]
     */
    public static double[] complementCode(double[] input) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        
        for (int i = 0; i < input.length; i++) {
            if (input[i] < 0.0 || input[i] > 1.0) {
                throw new IllegalArgumentException(
                    String.format("Input value at index %d is %f, must be in [0, 1]", i, input[i]));
            }
        }
        
        var coded = new double[input.length * 2];
        for (int i = 0; i < input.length; i++) {
            coded[i] = input[i];
            coded[input.length + i] = 1.0 - input[i];
        }
        return coded;
    }
    
    /**
     * Compute component-wise minimum of two vectors.
     * Implements the fuzzy AND operation (∧) used throughout TopoART.
     * 
     * @param a first vector
     * @param b second vector
     * @return vector where result[i] = min(a[i], b[i])
     * @throws NullPointerException if either vector is null
     * @throws IllegalArgumentException if vectors have different lengths
     */
    public static double[] componentWiseMin(double[] a, double[] b) {
        Objects.requireNonNull(a, "First vector cannot be null");
        Objects.requireNonNull(b, "Second vector cannot be null");
        
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                String.format("Vector lengths must match: %d != %d", a.length, b.length));
        }
        
        var result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = Math.min(a[i], b[i]);
        }
        return result;
    }
    
    /**
     * Compute the city block norm (L1 norm) of a vector.
     * 
     * L1 norm: |v|₁ = Σᵢ |vᵢ|
     * 
     * @param vector the input vector
     * @return the L1 norm of the vector
     * @throws NullPointerException if vector is null
     */
    public static double cityBlockNorm(double[] vector) {
        Objects.requireNonNull(vector, "Vector cannot be null");
        
        double sum = 0.0;
        for (double v : vector) {
            sum += Math.abs(v);
        }
        return sum;
    }
    
    /**
     * Compute the activation (choice function) for TopoART.
     * 
     * Choice function: T_j = |x ∧ w_j|₁ / (α + |w_j|₁)
     * Where:
     * - x is the complement-coded input vector
     * - w_j is the weight vector for neuron j
     * - ∧ is the fuzzy AND operation (element-wise minimum)
     * - |·|₁ is the L1 norm (city block norm)
     * - α is the choice parameter (typically small positive value)
     * 
     * @param input the complement-coded input vector
     * @param weights the weight vector for the neuron
     * @param alpha the choice parameter (α ≥ 0)
     * @return the activation value for this neuron
     * @throws NullPointerException if input or weights are null
     * @throws IllegalArgumentException if vectors have different lengths or alpha < 0
     */
    public static double activation(double[] input, double[] weights, double alpha) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(weights, "Weight vector cannot be null");
        
        if (alpha < 0.0) {
            throw new IllegalArgumentException("Alpha must be non-negative, got: " + alpha);
        }
        
        if (input.length != weights.length) {
            throw new IllegalArgumentException(
                String.format("Vector lengths must match: %d != %d", input.length, weights.length));
        }
        
        // Calculate fuzzy intersection: x ∧ w_j
        var intersection = componentWiseMin(input, weights);
        var numerator = cityBlockNorm(intersection);
        var denominator = alpha + cityBlockNorm(weights);
        
        // Avoid division by zero
        if (denominator == 0.0) {
            throw new IllegalStateException("Division by zero in activation function: α + |w_j| = 0");
        }
        
        return numerator / denominator;
    }
    
    /**
     * Test the match function (vigilance criterion) for TopoART.
     * 
     * Match function: |x ∧ w_j|₁ / |x|₁ ≥ ρ
     * Where:
     * - x is the complement-coded input vector
     * - w_j is the weight vector for the neuron
     * - ∧ is the fuzzy AND operation (element-wise minimum)
     * - |·|₁ is the L1 norm
     * - ρ is the vigilance parameter
     * 
     * @param input the complement-coded input vector
     * @param weights the weight vector for the neuron
     * @param vigilance the vigilance parameter (ρ ∈ [0, 1])
     * @return true if the match criterion is satisfied, false otherwise
     * @throws NullPointerException if input or weights are null
     * @throws IllegalArgumentException if vectors have different lengths or vigilance outside [0, 1]
     */
    public static boolean matchFunction(double[] input, double[] weights, double vigilance) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(weights, "Weight vector cannot be null");
        
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in [0, 1], got: " + vigilance);
        }
        
        if (input.length != weights.length) {
            throw new IllegalArgumentException(
                String.format("Vector lengths must match: %d != %d", input.length, weights.length));
        }
        
        // Calculate fuzzy intersection: x ∧ w_j
        var intersection = componentWiseMin(input, weights);
        var intersectionNorm = cityBlockNorm(intersection);
        var inputNorm = cityBlockNorm(input);
        
        // Avoid division by zero
        if (inputNorm == 0.0) {
            throw new IllegalStateException("Division by zero in match function: |x| = 0");
        }
        
        // Match test: |x ∧ w_j|₁ / |x|₁ ≥ ρ
        var matchRatio = intersectionNorm / inputNorm;
        return matchRatio >= vigilance;
    }
    
    /**
     * Calculate the category size for a complement-coded weight vector.
     * 
     * Category size: Σⱼ (1 - w_j^c - w_j)
     * Where w_j^c is the complement portion of the weight vector.
     * For a 2d weight vector [w₁, w₂, w₁^c, w₂^c], the size is:
     * (1 - w₁^c - w₁) + (1 - w₂^c - w₂)
     * 
     * Smaller category size indicates more specific (tighter) categories.
     * 
     * @param weights the complement-coded weight vector (length must be even)
     * @return the category size
     * @throws NullPointerException if weights is null
     * @throws IllegalArgumentException if weights length is odd
     */
    public static double categorySize(double[] weights) {
        Objects.requireNonNull(weights, "Weight vector cannot be null");
        
        if (weights.length % 2 != 0) {
            throw new IllegalArgumentException("Weight vector length must be even for complement coding, got: " + 
                                             weights.length);
        }
        
        int d = weights.length / 2;
        double size = 0.0;
        
        for (int j = 0; j < d; j++) {
            // For each dimension j: (1 - w_j^c - w_j)
            // where w_j^c is at index d+j
            size += 1.0 - weights[d + j] - weights[j];
        }
        
        return size;
    }
    
    /**
     * Update weight vector using fast learning (β = 1.0).
     * 
     * Fast learning rule: w_j^(new) = x ∧ w_j^(old)
     * This is a special case of the general learning rule with β = 1.
     * 
     * @param input the complement-coded input vector
     * @param currentWeights the current weight vector
     * @return the updated weight vector
     * @throws NullPointerException if input or currentWeights are null
     * @throws IllegalArgumentException if vectors have different lengths
     */
    public static double[] fastLearning(double[] input, double[] currentWeights) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(currentWeights, "Current weights cannot be null");
        
        if (input.length != currentWeights.length) {
            throw new IllegalArgumentException(
                String.format("Vector lengths must match: %d != %d", input.length, currentWeights.length));
        }
        
        return componentWiseMin(input, currentWeights);
    }
    
    /**
     * Update weight vector using partial learning with specified learning rate.
     * 
     * Partial learning rule: w_j^(new) = β(x ∧ w_j^(old)) + (1-β)w_j^(old)
     * Where β is the learning rate for second-best matching neurons.
     * 
     * @param input the complement-coded input vector
     * @param currentWeights the current weight vector
     * @param learningRate the learning rate β ∈ [0, 1]
     * @return the updated weight vector
     * @throws NullPointerException if input or currentWeights are null
     * @throws IllegalArgumentException if vectors have different lengths or learningRate outside [0, 1]
     */
    public static double[] partialLearning(double[] input, double[] currentWeights, double learningRate) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(currentWeights, "Current weights cannot be null");
        
        if (learningRate < 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException("Learning rate must be in [0, 1], got: " + learningRate);
        }
        
        if (input.length != currentWeights.length) {
            throw new IllegalArgumentException(
                String.format("Vector lengths must match: %d != %d", input.length, currentWeights.length));
        }
        
        var intersection = componentWiseMin(input, currentWeights);
        var result = new double[currentWeights.length];
        
        for (int i = 0; i < result.length; i++) {
            result[i] = learningRate * intersection[i] + (1.0 - learningRate) * currentWeights[i];
        }
        
        return result;
    }
    
    /**
     * Validate that all values in a vector are within the specified range.
     * 
     * @param vector the vector to validate
     * @param min minimum allowed value (inclusive)
     * @param max maximum allowed value (inclusive)
     * @param vectorName name of the vector for error messages
     * @throws NullPointerException if vector is null
     * @throws IllegalArgumentException if any value is outside the range
     */
    public static void validateRange(double[] vector, double min, double max, String vectorName) {
        Objects.requireNonNull(vector, vectorName + " cannot be null");
        
        for (int i = 0; i < vector.length; i++) {
            if (vector[i] < min || vector[i] > max) {
                throw new IllegalArgumentException(
                    String.format("%s value at index %d is %f, must be in [%f, %f]", 
                                vectorName, i, vector[i], min, max));
            }
        }
    }
}