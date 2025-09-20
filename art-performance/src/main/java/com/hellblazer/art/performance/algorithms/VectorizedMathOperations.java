package com.hellblazer.art.performance.algorithms;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;
import java.util.Arrays;

/**
 * Vectorized mathematical operations for TopoART algorithm using Java Vector API.
 * Provides SIMD-optimized implementations of core mathematical functions.
 * 
 * This implementation leverages Java's Vector API (incubator) for high-performance
 * parallel computation on supported hardware platforms.
 */
public final class VectorizedMathOperations {
    
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    
    private VectorizedMathOperations() {
        // Utility class - prevent instantiation
    }
    
    /**
     * Apply complement coding to input vector using vectorized operations.
     * Transforms d-dimensional input to 2d-dimensional complement coded vector.
     * 
     * Complement coding: x → [x, 1-x]
     * This is a key preprocessing step in ART algorithms.
     * 
     * @param input the input vector to complement code
     * @return complement coded vector [x, 1-x]
     * @throws NullPointerException if input is null
     * @throws IllegalArgumentException if input is empty
     */
    public static double[] complementCode(double[] input) {
        if (input == null) {
            throw new NullPointerException("Input vector cannot be null");
        }
        if (input.length == 0) {
            throw new IllegalArgumentException("Input vector cannot be empty");
        }
        
        var result = new double[input.length * 2];
        
        // Copy original input to first half
        System.arraycopy(input, 0, result, 0, input.length);
        
        // Vectorized computation of complement (1 - x) for second half
        var ones = DoubleVector.broadcast(SPECIES, 1.0);
        int loopBound = SPECIES.loopBound(input.length);
        
        int i = 0;
        for (; i < loopBound; i += SPECIES.length()) {
            var inputVec = DoubleVector.fromArray(SPECIES, input, i);
            var complement = ones.sub(inputVec);
            complement.intoArray(result, i + input.length);
        }
        
        // Handle remaining elements
        for (; i < input.length; i++) {
            result[i + input.length] = 1.0 - input[i];
        }
        
        return result;
    }
    
    /**
     * Compute component-wise minimum of two vectors using vectorized operations.
     * This is the fuzzy AND operation used in ART algorithms.
     * 
     * @param x first vector
     * @param y second vector  
     * @return component-wise minimum: min(x[i], y[i])
     * @throws NullPointerException if either vector is null
     * @throws IllegalArgumentException if vectors have different lengths
     */
    public static double[] componentWiseMin(double[] x, double[] y) {
        if (x == null || y == null) {
            throw new NullPointerException("Input vectors cannot be null");
        }
        if (x.length != y.length) {
            throw new IllegalArgumentException("Vector lengths must be equal: " + x.length + " vs " + y.length);
        }
        
        var result = new double[x.length];
        int loopBound = SPECIES.loopBound(x.length);
        
        int i = 0;
        for (; i < loopBound; i += SPECIES.length()) {
            var xVec = DoubleVector.fromArray(SPECIES, x, i);
            var yVec = DoubleVector.fromArray(SPECIES, y, i);
            var minVec = xVec.min(yVec);
            minVec.intoArray(result, i);
        }
        
        // Handle remaining elements
        for (; i < x.length; i++) {
            result[i] = Math.min(x[i], y[i]);
        }
        
        return result;
    }
    
    /**
     * Compute L1 norm (Manhattan distance) of a vector using vectorized operations.
     * 
     * @param vector the input vector
     * @return L1 norm: sum of absolute values
     * @throws NullPointerException if vector is null
     */
    public static double l1Norm(double[] vector) {
        if (vector == null) {
            throw new NullPointerException("Vector cannot be null");
        }
        
        int loopBound = SPECIES.loopBound(vector.length);
        var sumVec = DoubleVector.zero(SPECIES);
        
        int i = 0;
        for (; i < loopBound; i += SPECIES.length()) {
            var vec = DoubleVector.fromArray(SPECIES, vector, i);
            var absVec = vec.abs();
            sumVec = sumVec.add(absVec);
        }
        
        double sum = sumVec.reduceLanes(VectorOperators.ADD);
        
        // Handle remaining elements
        for (; i < vector.length; i++) {
            sum += Math.abs(vector[i]);
        }
        
        return sum;
    }
    
    /**
     * Compute TopoART choice function (activation) using vectorized operations.
     * 
     * Choice function: T_j = |x ∧ w_j|₁ / (α + |w_j|₁)
     * where ∧ is component-wise minimum (fuzzy AND)
     * 
     * @param input the input pattern
     * @param weights the neuron weight vector
     * @param alpha choice parameter (prevents division by zero)
     * @return activation value
     * @throws NullPointerException if input or weights are null
     * @throws IllegalArgumentException if vectors have different lengths or alpha < 0
     */
    public static double activation(double[] input, double[] weights, double alpha) {
        if (input == null || weights == null) {
            throw new NullPointerException("Input and weight vectors cannot be null");
        }
        if (input.length != weights.length) {
            throw new IllegalArgumentException("Vector lengths must be equal");
        }
        if (alpha < 0) {
            throw new IllegalArgumentException("Alpha must be non-negative");
        }
        
        var fuzzyAnd = componentWiseMin(input, weights);
        double numerator = l1Norm(fuzzyAnd);
        double denominator = alpha + l1Norm(weights);
        
        return numerator / denominator;
    }
    
    /**
     * Test vigilance criterion using vectorized operations.
     * 
     * Match function: |x ∧ w_j|₁ / |x|₁ ≥ ρ
     * where ∧ is component-wise minimum, ρ is vigilance parameter
     * 
     * @param input the input pattern
     * @param weights the neuron weight vector
     * @param vigilance the vigilance parameter (0 ≤ ρ ≤ 1)
     * @return true if vigilance criterion is satisfied
     * @throws NullPointerException if input or weights are null
     * @throws IllegalArgumentException if vectors have different lengths or vigilance not in [0,1]
     */
    public static boolean matchFunction(double[] input, double[] weights, double vigilance) {
        if (input == null || weights == null) {
            throw new NullPointerException("Input and weight vectors cannot be null");
        }
        if (input.length != weights.length) {
            throw new IllegalArgumentException("Vector lengths must be equal");
        }
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in range [0, 1]");
        }
        
        var fuzzyAnd = componentWiseMin(input, weights);
        double numerator = l1Norm(fuzzyAnd);
        double denominator = l1Norm(input);
        
        if (denominator == 0.0) {
            return true; // Zero input matches any pattern
        }
        
        double matchValue = numerator / denominator;
        // Temporary debug output
        if (false && input.length == 20) { // Only for TopoART's complement-coded patterns
            System.out.println("DEBUG Match: numerator=" + numerator + ", denominator=" + denominator + 
                              ", match=" + matchValue + ", vigilance=" + vigilance + ", passes=" + (matchValue >= vigilance));
        }
        
        return matchValue >= vigilance;
    }
    
    /**
     * Update neuron weights using fast learning rule with vectorized operations.
     * 
     * Fast learning: w_j^new = x ∧ w_j^old
     * where ∧ is component-wise minimum
     * 
     * @param weights the current weight vector (modified in place)
     * @param input the input pattern
     * @throws NullPointerException if weights or input are null
     * @throws IllegalArgumentException if vectors have different lengths
     */
    public static void fastLearning(double[] weights, double[] input) {
        if (weights == null || input == null) {
            throw new NullPointerException("Weight and input vectors cannot be null");
        }
        if (weights.length != input.length) {
            throw new IllegalArgumentException("Vector lengths must be equal");
        }
        
        int loopBound = SPECIES.loopBound(weights.length);
        
        int i = 0;
        for (; i < loopBound; i += SPECIES.length()) {
            var weightsVec = DoubleVector.fromArray(SPECIES, weights, i);
            var inputVec = DoubleVector.fromArray(SPECIES, input, i);
            var minVec = weightsVec.min(inputVec);
            minVec.intoArray(weights, i);
        }
        
        // Handle remaining elements
        for (; i < weights.length; i++) {
            weights[i] = Math.min(weights[i], input[i]);
        }
    }
    
    /**
     * Update neuron weights using partial learning rule with vectorized operations.
     * 
     * Partial learning: w_j^new = β(x ∧ w_j^old) + (1-β)w_j^old
     * where β is learning rate, ∧ is component-wise minimum
     * 
     * @param weights the current weight vector (modified in place)
     * @param input the input pattern
     * @param learningRate the learning rate β ∈ (0, 1]
     * @throws NullPointerException if weights or input are null
     * @throws IllegalArgumentException if vectors have different lengths or invalid learning rate
     */
    public static void partialLearning(double[] weights, double[] input, double learningRate) {
        if (weights == null || input == null) {
            throw new NullPointerException("Weight and input vectors cannot be null");
        }
        if (weights.length != input.length) {
            throw new IllegalArgumentException("Vector lengths must be equal");
        }
        if (learningRate <= 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException("Learning rate must be in range (0, 1]");
        }
        
        var beta = DoubleVector.broadcast(SPECIES, learningRate);
        var oneMinusBeta = DoubleVector.broadcast(SPECIES, 1.0 - learningRate);
        
        int loopBound = SPECIES.loopBound(weights.length);
        
        int i = 0;
        for (; i < loopBound; i += SPECIES.length()) {
            var weightsVec = DoubleVector.fromArray(SPECIES, weights, i);
            var inputVec = DoubleVector.fromArray(SPECIES, input, i);
            
            var fuzzyAndVec = weightsVec.min(inputVec);
            var newWeights = beta.mul(fuzzyAndVec).add(oneMinusBeta.mul(weightsVec));
            
            newWeights.intoArray(weights, i);
        }
        
        // Handle remaining elements
        for (; i < weights.length; i++) {
            double fuzzyAnd = Math.min(weights[i], input[i]);
            weights[i] = learningRate * fuzzyAnd + (1.0 - learningRate) * weights[i];
        }
    }
    
    /**
     * Compute Euclidean distance between two vectors using vectorized operations.
     * Used for topology distance calculations.
     * 
     * @param x first vector
     * @param y second vector
     * @return Euclidean distance
     * @throws NullPointerException if either vector is null
     * @throws IllegalArgumentException if vectors have different lengths
     */
    public static double euclideanDistance(double[] x, double[] y) {
        if (x == null || y == null) {
            throw new NullPointerException("Input vectors cannot be null");
        }
        if (x.length != y.length) {
            throw new IllegalArgumentException("Vector lengths must be equal");
        }
        
        int loopBound = SPECIES.loopBound(x.length);
        var sumVec = DoubleVector.zero(SPECIES);
        
        int i = 0;
        for (; i < loopBound; i += SPECIES.length()) {
            var xVec = DoubleVector.fromArray(SPECIES, x, i);
            var yVec = DoubleVector.fromArray(SPECIES, y, i);
            var diff = xVec.sub(yVec);
            var squared = diff.mul(diff);
            sumVec = sumVec.add(squared);
        }
        
        double sumSquaredDiffs = sumVec.reduceLanes(VectorOperators.ADD);
        
        // Handle remaining elements
        for (; i < x.length; i++) {
            double diff = x[i] - y[i];
            sumSquaredDiffs += diff * diff;
        }
        
        return Math.sqrt(sumSquaredDiffs);
    }
    
    /**
     * Check if the Vector API is available and supported on this platform.
     * 
     * @return true if vectorized operations will use SIMD instructions
     */
    public static boolean isVectorizedSupported() {
        try {
            // Try to create a vector to test availability
            var testVector = DoubleVector.broadcast(SPECIES, 1.0);
            return testVector.length() > 1; // Should be > 1 if SIMD is available
        } catch (Exception e) {
            return false;
        }
    }
    
    /**
     * Get information about the vector species being used.
     * 
     * @return description of the vector species and capabilities
     */
    public static String getVectorInfo() {
        return String.format("Vector Species: %s, Length: %d, Element Size: %d bits", 
                           SPECIES.toString(), 
                           SPECIES.length(), 
                           SPECIES.elementSize());
    }
}