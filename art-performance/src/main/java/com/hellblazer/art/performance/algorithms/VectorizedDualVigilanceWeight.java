package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.WeightVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;
import java.util.Objects;

/**
 * Weight vector implementation for VectorizedDualVigilanceART.
 * 
 * Represents dual vigilance fuzzy ART weights with SIMD-optimized operations.
 * Each weight vector contains fuzzy min/max patterns for dual threshold evaluation.
 * 
 * Weight format: [w_min_1, w_min_2, ..., w_min_n, w_max_1, w_max_2, ..., w_max_n]
 * where w_min represents the minimum learned values and w_max represents maximum learned values.
 * 
 * @author Hal Hildebrand
 * @version 1.0
 */
public final class VectorizedDualVigilanceWeight implements WeightVector {
    
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    private final float[] weights;
    private final int dimension;
    private final double patternCount;
    
    /**
     * Creates a new dual vigilance weight vector.
     * 
     * @param weights The weight array in [min_values, max_values] format
     * @param patternCount Number of patterns used to create this weight
     */
    public VectorizedDualVigilanceWeight(float[] weights, double patternCount) {
        if (weights == null || weights.length == 0) {
            throw new IllegalArgumentException("Weights cannot be null or empty");
        }
        if (weights.length % 2 != 0) {
            throw new IllegalArgumentException("Weight array length must be even (min + max values)");
        }
        if (patternCount < 0) {
            throw new IllegalArgumentException("Pattern count cannot be negative");
        }
        
        this.weights = weights.clone();
        this.dimension = weights.length / 2; // Actual input dimension
        this.patternCount = patternCount;
    }
    
    /**
     * Creates initial weight from input pattern using complement coding.
     * 
     * @param pattern Input pattern (assumes complement coded: [x, 1-x])
     * @return Initial weight vector
     */
    public static VectorizedDualVigilanceWeight createInitial(float[] pattern) {
        if (pattern == null || pattern.length == 0) {
            throw new IllegalArgumentException("Pattern cannot be null or empty");
        }
        if (pattern.length % 2 != 0) {
            throw new IllegalArgumentException("Pattern must be complement coded (even length)");
        }
        
        int inputDim = pattern.length / 2;
        var initialWeights = new float[pattern.length];
        
        // Initialize with pattern values: min = pattern, max = 1.0
        System.arraycopy(pattern, 0, initialWeights, 0, inputDim); // min values
        Arrays.fill(initialWeights, inputDim, pattern.length, 1.0f); // max values = 1.0
        
        return new VectorizedDualVigilanceWeight(initialWeights, 1.0);
    }
    
    /**
     * Computes fuzzy AND operation with dual vigilance consideration.
     * Uses SIMD operations for performance optimization.
     * 
     * @param pattern Input pattern (complement coded)
     * @return Fuzzy AND result
     */
    public VectorizedDualVigilanceWeight fuzzyAND(float[] pattern) {
        if (pattern.length != weights.length) {
            throw new IllegalArgumentException(
                String.format("Pattern dimension %d doesn't match weight dimension %d", 
                    pattern.length, weights.length));
        }
        
        var result = new float[weights.length];
        int inputDim = dimension;
        
        // SIMD-optimized fuzzy AND for min values
        int bound = SPECIES.loopBound(inputDim);
        for (int i = 0; i < bound; i += SPECIES.length()) {
            var weightVec = FloatVector.fromArray(SPECIES, weights, i);
            var patternVec = FloatVector.fromArray(SPECIES, pattern, i);
            var minResult = weightVec.min(patternVec);
            minResult.intoArray(result, i);
        }
        
        // Scalar tail for min values
        for (int i = bound; i < inputDim; i++) {
            result[i] = Math.min(weights[i], pattern[i]);
        }
        
        // SIMD-optimized fuzzy AND for max values (complement part)
        for (int i = 0; i < bound; i += SPECIES.length()) {
            int weightIdx = i + inputDim;
            int patternIdx = i + inputDim;
            
            var weightVec = FloatVector.fromArray(SPECIES, weights, weightIdx);
            var patternVec = FloatVector.fromArray(SPECIES, pattern, patternIdx);
            var minResult = weightVec.min(patternVec);
            minResult.intoArray(result, weightIdx);
        }
        
        // Scalar tail for max values
        for (int i = bound; i < inputDim; i++) {
            result[i + inputDim] = Math.min(weights[i + inputDim], pattern[i + inputDim]);
        }
        
        return new VectorizedDualVigilanceWeight(result, patternCount + 1.0);
    }
    
    /**
     * Computes L1 norm (sum of absolute values) using SIMD operations.
     * 
     * @return L1 norm of the weight vector
     */
    public double computeL1Norm() {
        double sum = 0.0;
        int bound = SPECIES.loopBound(weights.length);
        
        // SIMD computation
        var sumVec = FloatVector.zero(SPECIES);
        for (int i = 0; i < bound; i += SPECIES.length()) {
            var vec = FloatVector.fromArray(SPECIES, weights, i);
            var absVec = vec.abs();
            sumVec = sumVec.add(absVec);
        }
        
        sum += sumVec.reduceLanes(VectorOperators.ADD);
        
        // Scalar tail
        for (int i = bound; i < weights.length; i++) {
            sum += Math.abs(weights[i]);
        }
        
        return sum;
    }
    
    /**
     * Computes intersection size for dual vigilance evaluation.
     * This is the sum of min values only (not the complement part).
     * 
     * @param pattern Input pattern
     * @return Intersection size
     */
    public double computeIntersectionSize(float[] pattern) {
        if (pattern.length != weights.length) {
            throw new IllegalArgumentException("Pattern dimension mismatch");
        }
        
        double intersection = 0.0;
        int bound = SPECIES.loopBound(dimension);
        
        // SIMD computation for min values only
        var sumVec = FloatVector.zero(SPECIES);
        for (int i = 0; i < bound; i += SPECIES.length()) {
            var weightVec = FloatVector.fromArray(SPECIES, weights, i);
            var patternVec = FloatVector.fromArray(SPECIES, pattern, i);
            var minVec = weightVec.min(patternVec);
            sumVec = sumVec.add(minVec);
        }
        
        intersection += sumVec.reduceLanes(VectorOperators.ADD);
        
        // Scalar tail for min values
        for (int i = bound; i < dimension; i++) {
            intersection += Math.min(weights[i], pattern[i]);
        }
        
        return intersection;
    }
    
    /**
     * Computes the magnitude of the min component for dual vigilance.
     * 
     * @return Magnitude of min component
     */
    public double computeMinMagnitude() {
        double magnitude = 0.0;
        int bound = SPECIES.loopBound(dimension);
        
        // SIMD computation for min values only
        var sumVec = FloatVector.zero(SPECIES);
        for (int i = 0; i < bound; i += SPECIES.length()) {
            var vec = FloatVector.fromArray(SPECIES, weights, i);
            sumVec = sumVec.add(vec);
        }
        
        magnitude += sumVec.reduceLanes(VectorOperators.ADD);
        
        // Scalar tail
        for (int i = bound; i < dimension; i++) {
            magnitude += weights[i];
        }
        
        return magnitude;
    }
    
    /**
     * Creates a copy with updated learning using dual vigilance update rule.
     * 
     * @param pattern Input pattern
     * @param alpha Learning rate for min values
     * @param beta Learning rate for max values
     * @return Updated weight vector
     */
    public VectorizedDualVigilanceWeight updateLearning(float[] pattern, double alpha, double beta) {
        if (pattern.length != weights.length) {
            throw new IllegalArgumentException("Pattern dimension mismatch");
        }
        
        var newWeights = new float[weights.length];
        int inputDim = dimension;
        
        // Update min values: w_new = beta * min(w_old, pattern) + (1-beta) * w_old
        int bound = SPECIES.loopBound(inputDim);
        var betaVec = FloatVector.broadcast(SPECIES, (float) beta);
        var oneMinusBetaVec = FloatVector.broadcast(SPECIES, (float) (1.0 - beta));
        
        for (int i = 0; i < bound; i += SPECIES.length()) {
            var weightVec = FloatVector.fromArray(SPECIES, weights, i);
            var patternVec = FloatVector.fromArray(SPECIES, pattern, i);
            var minVec = weightVec.min(patternVec);
            var updateVec = betaVec.mul(minVec).add(oneMinusBetaVec.mul(weightVec));
            updateVec.intoArray(newWeights, i);
        }
        
        // Scalar tail for min values
        for (int i = bound; i < inputDim; i++) {
            var minVal = Math.min(weights[i], pattern[i]);
            newWeights[i] = (float) (beta * minVal + (1.0 - beta) * weights[i]);
        }
        
        // Update max values: w_new = alpha * max(w_old, pattern) + (1-alpha) * w_old
        var alphaVec = FloatVector.broadcast(SPECIES, (float) alpha);
        var oneMinusAlphaVec = FloatVector.broadcast(SPECIES, (float) (1.0 - alpha));
        
        for (int i = 0; i < bound; i += SPECIES.length()) {
            int weightIdx = i + inputDim;
            int patternIdx = i + inputDim;
            
            var weightVec = FloatVector.fromArray(SPECIES, weights, weightIdx);
            var patternVec = FloatVector.fromArray(SPECIES, pattern, patternIdx);
            var maxVec = weightVec.max(patternVec);
            var updateVec = alphaVec.mul(maxVec).add(oneMinusAlphaVec.mul(weightVec));
            updateVec.intoArray(newWeights, weightIdx);
        }
        
        // Scalar tail for max values
        for (int i = bound; i < inputDim; i++) {
            var maxVal = Math.max(weights[i + inputDim], pattern[i + inputDim]);
            newWeights[i + inputDim] = (float) (alpha * maxVal + (1.0 - alpha) * weights[i + inputDim]);
        }
        
        return new VectorizedDualVigilanceWeight(newWeights, patternCount + 1.0);
    }
    
    // WeightVector interface methods
    
    @Override
    public double get(int index) {
        if (index < 0 || index >= weights.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for length " + weights.length);
        }
        return weights[index];
    }
    
    @Override
    public int dimension() {
        return weights.length; // Full weight dimension (min + max)
    }
    
    @Override
    public double l1Norm() {
        return computeL1Norm();
    }
    
    @Override
    public WeightVector update(com.hellblazer.art.core.Pattern input, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof VectorizedDualVigilanceParameters params)) {
            throw new IllegalArgumentException("Parameters must be VectorizedDualVigilanceParameters");
        }
        
        if (input.dimension() != dimension()) {
            throw new IllegalArgumentException("Input dimension must match weight dimension");
        }
        
        // Convert input to float array
        var inputArray = new float[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            inputArray[i] = (float) input.get(i);
        }
        
        // Use dual vigilance learning update
        return updateLearning(inputArray, params.alpha(), params.beta());
    }
    
    public double[] data() {
        var result = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            result[i] = weights[i];
        }
        return result;
    }
    
    // Additional accessor methods
    
    /**
     * Gets the input dimension (half of weight dimension).
     * 
     * @return Input pattern dimension
     */
    public int getInputDimension() {
        return dimension;
    }
    
    /**
     * Gets the raw weight array (copy).
     * 
     * @return Copy of internal weight array
     */
    public float[] getWeights() {
        return weights.clone();
    }
    
    /**
     * Gets the pattern count used to create this weight.
     * 
     * @return Pattern count
     */
    public double getPatternCount() {
        return patternCount;
    }
    
    /**
     * Gets the min component of the weight vector.
     * 
     * @return Array containing min values
     */
    public float[] getMinComponent() {
        return Arrays.copyOfRange(weights, 0, dimension);
    }
    
    /**
     * Gets the max component of the weight vector.
     * 
     * @return Array containing max values
     */
    public float[] getMaxComponent() {
        return Arrays.copyOfRange(weights, dimension, weights.length);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof VectorizedDualVigilanceWeight other)) return false;
        return Arrays.equals(weights, other.weights) &&
               Double.compare(patternCount, other.patternCount) == 0;
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(Arrays.hashCode(weights), patternCount);
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedDualVigilanceWeight{dimension=%d, patternCount=%.1f, " +
                           "minMagnitude=%.3f, l1Norm=%.3f}",
                           dimension, patternCount, computeMinMagnitude(), computeL1Norm());
    }
}
