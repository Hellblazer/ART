package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.WeightVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;
import java.util.Objects;

/**
 * Weight vector implementation for VectorizedBinaryFuzzyART.
 * 
 * Optimized for binary patterns (values close to 0.0 or 1.0) with SIMD operations.
 * Uses fuzzy min operations with special optimizations for binary values.
 * 
 * Weight format: [w1, w2, ..., wn] where each wi represents the learned minimum
 * values from fuzzy AND operations with input patterns.
 * 
 * Binary optimizations:
 * - Fast path for exact binary values (0.0, 1.0)
 * - SIMD operations with binary-aware algorithms
 * - Reduced precision requirements for binary patterns
 * 
 * @author Claude (Anthropic AI)
 * @version 1.0
 */
public final class VectorizedBinaryFuzzyWeight implements WeightVector {
    
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final float BINARY_ZERO = 0.0f;
    private static final float BINARY_ONE = 1.0f;
    
    private final float[] weights;
    private final double patternCount;
    private final boolean isBinaryOptimized;
    
    /**
     * Creates a new binary fuzzy weight vector.
     * 
     * @param weights The weight values
     * @param patternCount Number of patterns used to create this weight
     * @param isBinaryOptimized Whether this weight uses binary optimizations
     */
    public VectorizedBinaryFuzzyWeight(float[] weights, double patternCount, boolean isBinaryOptimized) {
        if (weights == null || weights.length == 0) {
            throw new IllegalArgumentException("Weights cannot be null or empty");
        }
        if (patternCount < 0) {
            throw new IllegalArgumentException("Pattern count cannot be negative");
        }
        
        this.weights = weights.clone();
        this.patternCount = patternCount;
        this.isBinaryOptimized = isBinaryOptimized;
    }
    
    /**
     * Creates initial weight from input pattern.
     * 
     * @param pattern Input pattern (may be complement coded)
     * @param binaryThreshold Threshold for binary optimization detection
     * @return Initial weight vector
     */
    public static VectorizedBinaryFuzzyWeight createInitial(float[] pattern, double binaryThreshold) {
        if (pattern == null || pattern.length == 0) {
            throw new IllegalArgumentException("Pattern cannot be null or empty");
        }
        
        var initialWeights = pattern.clone();
        var isBinary = detectBinaryPattern(pattern, binaryThreshold);
        
        // Keep the actual pattern values for proper vigilance testing
        // Binary optimization is tracked but doesn't change initial weights
        
        return new VectorizedBinaryFuzzyWeight(initialWeights, 1.0, isBinary);
    }
    
    /**
     * Computes fuzzy AND operation with binary optimization.
     * 
     * @param pattern Input pattern
     * @param enableBinaryOpt Whether to use binary optimizations
     * @return Fuzzy AND result
     */
    public VectorizedBinaryFuzzyWeight fuzzyAND(float[] pattern, boolean enableBinaryOpt) {
        if (pattern.length != weights.length) {
            throw new IllegalArgumentException(
                String.format("Pattern dimension %d doesn't match weight dimension %d", 
                    pattern.length, weights.length));
        }
        
        var result = new float[weights.length];
        var isBinary = isBinaryOptimized && enableBinaryOpt;
        
        if (isBinary && detectBinaryPattern(pattern, 0.1)) {
            // Fast binary path using bitwise-like operations
            computeBinaryFuzzyAND(pattern, result);
        } else {
            // Standard SIMD fuzzy AND
            computeStandardFuzzyAND(pattern, result);
        }
        
        return new VectorizedBinaryFuzzyWeight(result, patternCount + 1.0, isBinary);
    }
    
    /**
     * Computes fuzzy AND with slow learning update.
     * 
     * @param pattern Input pattern
     * @param beta Learning rate parameter
     * @param enableBinaryOpt Whether to use binary optimizations
     * @return Updated weight vector
     */
    public VectorizedBinaryFuzzyWeight slowLearning(float[] pattern, double beta, boolean enableBinaryOpt) {
        if (pattern.length != weights.length) {
            throw new IllegalArgumentException("Pattern dimension mismatch");
        }
        
        var result = new float[weights.length];
        var isBinary = isBinaryOptimized && enableBinaryOpt;
        
        if (isBinary && detectBinaryPattern(pattern, 0.1) && beta == 1.0) {
            // Fast binary path for beta = 1.0
            computeBinaryFuzzyAND(pattern, result);
        } else {
            // Slow learning: w_new = beta * min(w_old, pattern) + (1-beta) * w_old
            computeSlowLearningUpdate(pattern, beta, result);
        }
        
        return new VectorizedBinaryFuzzyWeight(result, patternCount + 1.0, isBinary);
    }
    
    /**
     * Computes L1 norm using SIMD operations with binary optimization.
     * 
     * @return L1 norm of the weight vector
     */
    public double computeL1Norm() {
        if (isBinaryOptimized && allBinaryValues()) {
            // Fast binary sum using simple addition
            return computeBinarySum();
        }
        
        // Standard SIMD L1 norm computation
        double sum = 0.0;
        int bound = SPECIES.loopBound(weights.length);
        
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
     * Computes intersection size (sum of min values) with binary optimization.
     * 
     * @param pattern Input pattern
     * @return Intersection size
     */
    public double computeIntersectionSize(float[] pattern) {
        if (pattern.length != weights.length) {
            throw new IllegalArgumentException("Pattern dimension mismatch");
        }
        
        if (isBinaryOptimized && detectBinaryPattern(pattern, 0.1) && allBinaryValues()) {
            // Fast binary intersection using AND-like operations
            return computeBinaryIntersection(pattern);
        }
        
        // Standard SIMD min-sum computation
        double intersection = 0.0;
        int bound = SPECIES.loopBound(weights.length);
        
        var sumVec = FloatVector.zero(SPECIES);
        for (int i = 0; i < bound; i += SPECIES.length()) {
            var weightVec = FloatVector.fromArray(SPECIES, weights, i);
            var patternVec = FloatVector.fromArray(SPECIES, pattern, i);
            var minVec = weightVec.min(patternVec);
            sumVec = sumVec.add(minVec);
        }
        
        intersection += sumVec.reduceLanes(VectorOperators.ADD);
        
        // Scalar tail
        for (int i = bound; i < weights.length; i++) {
            intersection += Math.min(weights[i], pattern[i]);
        }
        
        return intersection;
    }
    
    // Private helper methods
    
    /**
     * Fast binary fuzzy AND computation.
     */
    private void computeBinaryFuzzyAND(float[] pattern, float[] result) {
        // Binary fuzzy AND: result[i] = min(weight[i], pattern[i])
        // For binary values, this is equivalent to logical AND
        for (int i = 0; i < weights.length; i++) {
            // Fast binary min operation
            if (weights[i] == BINARY_ZERO || pattern[i] == BINARY_ZERO) {
                result[i] = BINARY_ZERO;
            } else {
                result[i] = Math.min(weights[i], pattern[i]);
            }
        }
    }
    
    /**
     * Standard SIMD fuzzy AND computation.
     */
    private void computeStandardFuzzyAND(float[] pattern, float[] result) {
        int bound = SPECIES.loopBound(weights.length);
        
        // SIMD min operation
        for (int i = 0; i < bound; i += SPECIES.length()) {
            var weightVec = FloatVector.fromArray(SPECIES, weights, i);
            var patternVec = FloatVector.fromArray(SPECIES, pattern, i);
            var minResult = weightVec.min(patternVec);
            minResult.intoArray(result, i);
        }
        
        // Scalar tail
        for (int i = bound; i < weights.length; i++) {
            result[i] = Math.min(weights[i], pattern[i]);
        }
    }
    
    /**
     * Slow learning weight update computation.
     */
    private void computeSlowLearningUpdate(float[] pattern, double beta, float[] result) {
        int bound = SPECIES.loopBound(weights.length);
        
        var betaVec = FloatVector.broadcast(SPECIES, (float) beta);
        var oneMinusBetaVec = FloatVector.broadcast(SPECIES, (float) (1.0 - beta));
        
        // SIMD slow learning update
        for (int i = 0; i < bound; i += SPECIES.length()) {
            var weightVec = FloatVector.fromArray(SPECIES, weights, i);
            var patternVec = FloatVector.fromArray(SPECIES, pattern, i);
            var minVec = weightVec.min(patternVec);
            var updateVec = betaVec.mul(minVec).add(oneMinusBetaVec.mul(weightVec));
            updateVec.intoArray(result, i);
        }
        
        // Scalar tail
        for (int i = bound; i < weights.length; i++) {
            var minVal = Math.min(weights[i], pattern[i]);
            result[i] = (float) (beta * minVal + (1.0 - beta) * weights[i]);
        }
    }
    
    /**
     * Fast binary sum computation.
     */
    private double computeBinarySum() {
        double sum = 0.0;
        for (float weight : weights) {
            sum += weight; // No need for abs() since binary values are >= 0
        }
        return sum;
    }
    
    /**
     * Fast binary intersection computation.
     */
    private double computeBinaryIntersection(float[] pattern) {
        double intersection = 0.0;
        for (int i = 0; i < weights.length; i++) {
            // For binary values, min(a,b) = 0 if either is 0, else min(a,b)
            if (weights[i] != BINARY_ZERO && pattern[i] != BINARY_ZERO) {
                intersection += Math.min(weights[i], pattern[i]);
            }
        }
        return intersection;
    }
    
    /**
     * Detects if pattern contains primarily binary values.
     */
    private static boolean detectBinaryPattern(float[] pattern, double threshold) {
        int binaryCount = 0;
        for (float value : pattern) {
            if (Math.abs(value) <= threshold || Math.abs(value - 1.0f) <= threshold) {
                binaryCount++;
            }
        }
        return (double) binaryCount / pattern.length >= 0.8;
    }
    
    /**
     * Checks if all weight values are binary.
     */
    private boolean allBinaryValues() {
        for (float weight : weights) {
            if (weight != BINARY_ZERO && weight != BINARY_ONE) {
                return false;
            }
        }
        return true;
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
        return weights.length;
    }
    
    @Override
    public double l1Norm() {
        return computeL1Norm();
    }
    
    @Override
    public WeightVector update(com.hellblazer.art.core.Pattern input, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof VectorizedBinaryFuzzyParameters params)) {
            throw new IllegalArgumentException("Parameters must be VectorizedBinaryFuzzyParameters");
        }
        
        if (input.dimension() != dimension()) {
            throw new IllegalArgumentException("Input dimension must match weight dimension");
        }
        
        // Convert input to float array
        var inputArray = new float[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            inputArray[i] = (float) input.get(i);
        }
        
        // Use slow learning update with beta parameter
        return slowLearning(inputArray, params.beta(), params.enableSIMD());
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
     * Checks if this weight uses binary optimizations.
     * 
     * @return True if binary optimized
     */
    public boolean isBinaryOptimized() {
        return isBinaryOptimized;
    }
    
    /**
     * Gets the percentage of weights that are exact binary values.
     * 
     * @return Binary percentage [0,1]
     */
    public double getBinaryPercentage() {
        int binaryCount = 0;
        for (float weight : weights) {
            if (weight == BINARY_ZERO || weight == BINARY_ONE) {
                binaryCount++;
            }
        }
        return (double) binaryCount / weights.length;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof VectorizedBinaryFuzzyWeight other)) return false;
        return Arrays.equals(weights, other.weights) &&
               Double.compare(patternCount, other.patternCount) == 0 &&
               isBinaryOptimized == other.isBinaryOptimized;
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(Arrays.hashCode(weights), patternCount, isBinaryOptimized);
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedBinaryFuzzyWeight{dimension=%d, patternCount=%.1f, " +
                           "binaryOptimized=%s, binaryPercentage=%.1f%%, l1Norm=%.3f}",
                           weights.length, patternCount, isBinaryOptimized, 
                           getBinaryPercentage() * 100, computeL1Norm());
    }
}