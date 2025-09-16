package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.*;
import com.hellblazer.art.core.results.MatchResult;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.performance.VectorizedARTAlgorithm;
import com.hellblazer.art.performance.algorithms.VectorizedPerformanceStats;
import com.hellblazer.art.performance.AbstractVectorizedART;

import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Vectorized implementation of Dual Vigilance ART algorithm.
 * 
 * Dual Vigilance ART uses two vigilance thresholds (rhoLower and rhoUpper) to create
 * more selective category formation. Patterns must pass both thresholds to be accepted
 * into a category, providing better discrimination between similar patterns.
 * 
 * Key features:
 * - SIMD-optimized fuzzy ART operations
 * - Dual vigilance threshold evaluation
 * - Parallel category search for large networks
 * - Comprehensive performance tracking
 * - Complement coding support
 * 
 * Algorithm:
 * 1. For each category, compute fuzzy AND intersection
 * 2. Evaluate both lower and upper vigilance thresholds
 * 3. Accept category only if both thresholds are satisfied
 * 4. Update weights using dual vigilance learning rule
 * 
 * @author Claude (Anthropic AI)
 * @version 1.0
 */
public final class VectorizedDualVigilanceART extends AbstractVectorizedART<VectorizedPerformanceStats, VectorizedDualVigilanceParameters> {

    private final Map<Pattern, VectorizedDualVigilanceWeight> inputCache = new ConcurrentHashMap<>();
    
    /**
     * Creates a new VectorizedDualVigilanceART instance.
     */
    public VectorizedDualVigilanceART(VectorizedDualVigilanceParameters defaultParams) {
        super(defaultParams);
    }
    
    // Abstract method implementations

    protected void validateParameters(VectorizedDualVigilanceParameters params) {
        Objects.requireNonNull(params, "Parameters cannot be null");
    }

    @Override
    protected Object performVectorizedLearning(Pattern input, VectorizedDualVigilanceParameters params) {
        // Ensure input is complement coded
        var complementCodedInput = ensureComplementCoded(input);
        return stepFit(complementCodedInput, params);
    }

    @Override
    protected Object performVectorizedPrediction(Pattern input, VectorizedDualVigilanceParameters params) {
        // Ensure input is complement coded
        var complementCodedInput = ensureComplementCoded(input);
        return stepPredict(complementCodedInput, params);
    }

    protected void clearAlgorithmState() {
        inputCache.clear();
    }

    protected void closeAlgorithmResources() {
        // No algorithm-specific resources to close
    }
    
    @Override
    protected VectorizedPerformanceStats createPerformanceStats(
            long vectorOps, long parallelTasks, long activations,
            long matches, long learnings, double avgTime) {
        return new VectorizedPerformanceStats(
            vectorOps,
            parallelTasks,
            avgTime,
            getComputePool().getActiveThreadCount(),
            inputCache.size(),
            getCategoryCount(),
            activations,
            matches,
            learnings
        );
    }
    
    // BaseART method implementations
    
    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, Object parameters) {
        if (!(parameters instanceof VectorizedDualVigilanceParameters params)) {
            throw new IllegalArgumentException("Parameters must be VectorizedDualVigilanceParameters");
        }
        if (!(weight instanceof VectorizedDualVigilanceWeight dualWeight)) {
            throw new IllegalArgumentException("Weight must be VectorizedDualVigilanceWeight");
        }
        
        trackVectorOperation();
        
        // Input is already complement coded from learn/predict methods
        var inputData = convertToFloatArray(input);
        var intersection = dualWeight.computeIntersectionSize(inputData);
        var magnitude = dualWeight.computeMinMagnitude();
        
        // Choice function: T(I,J) = |I ∩ wj| / (α + |wj|)
        return intersection / (params.alpha() + magnitude);
    }
    
    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, Object parameters) {
        if (!(parameters instanceof VectorizedDualVigilanceParameters params)) {
            throw new IllegalArgumentException("Parameters must be VectorizedDualVigilanceParameters");
        }
        if (!(weight instanceof VectorizedDualVigilanceWeight dualWeight)) {
            throw new IllegalArgumentException("Weight must be VectorizedDualVigilanceWeight");
        }
        
        trackVectorOperation();
        trackMatchOperation();
        
        var inputData = convertToFloatArray(input);
        
        // Compute fuzzy AND intersection
        var intersection = dualWeight.computeIntersectionSize(inputData);
        
        // Compute input magnitude (for lower threshold)
        var inputMagnitude = computeInputMagnitude(inputData, params);
        
        // Compute category magnitude (for upper threshold)
        var categoryMagnitude = dualWeight.computeMinMagnitude();
        
        // Lower vigilance check: |I ∩ wj| / |I| >= rhoLower
        var lowerRatio = inputMagnitude > 0 ? intersection / inputMagnitude : 0.0;
        var lowerVigilancePass = lowerRatio >= params.rhoLower();
        
        // Upper vigilance check: |I ∩ wj| / |wj| >= rhoUpper
        var upperRatio = categoryMagnitude > 0 ? intersection / categoryMagnitude : 0.0;
        var upperVigilancePass = upperRatio >= params.rhoUpper();
        
        // Both thresholds must be satisfied
        var accepted = lowerVigilancePass && upperVigilancePass;
        
        return accepted ? 
               new MatchResult.Accepted(Math.min(lowerRatio, upperRatio), Math.min(params.rhoLower(), params.rhoUpper())) : 
               new MatchResult.Rejected(Math.min(lowerRatio, upperRatio), Math.max(params.rhoLower(), params.rhoUpper()));
    }
    
    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, Object parameters) {
        if (!(parameters instanceof VectorizedDualVigilanceParameters params)) {
            throw new IllegalArgumentException("Parameters must be VectorizedDualVigilanceParameters");
        }
        if (!(currentWeight instanceof VectorizedDualVigilanceWeight dualWeight)) {
            throw new IllegalArgumentException("Weight must be VectorizedDualVigilanceWeight");
        }
        
        trackVectorOperation();
        
        var inputData = convertToFloatArray(input);
        
        // Fast learning: new_weight = fuzzy_AND(input, old_weight)
        if (params.beta() >= 1.0) {
            return dualWeight.fuzzyAND(inputData);
        }
        
        // Slow learning with dual vigilance update rule
        return dualWeight.updateLearning(inputData, params.alpha(), params.beta());
    }
    
    @Override
    protected WeightVector createInitialWeight(Pattern input, Object parameters) {
        if (!(parameters instanceof VectorizedDualVigilanceParameters params)) {
            throw new IllegalArgumentException("Parameters must be VectorizedDualVigilanceParameters");
        }
        
        var inputData = convertToFloatArray(input);
        return VectorizedDualVigilanceWeight.createInitial(inputData);
    }
    
    // Helper methods
    
    /**
     * Ensures input pattern is complement coded.
     * 
     * @param input Original input pattern
     * @return Complement coded pattern
     */
    private Pattern ensureComplementCoded(Pattern input) {
        // Get values from pattern
        var dimension = input.dimension();
        var data = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            data[i] = input.get(i);
        }
        
        // For dual vigilance ART, we need consistent dimensions
        // If we have any existing categories, use their dimension to determine if complement coding is needed
        if (getCategoryCount() > 0 && getCategories().size() > 0) {
            var expectedDim = getCategories().get(0).dimension();
            if (dimension == expectedDim) {
                // Already the right dimension
                return input;
            }
            // Need to complement code to match existing categories
            if (dimension * 2 == expectedDim) {
                // Create complement coded version
                var complementCoded = new double[data.length * 2];
                System.arraycopy(data, 0, complementCoded, 0, data.length);
                for (int i = 0; i < data.length; i++) {
                    complementCoded[i + data.length] = 1.0 - data[i];
                }
                return Pattern.of(complementCoded);
            }
        } else {
            // No existing categories - always complement code raw input
            // Check if already complement coded using heuristic
            if (!needsComplementCoding(data)) {
                return input; // Already complement coded
            }
            
            // Create complement coded version
            var complementCoded = new double[data.length * 2];
            System.arraycopy(data, 0, complementCoded, 0, data.length);
            for (int i = 0; i < data.length; i++) {
                complementCoded[i + data.length] = 1.0 - data[i];
            }
            return Pattern.of(complementCoded);
        }
        
        // Fallback - return as is
        return input;
    }
    
    /**
     * Converts Pattern to float array.
     * Assumes input is already complement coded from ensureComplementCoded.
     */
    private float[] convertToFloatArray(Pattern input) {
        var dimension = input.dimension();
        var floatData = new float[dimension];
        
        for (int i = 0; i < dimension; i++) {
            floatData[i] = (float) input.get(i);
        }
        
        return floatData;
    }
    
    /**
     * Determines if input needs complement coding.
     * Assumes complement coding if sum of second half ≈ dimension - sum of first half.
     */
    private boolean needsComplementCoding(double[] data) {
        if (data.length % 2 != 0) {
            return true; // Odd length, needs complement coding
        }
        
        int halfLen = data.length / 2;
        var firstHalfSum = 0.0;
        var secondHalfSum = 0.0;
        
        for (int i = 0; i < halfLen; i++) {
            firstHalfSum += data[i];
            secondHalfSum += data[i + halfLen];
        }
        
        // Check if second half is approximately complement of first half
        var expectedSecondHalf = halfLen - firstHalfSum;
        return Math.abs(secondHalfSum - expectedSecondHalf) > 1e-6;
    }
    
    /**
     * Computes input magnitude for dual vigilance evaluation.
     */
    private double computeInputMagnitude(float[] inputData, VectorizedDualVigilanceParameters params) {
        var magnitude = 0.0;
        var halfLen = inputData.length / 2;
        
        // Sum first half (original values) for input magnitude
        for (int i = 0; i < halfLen; i++) {
            magnitude += inputData[i];
        }
        
        return magnitude;
    }
    
    
    // getVectorSpeciesLength is provided by AbstractVectorizedART
    
    @Override
    public String toString() {
        var stats = getPerformanceStats();
        return String.format("VectorizedDualVigilanceART{categories=%d, vectorOps=%d, " +
                           "avgComputeTime=%.3fms, cacheSize=%d}",
                           getCategoryCount(), stats.totalVectorOperations(),
                           stats.avgComputeTimeMs(), stats.cacheSize());
    }
}