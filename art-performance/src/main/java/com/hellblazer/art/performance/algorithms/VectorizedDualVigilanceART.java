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
 * @author Hal Hildebrand
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

    // Removed performVectorizedLearning and performVectorizedPrediction - no longer needed

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
    protected double calculateActivation(Pattern input, WeightVector weight, VectorizedDualVigilanceParameters parameters) {
        var params = parameters;
        if (!(weight instanceof VectorizedDualVigilanceWeight dualWeight)) {
            throw new IllegalArgumentException("Weight must be VectorizedDualVigilanceWeight");
        }
        
        trackVectorOperation();
        
        // Ensure input is complement coded
        var complementCoded = ensureComplementCoded(input);
        var inputData = convertToFloatArray(complementCoded);
        var intersection = dualWeight.computeIntersectionSize(inputData);
        var magnitude = dualWeight.computeMinMagnitude();
        
        // Choice function: T(I,J) = |I ∩ wj| / (α + |wj|)
        return intersection / (params.alpha() + magnitude);
    }
    
    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, VectorizedDualVigilanceParameters parameters) {
        var params = parameters;
        if (!(weight instanceof VectorizedDualVigilanceWeight dualWeight)) {
            throw new IllegalArgumentException("Weight must be VectorizedDualVigilanceWeight");
        }
        
        trackVectorOperation();
        trackMatchOperation();
        
        // Ensure input is complement coded
        var complementCoded = ensureComplementCoded(input);
        var inputData = convertToFloatArray(complementCoded);
        
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
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, VectorizedDualVigilanceParameters parameters) {
        var params = parameters;
        if (!(currentWeight instanceof VectorizedDualVigilanceWeight dualWeight)) {
            throw new IllegalArgumentException("Weight must be VectorizedDualVigilanceWeight");
        }
        
        trackVectorOperation();
        
        // Ensure input is complement coded
        var complementCoded = ensureComplementCoded(input);
        var inputData = convertToFloatArray(complementCoded);
        
        // Fast learning: new_weight = fuzzy_AND(input, old_weight)
        if (params.beta() >= 1.0) {
            return dualWeight.fuzzyAND(inputData);
        }
        
        // Slow learning with dual vigilance update rule
        return dualWeight.updateLearning(inputData, params.alpha(), params.beta());
    }
    
    @Override
    protected WeightVector createInitialWeight(Pattern input, VectorizedDualVigilanceParameters parameters) {
        var params = parameters;
        
        // Ensure input is complement coded before creating weight
        var complementCoded = ensureComplementCoded(input);
        var inputData = convertToFloatArray(complementCoded);
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
        var dimension = input.dimension();
        
        // For dual vigilance ART, we need consistent dimensions
        // If we have any existing categories, use their dimension to determine if complement coding is needed
        if (getCategoryCount() > 0 && getCategories().size() > 0) {
            var expectedDim = getCategories().get(0).dimension();
            if (dimension == expectedDim) {
                // Already the right dimension
                return input;
            }
            if (dimension * 2 == expectedDim) {
                // Input needs complement coding to match existing categories
                return createComplementCodedPattern(input);
            }
            if (dimension == expectedDim / 2) {
                // Input might be non-complement-coded, need to complement code
                return createComplementCodedPattern(input);
            }
            // Dimension mismatch - throw exception instead of silent fallback
            throw new IllegalArgumentException("Input dimension " + dimension + 
                " does not match expected dimension " + expectedDim + " or its half " + (expectedDim/2));
        } else {
            // No existing categories - check if input looks complement coded
            if (dimension % 2 == 0 && looksComplementCoded(input)) {
                return input; // Already complement coded
            }
            // Create complement coded version
            return createComplementCodedPattern(input);
        }
    }
    
    /**
     * Create complement coded pattern from input.
     */
    private Pattern createComplementCodedPattern(Pattern input) {
        var dimension = input.dimension();
        var complementCoded = new double[dimension * 2];
        
        // Copy original values
        for (int i = 0; i < dimension; i++) {
            complementCoded[i] = input.get(i);
        }
        
        // Add complement values
        for (int i = 0; i < dimension; i++) {
            complementCoded[i + dimension] = 1.0 - input.get(i);
        }
        
        return Pattern.of(complementCoded);
    }
    
    /**
     * Check if pattern looks like it's already complement coded.
     */
    private boolean looksComplementCoded(Pattern input) {
        var dimension = input.dimension();
        if (dimension % 2 != 0) {
            return false; // Odd dimension cannot be complement coded
        }
        
        int halfDim = dimension / 2;
        double tolerance = 1e-6;
        
        // Check if second half is approximately complement of first half
        for (int i = 0; i < halfDim; i++) {
            double first = input.get(i);
            double second = input.get(i + halfDim);
            if (Math.abs(first + second - 1.0) > tolerance) {
                return false;
            }
        }
        
        return true;
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
