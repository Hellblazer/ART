package com.hellblazer.art.algorithms;

import com.hellblazer.art.core.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

/**
 * High-performance vectorized FuzzyART implementation using Java Vector API.
 * 
 * Features:
 * - SIMD-optimized fuzzy operations (min, max, element-wise arithmetic)
 * - Vectorized complement coding operations
 * - Parallel processing for large category sets
 * - Cache-optimized data structures
 * - Performance monitoring and metrics
 * 
 * This implementation maintains full compatibility with FuzzyART semantics
 * while providing significant performance improvements through vectorization.
 */
public class VectorizedFuzzyART extends BaseART {
    
    private static final Logger log = LoggerFactory.getLogger(VectorizedFuzzyART.class);
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    private final ForkJoinPool computePool;
    private final Map<Integer, float[]> inputCache = new ConcurrentHashMap<>();
    private final VectorizedParameters defaultParams;
    
    // Performance metrics
    private long totalVectorOperations = 0;
    private long totalParallelTasks = 0;
    private double avgComputeTime = 0.0;
    
    public VectorizedFuzzyART(VectorizedParameters defaultParams) {
        super();
        this.defaultParams = Objects.requireNonNull(defaultParams, "Parameters cannot be null");
        this.computePool = new ForkJoinPool(defaultParams.parallelismLevel());
        log.info("Initialized VectorizedFuzzyART with {} parallel threads, vector species: {}", 
                 defaultParams.parallelismLevel(), SPECIES.toString());
    }
    
    /**
     * Convert WeightVector to VectorizedFuzzyWeight for compatibility with BaseART.
     */
    private VectorizedFuzzyWeight convertToVectorizedFuzzyWeight(WeightVector weight) {
        if (weight instanceof VectorizedFuzzyWeight vWeight) {
            return vWeight;
        }
        
        // Create VectorizedFuzzyWeight from any WeightVector
        var weights = new double[weight.dimension()];
        for (int i = 0; i < weight.dimension(); i++) {
            weights[i] = weight.get(i);
        }
        
        // Assume it's complement-coded if dimension is even
        int originalDim = weight.dimension() % 2 == 0 ? weight.dimension() / 2 : weight.dimension();
        return new VectorizedFuzzyWeight(weights, originalDim, System.currentTimeMillis(), 0);
    }
    
    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(weight, "Weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof VectorizedParameters vParams)) {
            throw new IllegalArgumentException("Parameters must be VectorizedParameters");
        }
        
        // Convert WeightVector to VectorizedFuzzyWeight
        VectorizedFuzzyWeight vWeight = convertToVectorizedFuzzyWeight(weight);
        
        totalVectorOperations++;
        return computeVectorizedActivation(input, vWeight, vParams);
    }
    
    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(weight, "Weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof VectorizedParameters vParams)) {
            throw new IllegalArgumentException("Parameters must be VectorizedParameters");
        }
        
        // Convert WeightVector to VectorizedFuzzyWeight
        VectorizedFuzzyWeight vWeight = convertToVectorizedFuzzyWeight(weight);
        
        double similarity = vWeight.computeVigilance(input, vParams);
        return similarity >= vParams.vigilanceThreshold() ? 
               new MatchResult.Accepted(similarity, vParams.vigilanceThreshold()) : 
               new MatchResult.Rejected(similarity, vParams.vigilanceThreshold());
    }
    
    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(currentWeight, "Current weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof VectorizedParameters vParams)) {
            throw new IllegalArgumentException("Parameters must be VectorizedParameters");
        }
        
        // Convert and update
        VectorizedFuzzyWeight vWeight = convertToVectorizedFuzzyWeight(currentWeight);
        return vWeight.updateFuzzy(input, vParams);
    }
    
    @Override
    protected WeightVector createInitialWeight(Pattern input, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof VectorizedParameters vParams)) {
            throw new IllegalArgumentException("Parameters must be VectorizedParameters");
        }
        
        return VectorizedFuzzyWeight.fromInput(input, vParams);
    }
    
    /**
     * Vectorized activation computation using SIMD operations for FuzzyART choice function.
     * Choice function: T_j = |I ∧ w_j| / (α + |w_j|)
     */
    private double computeVectorizedActivation(Pattern input, VectorizedFuzzyWeight weight, VectorizedParameters params) {
        // Get complement-coded input
        var complementCoded = VectorizedFuzzyWeight.getComplementCoded(input);
        var inputArray = convertToFloatArray(complementCoded);
        var weightArray = weight.getCategoryWeights();
        
        if (params.enableSIMD() && inputArray.length >= SPECIES.length()) {
            return computeSIMDActivation(inputArray, weightArray, params);
        } else {
            return computeStandardActivation(complementCoded, Pattern.of(weight.getWeights()), params);
        }
    }
    
    /**
     * SIMD-optimized activation computation.
     */
    private double computeSIMDActivation(float[] inputArray, float[] weightArray, VectorizedParameters params) {
        int dimension = inputArray.length;
        
        double intersectionSum = 0.0;
        double weightSum = 0.0;
        
        int vectorLength = SPECIES.length();
        int upperBound = SPECIES.loopBound(dimension);
        
        // Vectorized loop for fuzzy intersection and weight norm
        for (int i = 0; i < upperBound; i += vectorLength) {
            var inputVec = FloatVector.fromArray(SPECIES, inputArray, i);
            var weightVec = FloatVector.fromArray(SPECIES, weightArray, i);
            
            // Compute fuzzy minimum (intersection): I ∧ w_j
            var intersection = inputVec.min(weightVec);
            intersectionSum += intersection.reduceLanes(VectorOperators.ADD);
            
            // Sum weight values for normalization
            weightSum += weightVec.reduceLanes(VectorOperators.ADD);
        }
        
        // Handle remaining elements
        for (int i = upperBound; i < dimension; i++) {
            double inputVal = inputArray[i];
            double weightVal = weightArray[i];
            intersectionSum += Math.min(inputVal, weightVal);
            weightSum += weightVal;
        }
        
        // FuzzyART choice function: T_j = |I ∧ w_j| / (α + |w_j|)
        return intersectionSum / (params.alpha() + weightSum);
    }
    
    /**
     * Standard activation computation fallback.
     */
    private double computeStandardActivation(Pattern input, Pattern weight, VectorizedParameters params) {
        double intersection = 0.0;
        double weightNorm = 0.0;
        
        for (int i = 0; i < input.dimension(); i++) {
            double inputVal = input.get(i);
            double weightVal = weight.get(i);
            intersection += Math.min(inputVal, weightVal);
            weightNorm += weightVal;
        }
        
        return intersection / (params.alpha() + weightNorm);
    }
    
    /**
     * Convert Pattern to float array with caching to avoid repeated conversions.
     */
    private float[] convertToFloatArray(Pattern pattern) {
        int hash = pattern.hashCode();
        return inputCache.computeIfAbsent(hash, k -> {
            var array = new float[pattern.dimension()];
            for (int i = 0; i < pattern.dimension(); i++) {
                array[i] = (float) pattern.get(i);
            }
            return array;
        });
    }
    
    /**
     * Enhanced stepFit with performance optimizations and parallel processing.
     */
    public ActivationResult stepFitEnhanced(Pattern input, VectorizedParameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        long startTime = System.nanoTime();
        
        try {
            // Use parallel processing for large category sets
            if (getCategoryCount() > params.parallelThreshold()) {
                return parallelStepFit(input, params);
            } else {
                return stepFit(input, (Object) params);
            }
        } finally {
            updatePerformanceMetrics(startTime);
        }
    }
    
    /**
     * High-performance parallel step fit using ForkJoinPool.
     */
    private ActivationResult parallelStepFit(Pattern input, VectorizedParameters params) {
        if (getCategoryCount() == 0) {
            return stepFit(input, (Object) params);
        }
        
        var task = new ParallelActivationTask(input, params, 0, getCategoryCount());
        var result = computePool.invoke(task);
        totalParallelTasks++;
        return result;
    }
    
    /**
     * Parallel activation computation task.
     */
    private class ParallelActivationTask extends RecursiveTask<ActivationResult> {
        private final Pattern input;
        private final VectorizedParameters params;
        private final int startIndex;
        private final int endIndex;
        private static final int THRESHOLD = 100;
        
        ParallelActivationTask(Pattern input, VectorizedParameters params, int startIndex, int endIndex) {
            this.input = input;
            this.params = params;
            this.startIndex = startIndex;
            this.endIndex = endIndex;
        }
        
        @Override
        protected ActivationResult compute() {
            if (endIndex - startIndex <= THRESHOLD) {
                return computeSequentialRange();
            }
            
            int mid = (startIndex + endIndex) / 2;
            var leftTask = new ParallelActivationTask(input, params, startIndex, mid);
            var rightTask = new ParallelActivationTask(input, params, mid, endIndex);
            
            leftTask.fork();
            var rightResult = rightTask.compute();
            var leftResult = leftTask.join();
            
            return chooseBestResult(leftResult, rightResult);
        }
        
        private ActivationResult computeSequentialRange() {
            double maxActivation = -1.0;
            int bestCategory = -1;
            WeightVector bestWeight = null;
            
            for (int i = startIndex; i < endIndex; i++) {
                var weight = getCategory(i);
                double activation = calculateActivation(input, weight, params);
                if (activation > maxActivation) {
                    var vigilanceResult = checkVigilance(input, weight, params);
                    if (vigilanceResult.isAccepted()) {
                        maxActivation = activation;
                        bestCategory = i;
                        bestWeight = weight;
                    }
                }
            }
            
            if (bestCategory >= 0) {
                var updatedWeight = updateWeights(input, bestWeight, params);
                return new ActivationResult.Success(bestCategory, maxActivation, updatedWeight);
            } else {
                // Create new category
                var newWeight = createInitialWeight(input, params);
                return new ActivationResult.Success(getCategoryCount(), 1.0, newWeight);
            }
        }
        
        private ActivationResult chooseBestResult(ActivationResult left, ActivationResult right) {
            if (left instanceof ActivationResult.Success leftSuccess &&
                right instanceof ActivationResult.Success rightSuccess) {
                return leftSuccess.activationValue() >= rightSuccess.activationValue() ? left : right;
            } else if (left instanceof ActivationResult.Success) {
                return left;
            } else if (right instanceof ActivationResult.Success) {
                return right;
            } else {
                return left;
            }
        }
    }
    
    /**
     * Update performance metrics.
     */
    private void updatePerformanceMetrics(long startTime) {
        long elapsed = System.nanoTime() - startTime;
        double elapsedMs = elapsed / 1_000_000.0;
        avgComputeTime = (avgComputeTime + elapsedMs) / 2.0;
    }
    
    /**
     * Get performance statistics.
     */
    public VectorizedPerformanceStats getPerformanceStats() {
        return new VectorizedPerformanceStats(
            totalVectorOperations,
            totalParallelTasks,
            avgComputeTime,
            computePool.getActiveThreadCount(),
            inputCache.size(),
            getCategoryCount()
        );
    }
    
    /**
     * Clear caches and reset performance counters.
     */
    public void resetPerformanceTracking() {
        inputCache.clear();
        totalVectorOperations = 0;
        totalParallelTasks = 0;
        avgComputeTime = 0.0;
        log.info("Performance tracking reset");
    }
    
    /**
     * Optimize memory usage by trimming caches.
     */
    public void optimizeMemory() {
        if (inputCache.size() > defaultParams.maxCacheSize()) {
            inputCache.clear();
            log.info("Input cache cleared to optimize memory usage");
        }
    }
    
    /**
     * Close and cleanup resources.
     */
    public void close() {
        computePool.shutdown();
        inputCache.clear();
        log.info("VectorizedFuzzyART closed and resources cleaned up");
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedFuzzyART{categories=%d, vectorOps=%d, parallelTasks=%d, avgComputeMs=%.3f}",
                           getCategoryCount(), totalVectorOperations, totalParallelTasks, avgComputeTime);
    }
}