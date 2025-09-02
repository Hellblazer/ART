package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.*;
import com.hellblazer.art.core.results.MatchResult;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.performance.VectorizedARTAlgorithm;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.RecursiveTask;

/**
 * High-performance vectorized ART2 implementation using Java Vector API.
 * 
 * ART2 is designed for continuous analog input patterns with preprocessing:
 * - Contrast enhancement using theta parameter
 * - Noise suppression using epsilon parameter  
 * - Normalized weight vectors (unit length)
 * - Dot product activation function
 * - Distance-based vigilance criterion
 * - Convex combination learning
 * 
 * Features:
 * - SIMD-optimized vector operations (dot products, distances, updates)
 * - ART2-specific preprocessing with contrast enhancement and noise suppression
 * - Parallel processing for large category sets
 * - Cache-optimized data structures
 * - Performance monitoring and metrics
 * 
 * This implementation maintains full compatibility with ART2 semantics
 * while providing significant performance improvements through vectorization.
 */
public class VectorizedART2 extends BaseART implements VectorizedARTAlgorithm<VectorizedPerformanceStats, VectorizedART2Parameters> {
    
    private static final Logger log = LoggerFactory.getLogger(VectorizedART2.class);
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    private final ForkJoinPool computePool;
    private final Map<Integer, float[]> inputCache = new ConcurrentHashMap<>();
    private final VectorizedART2Parameters defaultParams;
    
    // Performance metrics
    private long totalVectorOperations = 0;
    private long totalParallelTasks = 0;
    private long activationCalls = 0;
    private long matchCalls = 0;
    private long learningCalls = 0;
    private double avgComputeTime = 0.0;    
    public VectorizedART2(VectorizedART2Parameters defaultParams) {
        super();
        this.defaultParams = Objects.requireNonNull(defaultParams, "Parameters cannot be null");
        this.computePool = new ForkJoinPool(defaultParams.parallelismLevel());
        log.info("Initialized VectorizedART2 with {} parallel threads, vector species: {}, theta={}, epsilon={}", 
                 defaultParams.parallelismLevel(), SPECIES.toString(), defaultParams.theta(), defaultParams.epsilon());
    }
    
    /**
     * Convert WeightVector to VectorizedART2Weight for compatibility with BaseART.
     */
    private VectorizedART2Weight convertToVectorizedART2Weight(WeightVector weight) {
        if (weight instanceof VectorizedART2Weight art2Weight) {
            return art2Weight;
        }
        
        // Create VectorizedART2Weight from any WeightVector
        var weights = new double[weight.dimension()];
        for (int i = 0; i < weight.dimension(); i++) {
            weights[i] = weight.get(i);
        }
        
        return new VectorizedART2Weight(weights, System.currentTimeMillis(), 0);
    }
    
    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(weight, "Weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof VectorizedART2Parameters art2Params)) {
            throw new IllegalArgumentException("Parameters must be VectorizedART2Parameters");
        }
        
        // Convert WeightVector to VectorizedART2Weight
        VectorizedART2Weight art2Weight = convertToVectorizedART2Weight(weight);
        
        activationCalls++;
        totalVectorOperations++;
        return art2Weight.computeActivation(input, art2Params);
    }
    
    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(weight, "Weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof VectorizedART2Parameters art2Params)) {
            throw new IllegalArgumentException("Parameters must be VectorizedART2Parameters");
        }
        
        // Convert WeightVector to VectorizedART2Weight
        VectorizedART2Weight art2Weight = convertToVectorizedART2Weight(weight);
        
        matchCalls++;
        totalVectorOperations++; // Track vector operations for vigilance check
        double similarity = art2Weight.computeVigilance(input, art2Params);
        return similarity >= art2Params.vigilance() ? 
               new MatchResult.Accepted(similarity, art2Params.vigilance()) : 
               new MatchResult.Rejected(similarity, art2Params.vigilance());
    }
    
    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(currentWeight, "Current weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof VectorizedART2Parameters art2Params)) {
            throw new IllegalArgumentException("Parameters must be VectorizedART2Parameters");
        }
        
        // Convert and update
        VectorizedART2Weight art2Weight = convertToVectorizedART2Weight(currentWeight);
        learningCalls++;
        totalVectorOperations++; // Track vector operations for weight update
        return art2Weight.updateART2(input, art2Params);
    }
    
    @Override
    protected WeightVector createInitialWeight(Pattern input, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof VectorizedART2Parameters art2Params)) {
            throw new IllegalArgumentException("Parameters must be VectorizedART2Parameters");
        }
        
        // Track vector operations for initial weight creation
        totalVectorOperations++;
        return VectorizedART2Weight.fromInput(input, art2Params);
    }

    /**
     * Enhanced stepFit with performance optimizations and parallel processing.
     */
    public ActivationResult stepFitEnhanced(Pattern input, VectorizedART2Parameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        long startTime = System.nanoTime();
        
        try {
            // Use parallel processing for large category sets
            if (getCategoryCount() > 50) { // Threshold for parallel processing
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
    private ActivationResult parallelStepFit(Pattern input, VectorizedART2Parameters params) {
        if (getCategoryCount() == 0) {
            return stepFit(input, (Object) params);
        }
        
        var task = new ParallelActivationTask(input, params, 0, getCategoryCount());
        var result = computePool.invoke(task);
        totalParallelTasks++;
        return result;
    }
    
    /**
     * Parallel activation computation task for ART2.
     */
    private class ParallelActivationTask extends RecursiveTask<ActivationResult> {
        private final Pattern input;
        private final VectorizedART2Parameters params;
        private final int startIndex;
        private final int endIndex;
        private static final int THRESHOLD = 100;
        
        ParallelActivationTask(Pattern input, VectorizedART2Parameters params, int startIndex, int endIndex) {
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
            getCategoryCount(),
            activationCalls,
            matchCalls,
            learningCalls
        );
    }
    
    /**
     * Clear caches and reset performance counters.
     */
    public void resetPerformanceTracking() {
        inputCache.clear();
        totalVectorOperations = 0;
        totalParallelTasks = 0;
        activationCalls = 0;
        matchCalls = 0;
        learningCalls = 0;
        avgComputeTime = 0.0;
        activationCalls = 0;
        matchCalls = 0;
        learningCalls = 0;        log.info("Performance tracking reset");
    }
    
    /**
     * Optimize memory usage by trimming caches.
     */
    public void optimizeMemory() {
        if (inputCache.size() > 1000) { // Default cache limit
            inputCache.clear();
            log.info("Input cache cleared to optimize memory usage");
        }
    }
    
    // VectorizedARTAlgorithm interface implementation
    
    @Override
    public Object learn(Pattern input, VectorizedART2Parameters parameters) {
        return stepFitEnhanced(input, parameters);
    }
    
    @Override
    public Object predict(Pattern input, VectorizedART2Parameters parameters) {
        return stepFit(input, parameters);
    }
    
    @Override
    public VectorizedART2Parameters getParameters() {
        return defaultParams;
    }
    
    @Override
    public int getVectorSpeciesLength() {
        return SPECIES.length();
    }
    
    @Override
    public boolean isVectorized() {
        return defaultParams.enableSIMD();
    }
    
    @Override
    public String getAlgorithmType() {
        return "VectorizedART2";
    }
    
    /**
     * ART2-specific preprocessing method for external use.
     */
    public Pattern preprocessInput(Pattern input) {
        return VectorizedART2Weight.preprocessART2Input(input, defaultParams);
    }
    
    /**
     * Normalize input to unit length (ART2 requirement).
     */
    public Pattern normalizeInput(Pattern input) {
        return VectorizedART2Weight.normalizeToUnitLength(input);
    }
    
    /**
     * Get contrast enhancement effectiveness for current parameters.
     */
    public double getContrastEnhancement() {
        return defaultParams.theta();
    }
    
    /**
     * Get noise suppression level for current parameters.
     */
    public double getNoiseSupression() {
        return defaultParams.epsilon();
    }
    
    /**
     * Batch processing with ART2 preprocessing.
     */
    @Override
    public Object[] learnBatch(Pattern[] patterns, VectorizedART2Parameters parameters) {
        Objects.requireNonNull(patterns, "Patterns cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (patterns.length == 0) {
            return new Object[0];
        }
        
        // For small batches, use sequential processing
        if (patterns.length < parameters.parallelismLevel() * 10) {
            return VectorizedARTAlgorithm.super.learnBatch(patterns, parameters);
        }
        
        // For large batches, use parallel processing
        return Arrays.stream(patterns)
                .parallel()
                .map(pattern -> learn(pattern, parameters))
                .toArray();
    }
    
    @Override
    public Object[] predictBatch(Pattern[] patterns, VectorizedART2Parameters parameters) {
        Objects.requireNonNull(patterns, "Patterns cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (patterns.length == 0) {
            return new Object[0];
        }
        
        // For small batches, use sequential processing
        if (patterns.length < parameters.parallelismLevel() * 10) {
            return VectorizedARTAlgorithm.super.predictBatch(patterns, parameters);
        }
        
        // For large batches, use parallel processing
        return Arrays.stream(patterns)
                .parallel()
                .map(pattern -> predict(pattern, parameters))
                .toArray();
    }
    
    /**
     * Close and cleanup resources.
     */
    @Override
    public void close() {
        computePool.shutdown();
        try {
            if (!computePool.awaitTermination(5, TimeUnit.SECONDS)) {
                computePool.shutdownNow();
            }
        } catch (InterruptedException e) {
            computePool.shutdownNow();
            Thread.currentThread().interrupt();
        }
        inputCache.clear();
        log.info("VectorizedART2 closed and resources cleaned up");
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedART2{categories=%d, vectorOps=%d, parallelTasks=%d, " +
                           "avgComputeMs=%.3f, theta=%.3f, epsilon=%.3f, simd=%s}",
                           getCategoryCount(), totalVectorOperations, totalParallelTasks, 
                           avgComputeTime, defaultParams.theta(), defaultParams.epsilon(), 
                           defaultParams.enableSIMD());
    }
}