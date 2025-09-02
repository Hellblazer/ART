package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.*;
import com.hellblazer.art.core.results.MatchResult;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.performance.VectorizedARTAlgorithm;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;
import java.util.concurrent.TimeUnit;

/**
 * High-performance vectorized GaussianART implementation using Java Vector API.
 * 
 * Features:
 * - SIMD-optimized Gaussian probability density calculations
 * - Vectorized incremental mean and covariance updates
 * - Parallel processing for large category sets
 * - Cache-optimized data structures with numerical stability
 * - Performance monitoring and metrics
 * 
 * This implementation maintains full compatibility with GaussianART semantics
 * while providing significant performance improvements through vectorization
 * and parallel processing for multivariate Gaussian distributions.
 */
public class VectorizedGaussianART extends BaseART implements VectorizedARTAlgorithm<VectorizedPerformanceStats, VectorizedGaussianParameters> {
    
    private static final Logger log = LoggerFactory.getLogger(VectorizedGaussianART.class);
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    
    private final ForkJoinPool computePool;
    private final Map<Integer, double[]> inputCache = new ConcurrentHashMap<>();
    private final VectorizedGaussianParameters defaultParams;
    
    // Performance metrics
    private long totalVectorOperations = 0;
    private long totalParallelTasks = 0;
    private double avgComputeTime = 0.0;
    private long activationCalls = 0;
    private long matchCalls = 0;
    private long learningCalls = 0;
    
    public VectorizedGaussianART(VectorizedGaussianParameters defaultParams) {
        super();
        this.defaultParams = Objects.requireNonNull(defaultParams, "Parameters cannot be null");
        this.computePool = new ForkJoinPool(defaultParams.parallelismLevel());
        
        // Validate parameters for GaussianART
        defaultParams.validateForGaussianART();
        
        log.info("Initialized VectorizedGaussianART with {} parallel threads, vector species: {}, scaling: {:.1f}x", 
                 defaultParams.parallelismLevel(), SPECIES.toString(), defaultParams.getPerformanceScaling());
    }
    
    /**
     * Convert WeightVector to VectorizedGaussianWeight for compatibility with BaseART.
     */
    private VectorizedGaussianWeight convertToVectorizedGaussianWeight(WeightVector weight) {
        if (weight instanceof VectorizedGaussianWeight vWeight) {
            return vWeight;
        }
        
        throw new IllegalArgumentException("Weight must be VectorizedGaussianWeight, got: " + 
                                         weight.getClass().getSimpleName());
    }
    
    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(weight, "Weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof VectorizedGaussianParameters vParams)) {
            throw new IllegalArgumentException("Parameters must be VectorizedGaussianParameters");
        }
        
        // Convert WeightVector to VectorizedGaussianWeight
        VectorizedGaussianWeight vWeight = convertToVectorizedGaussianWeight(weight);
        
        totalVectorOperations++;
        activationCalls++;
        return computeVectorizedActivation(input, vWeight, vParams);
    }
    
    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(weight, "Weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof VectorizedGaussianParameters vParams)) {
            throw new IllegalArgumentException("Parameters must be VectorizedGaussianParameters");
        }
        
        // Convert WeightVector to VectorizedGaussianWeight
        VectorizedGaussianWeight vWeight = convertToVectorizedGaussianWeight(weight);
        
        matchCalls++;
        double probability = vWeight.computeVigilance(input, vParams);
        return probability >= vParams.vigilance() ? 
               new MatchResult.Accepted(probability, vParams.vigilance()) : 
               new MatchResult.Rejected(probability, vParams.vigilance());
    }
    
    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(currentWeight, "Current weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof VectorizedGaussianParameters vParams)) {
            throw new IllegalArgumentException("Parameters must be VectorizedGaussianParameters");
        }
        
        // Convert and update
        VectorizedGaussianWeight vWeight = convertToVectorizedGaussianWeight(currentWeight);
        learningCalls++;
        return vWeight.updateGaussian(input, vParams);
    }
    
    @Override
    protected WeightVector createInitialWeight(Pattern input, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof VectorizedGaussianParameters vParams)) {
            throw new IllegalArgumentException("Parameters must be VectorizedGaussianParameters");
        }
        
        return VectorizedGaussianWeight.fromInput(input, vParams);
    }
    
    /**
     * Vectorized activation computation using SIMD operations for Gaussian probability density.
     * Activation: A_j = p(x | μ_j, Σ_j) (multivariate Gaussian PDF)
     */
    private double computeVectorizedActivation(Pattern input, VectorizedGaussianWeight weight, VectorizedGaussianParameters params) {
        if (input.dimension() != weight.dimension()) {
            throw new IllegalArgumentException("Input dimension must match weight dimension");
        }
        
        if (params.enableSIMD() && input.dimension() >= SPECIES.length()) {
            return computeSIMDActivation(input, weight, params);
        } else {
            return computeStandardActivation(input, weight, params);
        }
    }
    
    /**
     * SIMD-optimized Gaussian activation computation.
     */
    private double computeSIMDActivation(Pattern input, VectorizedGaussianWeight weight, VectorizedGaussianParameters params) {
        return weight.computeProbabilityDensity(input, params);
    }
    
    /**
     * Standard Gaussian activation computation fallback.
     */
    private double computeStandardActivation(Pattern input, VectorizedGaussianWeight weight, VectorizedGaussianParameters params) {
        return weight.computeProbabilityDensity(input, params);
    }
    
    /**
     * Enhanced stepFit with performance optimizations and parallel processing.
     */
    public ActivationResult stepFitEnhanced(Pattern input, VectorizedGaussianParameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        long startTime = System.nanoTime();
        
        try {
            // Use parallel processing for large category sets
            if (getCategoryCount() > 10 && params.parallelismLevel() > 1) {
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
    private ActivationResult parallelStepFit(Pattern input, VectorizedGaussianParameters params) {
        if (getCategoryCount() == 0) {
            return stepFit(input, (Object) params);
        }
        
        var task = new ParallelActivationTask(input, params, 0, getCategoryCount());
        var result = computePool.invoke(task);
        totalParallelTasks++;
        
        // Handle NoMatch result by creating new category in main thread (thread-safe)
        if (result instanceof ActivationResult.NoMatch) {
            var newWeight = createInitialWeight(input, params);
            synchronized(this) {
                categories.add(newWeight);
                int newCategoryIndex = getCategoryCount() - 1;
                return new ActivationResult.Success(newCategoryIndex, 1.0, newWeight);
            }
        }
        
        return result;
    }
    
    /**
     * Parallel activation computation task for Gaussian probability densities.
     */
    private class ParallelActivationTask extends RecursiveTask<ActivationResult> {
        private final Pattern input;
        private final VectorizedGaussianParameters params;
        private final int startIndex;
        private final int endIndex;
        private static final int THRESHOLD = 50; // Smaller threshold for Gaussian computations
        
        ParallelActivationTask(Pattern input, VectorizedGaussianParameters params, int startIndex, int endIndex) {
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
            double maxActivation = Double.NEGATIVE_INFINITY;
            int bestCategory = -1;
            WeightVector bestWeight = null;
            WeightVector updatedWeight = null;
            
            for (int i = startIndex; i < endIndex; i++) {
                var weight = getCategory(i);
                double activation = calculateActivation(input, weight, params);
                
                if (activation > maxActivation) {
                    var vigilanceResult = checkVigilance(input, weight, params);
                    if (vigilanceResult.isAccepted()) {
                        maxActivation = activation;
                        bestCategory = i;
                        bestWeight = weight;
                        updatedWeight = updateWeights(input, weight, params);
                    }
                }
            }
            
            if (bestCategory >= 0) {
                return new ActivationResult.Success(bestCategory, maxActivation, updatedWeight);
            } else {
                // No category passed vigilance - return NoMatch to signal need for new category
                return ActivationResult.NoMatch.instance();
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
        avgComputeTime = 0.0;
        activationCalls = 0;
        matchCalls = 0;
        learningCalls = 0;        activationCalls = 0;
        matchCalls = 0;
        learningCalls = 0;
        log.info("Performance tracking reset");
    }
    
    /**
     * Optimize memory usage by trimming caches.
     */
    public void optimizeMemory() {
        if (inputCache.size() > 1000) { // Reasonable cache size for Gaussian operations
            inputCache.clear();
            log.info("Input cache cleared to optimize memory usage");
        }
    }
    
    // VectorizedARTAlgorithm interface implementation
    
    @Override
    public Object learn(Pattern input, VectorizedGaussianParameters parameters) {
        return stepFitEnhanced(input, parameters);
    }
    
    @Override
    public Object predict(Pattern input, VectorizedGaussianParameters parameters) {
        return stepPredict(input, parameters);
    }
    
    @Override
    public VectorizedGaussianParameters getParameters() {
        return defaultParams;
    }
    
    @Override
    public int getVectorSpeciesLength() {
        return SPECIES.length();
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
                if (!computePool.awaitTermination(5, TimeUnit.SECONDS)) {
                    log.warn("Compute pool did not terminate cleanly");
                }
            }
        } catch (InterruptedException e) {
            computePool.shutdownNow();
            Thread.currentThread().interrupt();
        }
        
        inputCache.clear();
        log.info("VectorizedGaussianART closed and resources cleaned up");
    }
    
    /**
     * Get diagnostic information for debugging.
     */
    public String getDiagnosticInfo() {
        var stats = getPerformanceStats();
        return String.format(
            "VectorizedGaussianART Diagnostics:\n" +
            "  Categories: %d\n" +
            "  Vector Operations: %d\n" +
            "  Parallel Tasks: %d\n" +
            "  Activation Calls: %d\n" +
            "  Match Calls: %d\n" +
            "  Learning Calls: %d\n" +
            "  Avg Compute Time: %.3f ms\n" +
            "  Active Threads: %d\n" +
            "  Cache Size: %d\n" +
            "  Performance Scaling: %.1fx\n" +
            "  Parameters: %s",
            getCategoryCount(),
            stats.totalVectorOperations(),
            stats.totalParallelTasks(),
            stats.activationCalls(),
            stats.matchCalls(),
            stats.learningCalls(),
            stats.avgComputeTimeMs(),
            stats.activeThreads(),
            stats.cacheSize(),
            defaultParams.getPerformanceScaling(),
            defaultParams
        );
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedGaussianART{categories=%d, vectorOps=%d, parallelTasks=%d, avgComputeMs=%.3f}",
                           getCategoryCount(), totalVectorOperations, totalParallelTasks, avgComputeTime);
    }
}