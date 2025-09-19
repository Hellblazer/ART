package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.*;
import com.hellblazer.art.core.results.MatchResult;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.performance.AbstractVectorizedART;
import com.hellblazer.art.performance.VectorizedARTAlgorithm;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;
import java.util.concurrent.TimeUnit;

/**
 * High-performance vectorized ART1 implementation for binary pattern recognition.
 * 
 * VectorizedART1 extends the classic ART1 algorithm with significant performance optimizations
 * while maintaining full semantic compatibility with binary pattern recognition:
 * 
 * Features:
 * - SIMD-optimized binary operations (AND, choice function, vigilance computation)
 * - Parallel processing for large category sets using ForkJoinPool
 * - Cache-optimized data structures with LRU eviction
 * - Comprehensive performance monitoring and metrics
 * - Binary input validation with clear error messages
 * 
 * ART1 Algorithm:
 * - Choice function: T_j = |I ∧ w_j| / (L + |w_j|)
 * - Vigilance criterion: |I ∧ w_j| / |I| >= ρ
 * - Learning rule: new_weight = input AND old_weight (conservative learning)
 * 
 * This implementation is optimized for:
 * - Large binary datasets (thousands to millions of patterns)
 * - High-dimensional binary vectors (hundreds to thousands of dimensions)
 * - Real-time processing requirements with predictable performance
 */
public class VectorizedART1 extends AbstractVectorizedART<VectorizedPerformanceStats, VectorizedART1Parameters> {
    
    private static final Logger log = LoggerFactory.getLogger(VectorizedART1.class);
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    private final Map<Integer, float[]> inputCache = new ConcurrentHashMap<>();
    
    /**
     * Create a VectorizedART1 with default parameters.
     * Uses optimal configuration based on available system resources.
     */
    public VectorizedART1() {
        this(VectorizedART1Parameters.defaultParameters());
    }
    
    /**
     * Create a VectorizedART1 with specific parameters.
     * 
     * @param defaultParams Default parameters for operations
     * @throws NullPointerException if defaultParams is null
     */
    public VectorizedART1(VectorizedART1Parameters defaultParams) {
        super(defaultParams);

        log.info("Initialized VectorizedART1 with {} parallel threads, SIMD: {}, vector species: {}",
                 defaultParams.parallelismLevel(), defaultParams.enableSIMD(), SPECIES.toString());
    }
    
    /**
     * Validate that input pattern contains only binary values (0.0 or 1.0).
     * ART1 is specifically designed for binary pattern recognition.
     * 
     * @param pattern input pattern to validate
     * @throws IllegalArgumentException if pattern contains non-binary values
     */
    private void validateBinaryPattern(Pattern pattern) {
        for (int i = 0; i < pattern.dimension(); i++) {
            var value = pattern.get(i);
            if (value != 0.0 && value != 1.0) {
                throw new IllegalArgumentException(
                    String.format("ART1 requires binary input patterns (0.0 or 1.0), found %.6f at index %d", 
                                 value, i));
            }
        }
    }
    
    /**
     * Validate and cast parameters to VectorizedART1Parameters.
     * 
     * @param parameters Object parameters to validate
     * @return Validated VectorizedART1Parameters
     * @throws IllegalArgumentException if parameters are wrong type
     * @throws NullPointerException if parameters are null
     */
    private VectorizedART1Parameters validateParameters(Object parameters) {
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof VectorizedART1Parameters vecParams)) {
            var actualType = parameters.getClass().getSimpleName();
            throw new IllegalArgumentException(
                String.format("VectorizedART1 requires VectorizedART1Parameters but received: %s", actualType));
        }
        
        return vecParams;
    }
    
    /**
     * Convert and cache WeightVector to VectorizedART1Weight for optimal performance.
     * 
     * @param weight WeightVector to convert
     * @return VectorizedART1Weight instance
     */
    private VectorizedART1Weight convertToVectorizedART1Weight(WeightVector weight) {
        if (weight instanceof VectorizedART1Weight vWeight) {
            return vWeight;
        }
        
        // Convert from any WeightVector implementation
        var weights = new double[weight.dimension()];
        var topDown = new int[weight.dimension()];
        
        for (int i = 0; i < weight.dimension(); i++) {
            weights[i] = weight.get(i);
            topDown[i] = weights[i] != 0.0 ? 1 : 0; // Convert to binary
        }
        
        return new VectorizedART1Weight(weights, topDown, System.currentTimeMillis(), 0);
    }
    
    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, VectorizedART1Parameters parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(weight, "Weight cannot be null");
        
        var vecParams = parameters;
        validateBinaryPattern(input);
        
        var vWeight = convertToVectorizedART1Weight(weight);
        
        trackVectorOperation();
        
        return vWeight.computeActivation(input, vecParams);
    }
    
    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, VectorizedART1Parameters parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(weight, "Weight cannot be null");
        
        var vecParams = parameters;
        validateBinaryPattern(input);
        
        var vWeight = convertToVectorizedART1Weight(weight);
        
        trackMatchOperation();
        
        return vWeight.checkVigilance(input, vecParams);
    }
    
    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, VectorizedART1Parameters parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(currentWeight, "Current weight cannot be null");
        
        var vecParams = parameters;
        validateBinaryPattern(input);
        
        var vWeight = convertToVectorizedART1Weight(currentWeight);
        
        trackVectorOperation();
        
        return vWeight.updateWithLearning(input, vecParams);
    }
    
    @Override
    protected WeightVector createInitialWeight(Pattern input, VectorizedART1Parameters parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        validateBinaryPattern(input);
        
        var vecParams = parameters;
        
        return VectorizedART1Weight.fromInput(input, vecParams);
    }
    
    // AbstractVectorizedART implementation
    
    protected void validateParameters(VectorizedART1Parameters params) {
        Objects.requireNonNull(params, "Parameters cannot be null");
        // Additional ART1-specific validation could go here
    }
    
    protected Object performVectorizedLearning(Pattern input, VectorizedART1Parameters params) {
        return learnEnhanced(input, params);
    }

    protected Object performVectorizedPrediction(Pattern input, VectorizedART1Parameters params) {
        // Implement prediction logic directly to avoid recursion
        if (getCategoryCount() == 0) {
            return com.hellblazer.art.core.results.ActivationResult.NoMatch.instance();
        }
        
        var bestCategory = -1;
        var bestActivation = -1.0;
        
        // Find best matching category using ART1 choice function
        var categories = getCategories();
        for (int i = 0; i < categories.size(); i++) {
            var weight = categories.get(i);
            var activation = calculateActivation(input, weight, params);
            
            if (activation > bestActivation) {
                bestActivation = activation;
                bestCategory = i;
            }
        }
        
        if (bestCategory >= 0) {
            var bestWeight = categories.get(bestCategory);
            return new com.hellblazer.art.core.results.ActivationResult.Success(
                bestCategory, bestActivation, bestWeight
            );
        } else {
            return com.hellblazer.art.core.results.ActivationResult.NoMatch.instance();
        }
    }
    
    protected void clearAlgorithmState() {
        inputCache.clear();
    }
    
    protected void closeAlgorithmResources() {
        inputCache.clear();
    }
    
    /**
     * High-performance learning with parallel processing for large category sets.
     * This method optimizes performance based on current network size and parameters.
     * 
     * @param input Binary input pattern
     * @param parameters VectorizedART1Parameters
     * @return ActivationResult with learning outcome
     */
    public Object learnEnhanced(Pattern input, VectorizedART1Parameters parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        long startTime = System.nanoTime();
        
        try {
            // Use parallel processing for large category sets
            if (getCategoryCount() >= parameters.parallelThreshold()) {
                trackParallelTask();
                return parallelStepFit(input, parameters);
            } else {
                return stepFit(input, parameters);
            }
        } finally {
            updatePerformanceMetrics(startTime);
        }
    }
    
    /**
     * Parallel step fit using ForkJoinPool for enhanced performance on large networks.
     */
    private ActivationResult parallelStepFit(Pattern input, VectorizedART1Parameters parameters) {
        if (getCategoryCount() == 0) {
            return (ActivationResult) stepFit(input, parameters);
        }
        
        var task = new ParallelActivationTask(input, parameters, 0, getCategoryCount());
        return getComputePool().invoke(task);
    }
    
    /**
     * Parallel activation computation task using work-stealing for optimal performance.
     */
    private class ParallelActivationTask extends RecursiveTask<ActivationResult> {
        private final Pattern input;
        private final VectorizedART1Parameters params;
        private final int startIndex;
        private final int endIndex;
        private static final int THRESHOLD = 64; // Optimal threshold for binary patterns
        
        ParallelActivationTask(Pattern input, VectorizedART1Parameters params, int startIndex, int endIndex) {
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
            
            return chooseBestActivation(leftResult, rightResult);
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
                // Create new category if no match found
                var newWeight = createInitialWeight(input, params);
                categories.add(newWeight);
                return new ActivationResult.Success(getCategoryCount() - 1, 1.0, newWeight);
            }
        }
        
        private ActivationResult chooseBestActivation(ActivationResult left, ActivationResult right) {
            if (left instanceof ActivationResult.Success leftSuccess &&
                right instanceof ActivationResult.Success rightSuccess) {
                
                return leftSuccess.activationValue() >= rightSuccess.activationValue() ? left : right;
            } else if (left instanceof ActivationResult.Success) {
                return left;
            } else if (right instanceof ActivationResult.Success) {
                return right;
            } else {
                return ActivationResult.NoMatch.instance();
            }
        }
    }
    
    /**
     * Convert Pattern to cached float array for SIMD operations.
     * Uses LRU cache to avoid repeated conversions.
     */
    private float[] getOrCacheFloatArray(Pattern pattern) {
        int hash = pattern.hashCode();
        
        return inputCache.computeIfAbsent(hash, k -> {
            // Check cache size and evict if necessary
            if (inputCache.size() >= getParameters().maxCacheSize()) {
                evictLRUCacheEntry();
            }
            
            var array = new float[pattern.dimension()];
            for (int i = 0; i < pattern.dimension(); i++) {
                array[i] = (float) pattern.get(i);
            }
            return array;
        });
    }
    
    /**
     * Simple LRU eviction by removing one random entry.
     * More sophisticated LRU could be implemented if needed.
     */
    private void evictLRUCacheEntry() {
        var iterator = inputCache.entrySet().iterator();
        if (iterator.hasNext()) {
            iterator.next();
            iterator.remove();
        }
    }
    
    /**
     * Update performance metrics with thread-safe operations.
     */
    private void updatePerformanceMetrics(long startTime) {
        long elapsed = System.nanoTime() - startTime;
        double elapsedMs = elapsed / 1_000_000.0;
        
        // Simple moving average (could be enhanced with better statistics)
        updateComputeTime(elapsedMs);
    }
    
    /**
     * Optimize memory usage by clearing caches when they exceed thresholds.
     */
    public void optimizeMemory() {
        if (inputCache.size() > getParameters().maxCacheSize() * 0.8) {
            inputCache.clear();
            log.info("Input cache cleared for memory optimization (was {} entries)", inputCache.size());
        }
    }
    
    // Additional helper methods
    
    private Object predictInternal(Pattern input, VectorizedART1Parameters parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        validateBinaryPattern(input);
        
        if (getCategoryCount() == 0) {
            return ActivationResult.NoMatch.instance();
        }
        
        // Find best matching category without learning
        double maxActivation = -1.0;
        int bestCategory = -1;
        WeightVector bestWeight = null;
        
        for (int i = 0; i < getCategoryCount(); i++) {
            var weight = getCategory(i);
            double activation = calculateActivation(input, weight, parameters);
            
            if (activation > maxActivation) {
                var vigilanceResult = checkVigilance(input, weight, parameters);
                if (vigilanceResult.isAccepted()) {
                    maxActivation = activation;
                    bestCategory = i;
                    bestWeight = weight;
                }
            }
        }
        
        if (bestCategory >= 0) {
            return new ActivationResult.Success(bestCategory, maxActivation, bestWeight);
        } else {
            return ActivationResult.NoMatch.instance();
        }
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
    
    
    // getVectorSpeciesLength() is provided as final method by parent class
    
    /**
     * Check if SIMD optimizations are available and enabled.
     * 
     * @return true if SIMD is available and enabled
     */
    public boolean isSIMDEnabled() {
        return getParameters().enableSIMD();
    }
    
    /**
     * Get current compute pool utilization for monitoring.
     * 
     * @return Pool utilization statistics
     */
    public String getComputePoolStats() {
        return String.format("Pool[active=%d, queued=%d, steals=%d, parallelism=%d]",
                           getComputePool().getActiveThreadCount(),
                           getComputePool().getQueuedSubmissionCount(),
                           getComputePool().getStealCount(),
                           getComputePool().getParallelism());
    }
}