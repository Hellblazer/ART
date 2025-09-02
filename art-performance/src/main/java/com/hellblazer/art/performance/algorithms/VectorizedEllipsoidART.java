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
import java.util.concurrent.RecursiveTask;
import java.util.concurrent.TimeUnit;

/**
 * High-performance vectorized EllipsoidART implementation using Java Vector API.
 * 
 * Features:
 * - SIMD-optimized ellipsoidal distance calculations
 * - Vectorized covariance matrix operations
 * - Parallel processing for large category sets
 * - Adaptive ellipsoid shape control via mu parameter
 * - Performance monitoring and metrics
 * 
 * EllipsoidART represents categories as ellipsoids rather than hyperrectangles
 * or hyperspheres, providing more flexible category boundaries that adapt to
 * data distribution patterns.
 */
public class VectorizedEllipsoidART extends BaseART 
    implements VectorizedARTAlgorithm<VectorizedPerformanceStats, VectorizedEllipsoidParameters> {
    
    private static final Logger log = LoggerFactory.getLogger(VectorizedEllipsoidART.class);
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    private final ForkJoinPool computePool;
    private final Map<Integer, float[]> inputCache = new ConcurrentHashMap<>();
    private VectorizedEllipsoidParameters defaultParams;
    
    // Performance metrics
    private long totalVectorOperations = 0;
    private long totalParallelTasks = 0;
    private double avgComputeTime = 0.0;
    private long activationCalls = 0;
    private long matchCalls = 0;
    private long learningCalls = 0;    
    public VectorizedEllipsoidART() {
        this(VectorizedEllipsoidParameters.createDefault());
    }
    
    public VectorizedEllipsoidART(VectorizedEllipsoidParameters defaultParams) {
        super();
        this.defaultParams = Objects.requireNonNull(defaultParams, "Parameters cannot be null");
        this.computePool = new ForkJoinPool(defaultParams.parallelismLevel());
        log.info("Initialized VectorizedEllipsoidART with {} parallel threads, vector species: {}", 
                 defaultParams.parallelismLevel(), SPECIES.toString());
    }
    
    /**
     * Convert WeightVector to VectorizedEllipsoidWeight for compatibility with BaseART.
     */
    private VectorizedEllipsoidWeight convertToVectorizedEllipsoidWeight(WeightVector weight) {
        if (weight instanceof VectorizedEllipsoidWeight ellipsoidWeight) {
            return ellipsoidWeight;
        }
        
        // Create VectorizedEllipsoidWeight from any WeightVector
        // This is a fallback for compatibility - assume spherical covariance
        var center = new double[weight.dimension()];
        var covariance = new double[weight.dimension()][weight.dimension()];
        
        for (int i = 0; i < weight.dimension(); i++) {
            center[i] = weight.get(i);
            // Initialize as identity matrix
            covariance[i][i] = 1.0;
        }
        
        return new VectorizedEllipsoidWeight(center, covariance, 1, System.currentTimeMillis(), 0);
    }
    
    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(weight, "Weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof VectorizedEllipsoidParameters ellipsoidParams)) {
            throw new IllegalArgumentException("Parameters must be VectorizedEllipsoidParameters");
        }
        
        // Convert WeightVector to VectorizedEllipsoidWeight
        VectorizedEllipsoidWeight ellipsoidWeight = convertToVectorizedEllipsoidWeight(weight);
        
        totalVectorOperations++;
        return computeVectorizedActivation(input, ellipsoidWeight, ellipsoidParams);
    }
    
    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(weight, "Weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof VectorizedEllipsoidParameters ellipsoidParams)) {
            throw new IllegalArgumentException("Parameters must be VectorizedEllipsoidParameters");
        }
        
        // Convert WeightVector to VectorizedEllipsoidWeight
        VectorizedEllipsoidWeight ellipsoidWeight = convertToVectorizedEllipsoidWeight(weight);
        
        double similarity = ellipsoidWeight.computeVigilance(input, ellipsoidParams);
        return similarity >= ellipsoidParams.vigilance() ? 
               new MatchResult.Accepted(similarity, ellipsoidParams.vigilance()) : 
               new MatchResult.Rejected(similarity, ellipsoidParams.vigilance());
    }
    
    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(currentWeight, "Current weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof VectorizedEllipsoidParameters ellipsoidParams)) {
            throw new IllegalArgumentException("Parameters must be VectorizedEllipsoidParameters");
        }
        
        // Convert and update
        VectorizedEllipsoidWeight ellipsoidWeight = convertToVectorizedEllipsoidWeight(currentWeight);
        return ellipsoidWeight.updateEllipsoid(input, ellipsoidParams);
    }
    
    @Override
    protected WeightVector createInitialWeight(Pattern input, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof VectorizedEllipsoidParameters ellipsoidParams)) {
            throw new IllegalArgumentException("Parameters must be VectorizedEllipsoidParameters");
        }
        
        return VectorizedEllipsoidWeight.fromInput(input, ellipsoidParams);
    }
    
    /**
     * Vectorized activation computation for EllipsoidART.
     * Activation is based on ellipsoidal distance from category center.
     */
    private double computeVectorizedActivation(Pattern input, VectorizedEllipsoidWeight weight, 
                                             VectorizedEllipsoidParameters params) {
        if (params.enableSIMD() && input.dimension() >= SPECIES.length()) {
            return computeSIMDActivation(input, weight, params);
        } else {
            return computeStandardActivation(input, weight, params);
        }
    }
    
    /**
     * SIMD-optimized activation computation.
     */
    private double computeSIMDActivation(Pattern input, VectorizedEllipsoidWeight weight, 
                                       VectorizedEllipsoidParameters params) {
        var inputArray = convertToFloatArray(input);
        var centerArray = convertToFloatArray(Pattern.of(weight.getCenter()));
        
        double distance = computeSIMDDistance(inputArray, centerArray);
        
        // EllipsoidART activation: higher activation for closer patterns
        // Activation = exp(-distance^2 / (2 * sigma^2))
        double sigma = params.baseRadius();
        double activation = Math.exp(-distance * distance / (2 * sigma * sigma));
        
        // Apply mu parameter for ellipsoid shape influence
        activation *= params.mu();
        
        return Math.max(0.0, Math.min(1.0, activation));
    }
    
    /**
     * Standard activation computation fallback.
     */
    private double computeStandardActivation(Pattern input, VectorizedEllipsoidWeight weight, 
                                           VectorizedEllipsoidParameters params) {
        var center = weight.getCenter();
        double squaredDistance = 0.0;
        
        for (int i = 0; i < input.dimension(); i++) {
            double diff = input.get(i) - center[i];
            squaredDistance += diff * diff;
        }
        
        double distance = Math.sqrt(squaredDistance);
        double sigma = params.baseRadius();
        double activation = Math.exp(-distance * distance / (2 * sigma * sigma));
        
        // Apply mu parameter
        activation *= params.mu();
        
        return Math.max(0.0, Math.min(1.0, activation));
    }
    
    /**
     * SIMD-optimized distance computation.
     */
    private double computeSIMDDistance(float[] inputArray, float[] centerArray) {
        double squaredDistance = 0.0;
        int vectorLength = SPECIES.length();
        int upperBound = SPECIES.loopBound(inputArray.length);
        
        // Vectorized distance computation
        for (int i = 0; i < upperBound; i += vectorLength) {
            var inputVec = FloatVector.fromArray(SPECIES, inputArray, i);
            var centerVec = FloatVector.fromArray(SPECIES, centerArray, i);
            
            var diff = inputVec.sub(centerVec);
            var squaredDiff = diff.mul(diff);
            squaredDistance += squaredDiff.reduceLanes(VectorOperators.ADD);
        }
        
        // Handle remaining elements
        for (int i = upperBound; i < inputArray.length; i++) {
            double diff = inputArray[i] - centerArray[i];
            squaredDistance += diff * diff;
        }
        
        return Math.sqrt(squaredDistance);
    }
    
    /**
     * Convert Pattern to float array with caching.
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
     * Enhanced learning with performance optimizations and parallel processing.
     */
    public ActivationResult learnEnhanced(Pattern input, VectorizedEllipsoidParameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        long startTime = System.nanoTime();
        
        try {
            // Use parallel processing for large category sets
            if (getCategoryCount() > params.parallelThreshold()) {
                return parallelLearn(input, params);
            } else {
                return stepFit(input, params);
            }
        } finally {
            updatePerformanceMetrics(startTime);
        }
    }
    
    /**
     * High-performance parallel learning using ForkJoinPool.
     */
    private ActivationResult parallelLearn(Pattern input, VectorizedEllipsoidParameters params) {
        if (getCategoryCount() == 0) {
            return stepFit(input, params);
        }
        
        var task = new ParallelEllipsoidTask(input, params, 0, getCategoryCount());
        var result = computePool.invoke(task);
        totalParallelTasks++;
        return result;
    }
    
    /**
     * Parallel ellipsoid computation task.
     */
    private class ParallelEllipsoidTask extends RecursiveTask<ActivationResult> {
        private final Pattern input;
        private final VectorizedEllipsoidParameters params;
        private final int startIndex;
        private final int endIndex;
        private static final int THRESHOLD = 100;
        
        ParallelEllipsoidTask(Pattern input, VectorizedEllipsoidParameters params, int startIndex, int endIndex) {
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
            var leftTask = new ParallelEllipsoidTask(input, params, startIndex, mid);
            var rightTask = new ParallelEllipsoidTask(input, params, mid, endIndex);
            
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
     * Optimize memory usage by trimming caches.
     */
    public void optimizeMemory() {
        if (inputCache.size() > defaultParams.maxCacheSize()) {
            inputCache.clear();
            log.info("Input cache cleared to optimize memory usage");
        }
    }
    
    // VectorizedARTAlgorithm interface implementation
    
    @Override
    public Object learn(Pattern input, VectorizedEllipsoidParameters parameters) {
        return learnEnhanced(input, parameters);
    }
    
    @Override
    public Object predict(Pattern input, VectorizedEllipsoidParameters parameters) {
        return stepFit(input, parameters);
    }
    
    @Override
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
    
    @Override
    public void resetPerformanceTracking() {
        inputCache.clear();
        totalVectorOperations = 0;
        totalParallelTasks = 0;
        avgComputeTime = 0.0;
        activationCalls = 0;
        matchCalls = 0;
        learningCalls = 0;        log.info("Performance tracking reset");
    }
    
    @Override
    public VectorizedEllipsoidParameters getParameters() {
        return defaultParams;
    }
    
    public void setParameters(VectorizedEllipsoidParameters parameters) {
        this.defaultParams = Objects.requireNonNull(parameters, "Parameters cannot be null");
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
        if (computePool != null && !computePool.isShutdown()) {
            computePool.shutdown();
            try {
                if (!computePool.awaitTermination(5, TimeUnit.SECONDS)) {
                    computePool.shutdownNow();
                }
            } catch (InterruptedException e) {
                computePool.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
        inputCache.clear();
        log.info("VectorizedEllipsoidART closed and resources cleaned up");
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedEllipsoidART{categories=%d, vectorOps=%d, parallelTasks=%d, " +
                           "avgComputeMs=%.3f, mu=%.3f}",
                           getCategoryCount(), totalVectorOperations, totalParallelTasks, 
                           avgComputeTime, defaultParams != null ? defaultParams.mu() : 0.0);
    }
}