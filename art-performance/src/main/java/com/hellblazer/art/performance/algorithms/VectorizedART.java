package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.*;
import com.hellblazer.art.core.results.MatchResult;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.performance.VectorizedARTAlgorithm;
import org.joml.Vector3f;
import org.joml.Vector4f;
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
 * High-performance vectorized ART implementation using:
 * - Java Pattern API for SIMD operations
 * - JOML for optimized 3D/4D vector math
 * - Parallel processing with ForkJoinPool
 * - Memory-efficient data structures
 * - Cache-aware algorithms
 * 
 * This implementation extends BaseART and provides high-performance
 * implementations of the abstract methods using vectorized operations.
 */
public class VectorizedART extends BaseART implements VectorizedARTAlgorithm<VectorizedPerformanceStats, VectorizedParameters> {
    
    private static final Logger log = LoggerFactory.getLogger(VectorizedART.class);
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    // Performance tracking fields
    private final ForkJoinPool computePool;
    private final Map<Integer, Vector3f> vectorCache = new ConcurrentHashMap<>();
    private final VectorizedParameters defaultParams;
    
    // Performance metrics
    private long totalVectorOperations = 0;
    private long totalParallelTasks = 0;
    private double avgComputeTime = 0.0;
    private long activationCalls = 0;
    private long matchCalls = 0;
    private long learningCalls = 0;    
    public VectorizedART(VectorizedParameters defaultParams) {
        super();
        this.defaultParams = Objects.requireNonNull(defaultParams, "Parameters cannot be null");
        this.computePool = new ForkJoinPool(defaultParams.parallelismLevel());
        log.info("Initialized VectorizedART with {} parallel threads, vector species: {}", 
                 defaultParams.parallelismLevel(), SPECIES.toString());
    }
    
    /**
     * Convert WeightVector to VectorizedFuzzyWeight for compatibility with BaseART.
     */
    private VectorizedFuzzyWeight convertToVectorizedFuzzyWeight(WeightVector weight) {
        if (weight instanceof VectorizedFuzzyWeight vWeight) {
            return vWeight;
        }
        
        // Check if the weight is already complement-coded (even dimension)
        if (weight.dimension() % 2 == 0) {
            // Assume it's already complement-coded
            var weights = new double[weight.dimension()];
            for (int i = 0; i < weight.dimension(); i++) {
                weights[i] = weight.get(i);
            }
            int originalDim = weight.dimension() / 2;
            return new VectorizedFuzzyWeight(weights, originalDim, System.currentTimeMillis(), 0);
        } else {
            // Not complement-coded - need to apply complement coding
            var originalWeights = new double[weight.dimension()];
            for (int i = 0; i < weight.dimension(); i++) {
                originalWeights[i] = weight.get(i);
            }
            
            // Create complement-coded weights: [w1, w2, ..., wn, 1-w1, 1-w2, ..., 1-wn]
            var complementWeights = new double[weight.dimension() * 2];
            for (int i = 0; i < weight.dimension(); i++) {
                double w = Math.max(0.0, Math.min(1.0, originalWeights[i])); // clamp to [0,1]
                complementWeights[i] = w;
                complementWeights[weight.dimension() + i] = 1.0 - w;
            }
            
            return new VectorizedFuzzyWeight(complementWeights, weight.dimension(), System.currentTimeMillis(), 0);
        }
    }
    
    @Override
    protected double calculateActivation(com.hellblazer.art.core.Pattern input, WeightVector weight, Object parameters) {
        totalVectorOperations++; // Track each activation calculation as a vector operation
        // Use FuzzyART activation calculation with complement coding
        var complementInput = VectorizedFuzzyWeight.getComplementCoded(input);
        if (!(parameters instanceof VectorizedParameters)) {
            throw new IllegalArgumentException("Parameters must be VectorizedParameters, got: " + 
                (parameters == null ? "null" : parameters.getClass().getName()));
        }
        var params = (VectorizedParameters) parameters;
        
        // Calculate fuzzy intersection |I ∧ w|
        double intersection = 0.0;
        for (int i = 0; i < weight.dimension(); i++) {
            intersection += Math.min(complementInput.get(i), weight.get(i));
        }
        
        // Calculate input norm |I|
        double inputNorm = 0.0;
        for (int i = 0; i < complementInput.dimension(); i++) {
            inputNorm += complementInput.get(i);
        }
        
        return intersection / (params.alpha() + inputNorm);
    }
    
    @Override
    protected MatchResult checkVigilance(com.hellblazer.art.core.Pattern input, WeightVector weight, Object parameters) {
        // Use VectorizedFuzzyWeight's vigilance calculation for consistent behavior
        var fuzzyWeight = convertToVectorizedFuzzyWeight(weight);
        if (!(parameters instanceof VectorizedParameters)) {
            throw new IllegalArgumentException("Parameters must be VectorizedParameters, got: " + 
                (parameters == null ? "null" : parameters.getClass().getName()));
        }
        var params = (VectorizedParameters) parameters;
        double vigilanceValue = fuzzyWeight.computeVigilance(input, params);
        double threshold = params.vigilanceThreshold();
        
        if (vigilanceValue >= threshold) {
            return new MatchResult.Accepted(vigilanceValue, threshold);
        } else {
            return new MatchResult.Rejected(vigilanceValue, threshold);
        }
    }
    
    @Override
    protected WeightVector updateWeights(com.hellblazer.art.core.Pattern input, WeightVector currentWeight, Object parameters) {
        // Use VectorizedFuzzyWeight's update method for consistent behavior
        var fuzzyWeight = convertToVectorizedFuzzyWeight(currentWeight);
        if (!(parameters instanceof VectorizedParameters)) {
            throw new IllegalArgumentException("Parameters must be VectorizedParameters, got: " + 
                (parameters == null ? "null" : parameters.getClass().getName()));
        }
        var params = (VectorizedParameters) parameters;
        
        // Check if input is already complement-coded (matching weight dimension)
        if (input.dimension() == fuzzyWeight.dimension()) {
            // Input is already complement-coded, pass it directly
            return fuzzyWeight.updateFuzzyDirect(input, params);
        } else {
            // Input needs complement coding
            return fuzzyWeight.updateFuzzy(input, params);
        }
    }
    
    @Override
    protected WeightVector createInitialWeight(com.hellblazer.art.core.Pattern input, Object parameters) {
        // Create VectorizedFuzzyWeight using the static factory method for consistency
        if (!(parameters instanceof VectorizedParameters)) {
            throw new IllegalArgumentException("Parameters must be VectorizedParameters, got: " + 
                (parameters == null ? "null" : parameters.getClass().getName()));
        }
        var params = (VectorizedParameters) parameters;
        totalVectorOperations++; // Track vector operation when creating weights
        return VectorizedFuzzyWeight.fromInput(input, params);
    }
    
    /**
     * Enhanced stepFit with performance optimizations and parallel processing.
     */
    public ActivationResult stepFitEnhanced(com.hellblazer.art.core.Pattern input, VectorizedParameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        long startTime = System.nanoTime();
        totalVectorOperations++; // Track each step fit as a vector operation
        
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
    private ActivationResult parallelStepFit(com.hellblazer.art.core.Pattern input, VectorizedParameters params) {
        if (getCategoryCount() == 0) {
            return stepFit(input, (Object) params);
        }
        
        var task = new ParallelActivationTask(input, params, 0, getCategoryCount());
        var result = computePool.invoke(task);
        totalParallelTasks++;
        return result;
    }
    
    /**
     * Vectorized activation computation using Java Pattern API and JOML optimizations.
     */
    private double computeVectorizedActivation(com.hellblazer.art.core.Pattern input, VectorizedFuzzyWeight weight, VectorizedParameters params) {
        if ((input.dimension() == 3 || input.dimension() == 4) && params.enableJOML()) {
            return computeJOMLActivation(input, weight, params);
        } else if (params.enableSIMD()) {
            return computeSIMDActivation(input, weight, params);
        } else {
            return computeStandardActivation(input, weight, params);
        }
    }
    
    /**
     * JOML-optimized activation for 3D/4D vectors.
     */
    private double computeJOMLActivation(com.hellblazer.art.core.Pattern input, VectorizedFuzzyWeight weight, VectorizedParameters params) {
        totalVectorOperations++; // Track JOML vector operation
        if (input.dimension() == 3) {
            var inputVec = getCachedVector3f(input);
            // Convert weight to Vector3f
            var weightVec = new Vector3f((float) weight.get(0), (float) weight.get(1), (float) weight.get(2));
            
            // Compute fuzzy intersection using JOML
            var intersection = new Vector3f();
            intersection.x = Math.min(inputVec.x, weightVec.x);
            intersection.y = Math.min(inputVec.y, weightVec.y);
            intersection.z = Math.min(inputVec.z, weightVec.z);
            
            double intersectionNorm = intersection.length();
            double inputNorm = inputVec.length();
            
            return intersectionNorm / (params.alpha() + inputNorm);
        } else if (input.dimension() == 4) {
            // Convert input and weight to Vector4f
            var inputVec = new Vector4f((float) input.get(0), (float) input.get(1), (float) input.get(2), (float) input.get(3));
            var weightVec = new Vector4f((float) weight.get(0), (float) weight.get(1), (float) weight.get(2), (float) weight.get(3));
            
            var intersection = new Vector4f();
            intersection.x = Math.min(inputVec.x, weightVec.x);
            intersection.y = Math.min(inputVec.y, weightVec.y);
            intersection.z = Math.min(inputVec.z, weightVec.z);
            intersection.w = Math.min(inputVec.w, weightVec.w);
            
            double intersectionNorm = intersection.length();
            double inputNorm = inputVec.length();
            
            return intersectionNorm / (params.alpha() + inputNorm);
        }
        
        // Fallback to standard computation
        return computeStandardActivation(input, weight, params);
    }
    
    /**
     * SIMD-optimized activation using Pattern API for larger dimensions.
     */
    private double computeSIMDActivation(com.hellblazer.art.core.Pattern input, VectorizedFuzzyWeight weight, VectorizedParameters params) {
        int dimension = input.dimension();
        var inputArray = convertToFloatArray(input);
        var weightArray = weight.getCategoryWeights();
        
        double intersectionSum = 0.0;
        double inputSum = 0.0;
        
        int vectorLength = SPECIES.length();
        int upperBound = SPECIES.loopBound(dimension);
        
        // Vectorized loop
        for (int i = 0; i < upperBound; i += vectorLength) {
            var inputVec = FloatVector.fromArray(SPECIES, inputArray, i);
            var weightVec = FloatVector.fromArray(SPECIES, weightArray, i);
            
            // Compute fuzzy minimum (intersection)
            var intersection = inputVec.min(weightVec);
            intersectionSum += intersection.reduceLanes(jdk.incubator.vector.VectorOperators.ADD);
            
            // Sum input values
            inputSum += inputVec.reduceLanes(jdk.incubator.vector.VectorOperators.ADD);
            
            totalVectorOperations++; // Track SIMD vector operations
        }
        
        // Handle remaining elements
        for (int i = upperBound; i < dimension; i++) {
            double inputVal = inputArray[i];
            double weightVal = weightArray[i];
            intersectionSum += Math.min(inputVal, weightVal);
            inputSum += inputVal;
        }
        
        return intersectionSum / (params.alpha() + inputSum);
    }
    
    /**
     * Standard activation computation fallback.
     */
    private double computeStandardActivation(com.hellblazer.art.core.Pattern input, VectorizedFuzzyWeight weight, VectorizedParameters params) {
        double intersection = 0.0;
        double inputNorm = 0.0;
        
        for (int i = 0; i < input.dimension(); i++) {
            double inputVal = input.get(i);
            intersection += Math.min(inputVal, weight.get(i));
            inputNorm += inputVal;
        }
        
        return intersection / (params.alpha() + inputNorm);
    }
    
    /**
     * Convert Pattern to float array for SIMD operations.
     */
    private float[] convertToFloatArray(com.hellblazer.art.core.Pattern input) {
        var array = new float[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            array[i] = (float) input.get(i);
        }
        return array;
    }
    
    /**
     * Get cached Vector3f to avoid repeated allocations.
     */
    private Vector3f getCachedVector3f(com.hellblazer.art.core.Pattern input) {
        int hash = Arrays.hashCode(new double[]{input.get(0), input.get(1), input.get(2)});
        return vectorCache.computeIfAbsent(hash, k -> 
            new Vector3f((float) input.get(0), (float) input.get(1), (float) input.get(2)));
    }
    
    /**
     * Parallel activation computation task.
     */
    private class ParallelActivationTask extends RecursiveTask<ActivationResult> {
        private final com.hellblazer.art.core.Pattern input;
        private final VectorizedParameters params;
        private final int startIndex;
        private final int endIndex;
        private static final int THRESHOLD = 100;
        
        ParallelActivationTask(com.hellblazer.art.core.Pattern input, VectorizedParameters params, int startIndex, int endIndex) {
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
                // Create new category - this should be handled by the main stepFit method
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
                // Both failed - will be handled by main stepFit
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
            vectorCache.size(),
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
        vectorCache.clear();
        totalVectorOperations = 0;
        totalParallelTasks = 0;
        avgComputeTime = 0.0;
        activationCalls = 0;
        matchCalls = 0;
        learningCalls = 0;        log.info("Performance tracking reset");
    }
    
    /**
     * Optimize memory usage by trimming caches.
     */
    public void optimizeMemory() {
        if (vectorCache.size() > defaultParams.maxCacheSize()) {
            vectorCache.clear();
            log.info("Vector cache cleared to optimize memory usage");
        }
    }
    
    /**
     * Get a VectorizedFuzzyWeight by index (type-safe accessor).
     */
    public VectorizedFuzzyWeight getVectorizedCategory(int index) {
        var category = getCategory(index);
        if (!(category instanceof VectorizedFuzzyWeight vWeight)) {
            throw new IllegalStateException("Category at index " + index + " is not a VectorizedFuzzyWeight");
        }
        return vWeight;
    }
    
    /**
     * Calculate activations for all categories against the input pattern.
     * This is a public interface for prediction without modifying categories.
     * 
     * @param input the input pattern
     * @param parameters the vectorized parameters
     * @return array of activation values for all categories
     */
    public double[] calculateAllActivations(Pattern input, VectorizedParameters parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (getCategoryCount() == 0) {
            return new double[0];
        }
        
        var activations = new double[getCategoryCount()];
        for (int i = 0; i < getCategoryCount(); i++) {
            var weight = getCategory(i);
            activations[i] = calculateActivation(input, weight, parameters);
        }
        
        return activations;
    }
    
    /**
     * Find the category with highest activation that passes vigilance.
     * This is designed for prediction without modifying categories.
     * 
     * @param input the input pattern
     * @param parameters the vectorized parameters
     * @return optional category result with index and activation, empty if no match
     */
    public Optional<CategoryActivation> findBestMatch(Pattern input, VectorizedParameters parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (getCategoryCount() == 0) {
            return Optional.empty();
        }
        
        double bestActivation = -1.0;
        int bestIndex = -1;
        
        for (int i = 0; i < getCategoryCount(); i++) {
            var weight = getCategory(i);
            var activation = calculateActivation(input, weight, parameters);
            
            // Check if this category passes vigilance
            var matchResult = checkVigilance(input, weight, parameters);
            if (matchResult.isAccepted() && activation > bestActivation) {
                bestActivation = activation;
                bestIndex = i;
            }
        }
        
        if (bestIndex >= 0) {
            return Optional.of(new CategoryActivation(bestIndex, bestActivation));
        }
        
        return Optional.empty();
    }
    
    /**
     * Record for category activation result.
     */
    public record CategoryActivation(int categoryIndex, double activation) {}
    
    // VectorizedARTAlgorithm interface implementation
    
    @Override
    public Object learn(Pattern input, VectorizedParameters parameters) {
        totalVectorOperations++; // Track each learning operation as a vector operation
        return stepFitEnhanced(input, parameters);
    }
    
    @Override
    public Object predict(Pattern input, VectorizedParameters parameters) {
        return findBestMatch(input, parameters);
    }
    
    @Override
    public VectorizedParameters getParameters() {
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
        vectorCache.clear();
        log.info("VectorizedART closed and resources cleaned up");
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedART{categories=%d, vectorOps=%d, parallelTasks=%d, avgComputeMs=%.3f}",
                           getCategoryCount(), totalVectorOperations, totalParallelTasks, avgComputeTime);
    }
}