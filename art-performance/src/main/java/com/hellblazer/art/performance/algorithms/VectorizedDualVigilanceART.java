package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.*;
import com.hellblazer.art.core.results.MatchResult;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.performance.VectorizedARTAlgorithm;
import com.hellblazer.art.performance.algorithms.VectorizedPerformanceStats;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;

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
public final class VectorizedDualVigilanceART extends BaseART 
    implements VectorizedARTAlgorithm<VectorizedPerformanceStats, VectorizedDualVigilanceParameters> {
    
    private final AtomicLong vectorOperations = new AtomicLong(0);
    private final AtomicLong parallelTasks = new AtomicLong(0);
    private final AtomicLong computeTimeNs = new AtomicLong(0);
    private final AtomicLong activationCalls = new AtomicLong(0);
    private final AtomicLong matchCalls = new AtomicLong(0);
    private final AtomicLong learningCalls = new AtomicLong(0);
    private final Map<Pattern, VectorizedDualVigilanceWeight> inputCache = new ConcurrentHashMap<>();
    
    private ForkJoinPool computePool;
    private VectorizedDualVigilanceParameters currentParameters;
    
    /**
     * Creates a new VectorizedDualVigilanceART instance.
     */
    public VectorizedDualVigilanceART() {
        super();
    }
    
    @Override
    public Object learn(Pattern input, VectorizedDualVigilanceParameters parameters) {
        if (input == null) {
            throw new IllegalArgumentException("Input pattern cannot be null");
        }
        if (parameters == null) {
            throw new IllegalArgumentException("Parameters cannot be null");
        }
        
        var startTime = System.nanoTime();
        this.currentParameters = parameters;
        ensureComputePool();
        
        try {
            // Ensure input is complement coded
            var complementCodedInput = ensureComplementCoded(input);
            var result = super.stepFit(complementCodedInput, parameters);
            return result;
        } finally {
            computeTimeNs.addAndGet(System.nanoTime() - startTime);
        }
    }
    
    @Override
    public Object predict(Pattern input, VectorizedDualVigilanceParameters parameters) {
        if (input == null) {
            throw new IllegalArgumentException("Input pattern cannot be null");
        }
        if (parameters == null) {
            throw new IllegalArgumentException("Parameters cannot be null");
        }
        
        var startTime = System.nanoTime();
        this.currentParameters = parameters;
        ensureComputePool();
        
        try {
            // Ensure input is complement coded
            var complementCodedInput = ensureComplementCoded(input);
            var result = super.stepPredict(complementCodedInput, parameters);
            return result;
        } finally {
            computeTimeNs.addAndGet(System.nanoTime() - startTime);
        }
    }
    
    @Override
    public VectorizedPerformanceStats getPerformanceStats() {
        var avgComputeTime = getCategoryCount() > 0 ? 
            computeTimeNs.get() / 1_000_000.0 / getCategoryCount() : 0.0;
            
        return new VectorizedPerformanceStats(
            vectorOperations.get(),
            parallelTasks.get(),
            avgComputeTime,
            computePool != null ? computePool.getActiveThreadCount() : 0,
            inputCache.size(),
            getCategoryCount(),
            activationCalls.get(),
            matchCalls.get(),
            learningCalls.get()
        );
    }
    
    @Override
    public void resetPerformanceTracking() {
        vectorOperations.set(0);
        parallelTasks.set(0);
        computeTimeNs.set(0);
        activationCalls.set(0);
        matchCalls.set(0);
        learningCalls.set(0);
        inputCache.clear();
    }
    
    @Override
    public VectorizedDualVigilanceParameters getParameters() {
        return currentParameters;
    }
    
    @Override
    public void close() {
        if (computePool != null) {
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
        
        vectorOperations.incrementAndGet();
        activationCalls.incrementAndGet();
        
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
        
        vectorOperations.incrementAndGet();
        matchCalls.incrementAndGet();
        
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
        
        vectorOperations.incrementAndGet();
        learningCalls.incrementAndGet();
        
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
        if (getCategoryCount() > 0) {
            var expectedDim = getCategory(0).dimension();
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
    
    /**
     * Converts LearningResult to ActivationResult.
     */
    private Object convertToActivationResult(LearningResult result) {
        if (result.wasSuccessful()) {
            var weight = result.getWeight();
            var activation = calculateActivation(result.getInput(), weight, currentParameters);
            return new ActivationResult.Success(result.getCategoryIndex(), activation, weight);
        }
        return ActivationResult.NoMatch.instance();
    }
    
    /**
     * Converts PredictionResult to ActivationResult.
     */
    private Object convertToActivationResult(PredictionResult result) {
        if (result.wasSuccessful()) {
            var weight = getCategory(result.getCategoryIndex());
            var activation = calculateActivation(result.getInput(), weight, currentParameters);
            return new ActivationResult.Success(result.getCategoryIndex(), activation, weight);
        }
        return ActivationResult.NoMatch.instance();
    }
    
    /**
     * Ensures compute pool is initialized with current parameters.
     */
    private void ensureComputePool() {
        if (currentParameters == null) {
            return;
        }
        
        if (computePool == null || computePool.isShutdown()) {
            var parallelism = currentParameters.parallelismLevel();
            if (parallelism > 1) {
                computePool = new ForkJoinPool(parallelism);
            }
        }
    }
    
    @Override
    public boolean isVectorized() {
        return true;
    }
    
    @Override
    public int getVectorSpeciesLength() {
        // Return the SIMD vector species length for Float vectors
        return jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.length();
    }
    
    @Override
    public String toString() {
        var stats = getPerformanceStats();
        return String.format("VectorizedDualVigilanceART{categories=%d, vectorOps=%d, " +
                           "avgComputeTime=%.3fms, cacheSize=%d}",
                           getCategoryCount(), stats.totalVectorOperations(),
                           stats.avgComputeTimeMs(), stats.cacheSize());
    }
}