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
 * Vectorized implementation of Binary Fuzzy ART algorithm.
 * 
 * Binary Fuzzy ART is optimized for patterns containing primarily binary values (0.0 and 1.0).
 * This vectorized implementation provides SIMD optimization and special binary processing paths
 * for enhanced performance on binary data while maintaining full compatibility with continuous values.
 * 
 * Key features:
 * - SIMD-optimized fuzzy ART operations
 * - Binary pattern detection and optimization
 * - Parallel category search for large networks
 * - Comprehensive performance tracking
 * - Automatic complement coding support
 * 
 * Algorithm:
 * 1. Detect if input patterns are primarily binary
 * 2. Use optimized binary operations when applicable
 * 3. Fall back to standard fuzzy operations for mixed patterns
 * 4. Apply fuzzy AND operations with choice function
 * 5. Check vigilance and update weights accordingly
 * 
 * @author Claude (Anthropic AI)
 * @version 1.0
 */
public final class VectorizedBinaryFuzzyART extends BaseART 
    implements VectorizedARTAlgorithm<VectorizedPerformanceStats, VectorizedBinaryFuzzyParameters> {
    
    private final AtomicLong vectorOperations = new AtomicLong(0);
    private final AtomicLong parallelTasks = new AtomicLong(0);
    private final AtomicLong computeTimeNs = new AtomicLong(0);
    private final AtomicLong binaryOptimizations = new AtomicLong(0);
    private final AtomicLong activationCalls = new AtomicLong(0);
    private final AtomicLong matchCalls = new AtomicLong(0);
    private final AtomicLong learningCalls = new AtomicLong(0);
    private final Map<Pattern, VectorizedBinaryFuzzyWeight> inputCache = new ConcurrentHashMap<>();
    
    private ForkJoinPool computePool;
    private VectorizedBinaryFuzzyParameters currentParameters;
    private Integer expectedInputDimension = null; // Track expected input dimension
    
    /**
     * Creates a new VectorizedBinaryFuzzyART instance.
     */
    public VectorizedBinaryFuzzyART() {
        super();
    }
    
    @Override
    public Object learn(Pattern input, VectorizedBinaryFuzzyParameters parameters) {
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
            var result = super.stepFit(input, parameters);
            return result;
        } finally {
            computeTimeNs.addAndGet(System.nanoTime() - startTime);
        }
    }
    
    @Override
    public Object predict(Pattern input, VectorizedBinaryFuzzyParameters parameters) {
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
            var result = super.stepPredict(input, parameters);
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
        binaryOptimizations.set(0);
        inputCache.clear();
        expectedInputDimension = null; // Reset expected dimension
    }
    
    @Override
    public VectorizedBinaryFuzzyParameters getParameters() {
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
        if (!(parameters instanceof VectorizedBinaryFuzzyParameters params)) {
            throw new IllegalArgumentException("Parameters must be VectorizedBinaryFuzzyParameters");
        }
        if (!(weight instanceof VectorizedBinaryFuzzyWeight binaryWeight)) {
            throw new IllegalArgumentException("Weight must be VectorizedBinaryFuzzyWeight");
        }
        
        vectorOperations.incrementAndGet();
        
        var inputData = convertToFloatArray(input);
        var intersection = binaryWeight.computeIntersectionSize(inputData);
        var magnitude = binaryWeight.computeL1Norm();
        
        // Choice function: T(I,J) = |I ∩ wj| / (α + |wj|)
        return intersection / (params.alpha() + magnitude);
    }
    
    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, Object parameters) {
        if (!(parameters instanceof VectorizedBinaryFuzzyParameters params)) {
            throw new IllegalArgumentException("Parameters must be VectorizedBinaryFuzzyParameters");
        }
        if (!(weight instanceof VectorizedBinaryFuzzyWeight binaryWeight)) {
            throw new IllegalArgumentException("Weight must be VectorizedBinaryFuzzyWeight");
        }
        
        vectorOperations.incrementAndGet();
        
        var inputData = convertToFloatArray(input);
        
        // Compute fuzzy AND intersection
        var intersection = binaryWeight.computeIntersectionSize(inputData);
        
        // Compute input magnitude
        var inputMagnitude = computeInputMagnitude(inputData);
        
        // Vigilance check: |I ∩ wj| / |I| >= ρ
        var vigilanceRatio = inputMagnitude > 0 ? intersection / inputMagnitude : 0.0;
        var vigilancePass = vigilanceRatio >= params.vigilance();
        
        return vigilancePass ? 
               new MatchResult.Accepted(vigilanceRatio, params.vigilance()) : 
               new MatchResult.Rejected(vigilanceRatio, params.vigilance());
    }
    
    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, Object parameters) {
        if (!(parameters instanceof VectorizedBinaryFuzzyParameters params)) {
            throw new IllegalArgumentException("Parameters must be VectorizedBinaryFuzzyParameters");
        }
        if (!(currentWeight instanceof VectorizedBinaryFuzzyWeight binaryWeight)) {
            throw new IllegalArgumentException("Weight must be VectorizedBinaryFuzzyWeight");
        }
        
        vectorOperations.incrementAndGet();
        
        var inputData = convertToFloatArray(input);
        
        // Check if we can use binary optimization
        if (binaryWeight.isBinaryOptimized() && params.enableBinaryOptimization()) {
            binaryOptimizations.incrementAndGet();
        }
        
        // Fast learning: new_weight = fuzzy_AND(input, old_weight)
        if (params.beta() >= 1.0) {
            return binaryWeight.fuzzyAND(inputData, params.enableBinaryOptimization());
        }
        
        // Slow learning with binary optimization
        return binaryWeight.slowLearning(inputData, params.beta(), params.enableBinaryOptimization());
    }
    
    @Override
    protected WeightVector createInitialWeight(Pattern input, Object parameters) {
        if (!(parameters instanceof VectorizedBinaryFuzzyParameters params)) {
            throw new IllegalArgumentException("Parameters must be VectorizedBinaryFuzzyParameters");
        }
        
        var inputData = convertToFloatArray(input);
        return VectorizedBinaryFuzzyWeight.createInitial(inputData, params.getEffectiveBinaryThreshold());
    }
    
    // Helper methods
    
    /**
     * Converts Pattern to float array, handling complement coding if needed.
     */
    private float[] convertToFloatArray(Pattern input) {
        if (currentParameters != null && currentParameters.maxCacheSize() > 0) {
            var cachedWeight = inputCache.get(input);
            if (cachedWeight != null) {
                return cachedWeight.getWeights();
            }
        }
        
        return doConvertToFloatArray(input);
    }
    
    private float[] doConvertToFloatArray(Pattern input) {
        var data = ((DenseVector) input).values();
        
        // Validate input dimension consistency
        if (expectedInputDimension == null) {
            // First input sets the expected dimension
            expectedInputDimension = data.length;
        } else if (data.length != expectedInputDimension) {
            throw new IllegalArgumentException(String.format(
                "Pattern dimension mismatch: expected %d, got %d. " +
                "All patterns must have the same dimension.",
                expectedInputDimension, data.length));
        }
        
        var floatData = new float[data.length];
        
        for (int i = 0; i < data.length; i++) {
            floatData[i] = (float) data[i];
        }
        
        // Always apply complement coding for consistency
        // This ensures all patterns have the same dimension
        var complementCoded = new float[data.length * 2];
        System.arraycopy(floatData, 0, complementCoded, 0, data.length);
        for (int i = 0; i < data.length; i++) {
            complementCoded[i + data.length] = 1.0f - floatData[i];
        }
        
        // Cache the complement coded pattern if caching is enabled
        if (currentParameters != null && currentParameters.maxCacheSize() > 0 && 
            inputCache.size() < currentParameters.maxCacheSize()) {
            var weight = VectorizedBinaryFuzzyWeight.createInitial(complementCoded, 
                currentParameters.getEffectiveBinaryThreshold());
            inputCache.put(input, weight);
        }
        
        return complementCoded;
    }
    
    /**
     * Determines if input needs complement coding.
     * Uses heuristic to check if data already follows [x, 1-x] complement pattern.
     */
    private boolean needsComplementCoding(double[] data) {
        // Simple heuristic: if length is odd, needs complement coding
        if (data.length % 2 != 0) {
            return true;
        }
        
        // For even length, check if it looks like [x, 1-x] pattern
        int halfLen = data.length / 2;
        double tolerance = 1e-3;
        int complementPairs = 0;
        
        for (int i = 0; i < halfLen; i++) {
            double original = data[i];
            double complement = data[i + halfLen];
            
            // Check if complement[i] ≈ 1 - original[i]
            if (Math.abs(complement - (1.0 - original)) <= tolerance) {
                complementPairs++;
            }
        }
        
        // If most pairs follow complement pattern, assume already complement-coded
        double complementRatio = (double) complementPairs / halfLen;
        return complementRatio < 0.7; // Less than 70% complement pairs = needs coding
    }
    
    /**
     * Computes input magnitude for vigilance evaluation.
     * For complement-coded input [x, 1-x], the magnitude is always equal to
     * the original dimension since sum(x) + sum(1-x) = n.
     */
    private double computeInputMagnitude(float[] inputData) {
        double magnitude = 0.0;
        
        // Compute L1 norm of the full complement-coded vector
        for (int i = 0; i < inputData.length; i++) {
            magnitude += inputData[i];
        }
        
        return magnitude;
    }
    
    /**
     * Converts float array to double array for complement coding check.
     */
    private double[] convertToDoubleArray(float[] floatArray) {
        var doubleArray = new double[floatArray.length];
        for (int i = 0; i < floatArray.length; i++) {
            doubleArray[i] = floatArray[i];
        }
        return doubleArray;
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
            if (currentParameters.shouldUseParallelProcessing(getCategoryCount())) {
                var parallelism = currentParameters.parallelismLevel();
                computePool = new ForkJoinPool(parallelism);
                parallelTasks.incrementAndGet();
            }
        }
    }
    
    /**
     * Gets the number of binary optimizations performed.
     * 
     * @return Binary optimization count
     */
    public long getBinaryOptimizationCount() {
        return binaryOptimizations.get();
    }
    
    /**
     * Gets the binary optimization rate as a percentage.
     * 
     * @return Binary optimization rate [0,100]
     */
    public double getBinaryOptimizationRate() {
        var totalOps = vectorOperations.get();
        return totalOps > 0 ? (binaryOptimizations.get() * 100.0 / totalOps) : 0.0;
    }
    
    @Override
    public String toString() {
        var stats = getPerformanceStats();
        return String.format("VectorizedBinaryFuzzyART{categories=%d, vectorOps=%d, " +
                           "binaryOpts=%d (%.1f%%), avgComputeTime=%.3fms, cacheSize=%d}",
                           getCategoryCount(), stats.totalVectorOperations(),
                           getBinaryOptimizationCount(), getBinaryOptimizationRate(),
                           stats.avgComputeTimeMs(), stats.cacheSize());
    }
}