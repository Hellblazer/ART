package com.hellblazer.art.performance.algorithms;

/**
 * Parameters for VectorizedDualVigilanceART with dual vigilance threshold control.
 * 
 * DualVigilanceART extends FuzzyART by using two vigilance parameters:
 * - rhoLower: Lower vigilance threshold for initial matching
 * - rhoUpper: Upper vigilance threshold for final acceptance
 * 
 * This dual-threshold approach provides more flexible category creation
 * and matching behavior, allowing for graduated acceptance criteria.
 * 
 * Key Parameters:
 * - rhoLower: Lower vigilance threshold (must be < rhoUpper)
 * - rhoUpper: Upper vigilance threshold (must be > rhoLower)
 * - alpha: Choice parameter for activation function
 * - beta: Learning rate parameter for weight updates
 * 
 * Performance Parameters:
 * - parallelismLevel: Number of parallel threads for large category sets
 * - parallelThreshold: Minimum categories before enabling parallelism
 * - maxCacheSize: Maximum cached input patterns for SIMD optimization
 * - enableSIMD: Enable SIMD vectorization for dual vigilance calculations
 */
public record VectorizedDualVigilanceParameters(
    double rhoLower,
    double rhoUpper,
    double alpha,
    double beta,
    int parallelismLevel,
    int parallelThreshold,
    int maxCacheSize,
    boolean enableSIMD
) {
    
    /**
     * Create VectorizedDualVigilanceParameters with validation.
     */
    public VectorizedDualVigilanceParameters {
        if (rhoLower < 0.0 || rhoLower > 1.0) {
            throw new IllegalArgumentException("Lower vigilance must be in [0,1], got: " + rhoLower);
        }
        if (rhoUpper < 0.0 || rhoUpper > 1.0) {
            throw new IllegalArgumentException("Upper vigilance must be in [0,1], got: " + rhoUpper);
        }
        if (rhoLower >= rhoUpper) {
            throw new IllegalArgumentException("Lower vigilance must be < upper vigilance, got: " + 
                                             rhoLower + " >= " + rhoUpper);
        }
        if (alpha <= 0.0) {
            throw new IllegalArgumentException("Alpha must be > 0, got: " + alpha);
        }
        if (beta < 0.0 || beta > 1.0) {
            throw new IllegalArgumentException("Beta must be in [0,1], got: " + beta);
        }
        if (parallelismLevel < 1) {
            throw new IllegalArgumentException("Parallelism level must be >= 1, got: " + parallelismLevel);
        }
        if (parallelThreshold < 1) {
            throw new IllegalArgumentException("Parallel threshold must be >= 1, got: " + parallelThreshold);
        }
        if (maxCacheSize < 0) {
            throw new IllegalArgumentException("Max cache size must be >= 0, got: " + maxCacheSize);
        }
    }
    
    /**
     * Create default parameters optimized for dual vigilance clustering.
     */
    public static VectorizedDualVigilanceParameters createDefault() {
        int processors = Runtime.getRuntime().availableProcessors();
        return new VectorizedDualVigilanceParameters(
            0.6,                       // rhoLower - moderate lower threshold
            0.85,                      // rhoUpper - high upper threshold
            0.1,                       // alpha - standard choice parameter
            0.9,                       // beta - fast learning
            Math.max(2, processors/2), // parallelismLevel
            50,                        // parallelThreshold
            1000,                      // maxCacheSize
            true                       // enableSIMD
        );
    }
    
    /**
     * Create parameters for fine-grained clustering (tight thresholds).
     */
    public static VectorizedDualVigilanceParameters createFineGrained() {
        int processors = Runtime.getRuntime().availableProcessors();
        return new VectorizedDualVigilanceParameters(
            0.8,                       // High lower threshold
            0.95,                      // Very high upper threshold
            0.05,                      // Low alpha for precise matching
            0.95,                      // Fast learning
            processors,                // Use all cores
            20,                        // Lower threshold for parallelism
            2000,                      // Larger cache
            true                       // enableSIMD
        );
    }
    
    /**
     * Create parameters for coarse clustering (loose thresholds).
     */
    public static VectorizedDualVigilanceParameters createCoarseClustering() {
        return new VectorizedDualVigilanceParameters(
            0.3,                       // Low lower threshold
            0.6,                       // Moderate upper threshold
            0.2,                       // Higher alpha for broader matching
            0.8,                       // Moderate learning rate
            4,                         // Moderate parallelism
            100,                       // Higher threshold
            500,                       // Moderate cache
            true                       // enableSIMD
        );
    }
    
    /**
     * Create parameters for adaptive clustering (dynamic thresholds).
     */
    public static VectorizedDualVigilanceParameters createAdaptive() {
        return new VectorizedDualVigilanceParameters(
            0.4,                       // Low-moderate lower threshold
            0.75,                      // Moderate-high upper threshold
            0.1,                       // Standard alpha
            0.7,                       // Adaptive learning rate
            6,                         // Higher parallelism for adaptation
            30,                        // Lower threshold for parallel processing
            1500,                      // Larger cache for adaptation
            true                       // enableSIMD
        );
    }
    
    /**
     * Create memory-optimized parameters for resource-constrained environments.
     */
    public static VectorizedDualVigilanceParameters createMemoryOptimized() {
        return new VectorizedDualVigilanceParameters(
            0.5,                       // Standard lower threshold
            0.8,                       // Standard upper threshold
            0.1,                       // Standard alpha
            0.9,                       // Fast learning to converge quickly
            2,                         // Minimal parallelism
            200,                       // High threshold for parallelism
            100,                       // Small cache
            false                      // Disable SIMD to save memory
        );
    }
    
    /**
     * Create real-time parameters optimized for low-latency processing.
     */
    public static VectorizedDualVigilanceParameters createRealTime() {
        return new VectorizedDualVigilanceParameters(
            0.6,                       // Moderate lower threshold
            0.85,                      // High upper threshold
            0.05,                      // Low alpha for quick decisions
            0.95,                      // Very fast learning
            4,                         // Moderate parallelism
            10,                        // Very low threshold
            500,                       // Moderate cache
            true                       // enableSIMD for speed
        );
    }
    
    /**
     * Create a new VectorizedDualVigilanceParameters with modified lower vigilance.
     */
    public VectorizedDualVigilanceParameters withRhoLower(double newRhoLower) {
        return new VectorizedDualVigilanceParameters(
            newRhoLower, rhoUpper, alpha, beta, parallelismLevel,
            parallelThreshold, maxCacheSize, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedDualVigilanceParameters with modified upper vigilance.
     */
    public VectorizedDualVigilanceParameters withRhoUpper(double newRhoUpper) {
        return new VectorizedDualVigilanceParameters(
            rhoLower, newRhoUpper, alpha, beta, parallelismLevel,
            parallelThreshold, maxCacheSize, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedDualVigilanceParameters with modified vigilance pair.
     */
    public VectorizedDualVigilanceParameters withVigilancePair(double newRhoLower, double newRhoUpper) {
        return new VectorizedDualVigilanceParameters(
            newRhoLower, newRhoUpper, alpha, beta, parallelismLevel,
            parallelThreshold, maxCacheSize, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedDualVigilanceParameters with modified alpha.
     */
    public VectorizedDualVigilanceParameters withAlpha(double newAlpha) {
        return new VectorizedDualVigilanceParameters(
            rhoLower, rhoUpper, newAlpha, beta, parallelismLevel,
            parallelThreshold, maxCacheSize, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedDualVigilanceParameters with modified beta.
     */
    public VectorizedDualVigilanceParameters withBeta(double newBeta) {
        return new VectorizedDualVigilanceParameters(
            rhoLower, rhoUpper, alpha, newBeta, parallelismLevel,
            parallelThreshold, maxCacheSize, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedDualVigilanceParameters with modified parallelism settings.
     */
    public VectorizedDualVigilanceParameters withParallelism(int newParallelismLevel, int newParallelThreshold) {
        return new VectorizedDualVigilanceParameters(
            rhoLower, rhoUpper, alpha, beta, newParallelismLevel,
            newParallelThreshold, maxCacheSize, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedDualVigilanceParameters with modified cache settings.
     */
    public VectorizedDualVigilanceParameters withCacheSettings(int newMaxCacheSize, boolean newEnableSIMD) {
        return new VectorizedDualVigilanceParameters(
            rhoLower, rhoUpper, alpha, beta, parallelismLevel,
            parallelThreshold, newMaxCacheSize, newEnableSIMD
        );
    }
    
    /**
     * Get the vigilance range (rhoUpper - rhoLower).
     */
    public double getVigilanceRange() {
        return rhoUpper - rhoLower;
    }
    
    /**
     * Get the middle vigilance value.
     */
    public double getMiddleVigilance() {
        return (rhoLower + rhoUpper) / 2.0;
    }
    
    /**
     * Check if vigilance range is narrow (< 0.2).
     */
    public boolean isNarrowRange() {
        return getVigilanceRange() < 0.2;
    }
    
    /**
     * Check if vigilance range is wide (> 0.5).
     */
    public boolean isWideRange() {
        return getVigilanceRange() > 0.5;
    }
    
    /**
     * Get estimated memory usage for given problem size.
     */
    public long getEstimatedMemoryUsage(int expectedCategories, int inputDimension) {
        long baseUsage = 1000L;
        // Each category stores fuzzy weights (complement coded)
        long categoryUsage = expectedCategories * inputDimension * 2 * 8L; // double arrays
        long cacheUsage = maxCacheSize * inputDimension * 4L; // float arrays for SIMD
        return baseUsage + categoryUsage + cacheUsage;
    }
    
    /**
     * Check if configuration is suitable for given problem size.
     */
    public boolean isSuitableFor(int expectedCategories, int inputDimension) {
        long memoryUsage = getEstimatedMemoryUsage(expectedCategories, inputDimension);
        long availableMemory = Runtime.getRuntime().maxMemory();
        return memoryUsage < (availableMemory * 0.8); // Use 80% threshold
    }
    
    /**
     * Get recommended configuration for problem characteristics.
     */
    public static VectorizedDualVigilanceParameters recommend(int expectedCategories, int inputDimension, String priority) {
        return switch (priority.toLowerCase()) {
            case "precision", "fine" -> {
                var params = createFineGrained();
                if (!params.isSuitableFor(expectedCategories, inputDimension)) {
                    yield params.withCacheSettings(params.maxCacheSize / 2, true);
                }
                yield params;
            }
            case "coarse", "broad" -> createCoarseClustering();
            case "adaptive", "dynamic" -> createAdaptive();
            case "memory" -> createMemoryOptimized();
            case "speed", "realtime" -> {
                var params = createRealTime();
                if (!params.isSuitableFor(expectedCategories, inputDimension)) {
                    yield params.withCacheSettings(100, false);
                }
                yield params;
            }
            default -> createDefault();
        };
    }
    
    /**
     * Validate that a similarity value falls within the dual vigilance range.
     */
    public VigilanceResult evaluateVigilance(double similarity) {
        if (similarity >= rhoUpper) {
            return VigilanceResult.ACCEPTED_UPPER;
        } else if (similarity >= rhoLower) {
            return VigilanceResult.ACCEPTED_LOWER;
        } else {
            return VigilanceResult.REJECTED;
        }
    }
    
    /**
     * Enum representing the result of dual vigilance evaluation.
     */
    public enum VigilanceResult {
        ACCEPTED_UPPER,  // Passes upper threshold - strong match
        ACCEPTED_LOWER,  // Passes lower threshold only - weak match
        REJECTED         // Fails both thresholds
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedDualVigilanceParameters{rhoLower=%.3f, rhoUpper=%.3f, " +
                           "alpha=%.3f, beta=%.3f, parallel=%d/%d, cache=%d, simd=%s, range=%.3f}",
                           rhoLower, rhoUpper, alpha, beta, parallelismLevel, parallelThreshold,
                           maxCacheSize, enableSIMD, getVigilanceRange());
    }
}