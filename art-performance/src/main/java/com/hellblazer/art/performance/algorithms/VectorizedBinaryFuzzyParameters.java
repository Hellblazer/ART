package com.hellblazer.art.performance.algorithms;

/**
 * Parameters for VectorizedBinaryFuzzyART algorithm.
 * 
 * Binary-optimized fuzzy ART parameters with SIMD and parallel processing configuration.
 * Optimized for patterns containing primarily binary values (0.0 and 1.0).
 * 
 * Key parameters:
 * - vigilance: Similarity threshold for category acceptance [0,1]
 * - alpha: Choice function parameter (small positive value)
 * - beta: Learning rate parameter [0,1]
 * - binaryThreshold: Threshold for binary optimization (default 0.1)
 * 
 * @author Claude (Anthropic AI)
 * @version 1.0
 */
public record VectorizedBinaryFuzzyParameters(
    double vigilance,           // Vigilance parameter [0,1]
    double alpha,              // Choice function parameter (> 0)
    double beta,               // Learning rate [0,1] 
    double binaryThreshold,    // Binary optimization threshold [0,1]
    int parallelismLevel,      // Parallel processing level
    int parallelThreshold,     // Minimum categories for parallel processing
    int maxCacheSize,          // Maximum input pattern cache size
    boolean enableSIMD,        // Enable SIMD vectorization
    boolean enableBinaryOptimization // Enable binary-specific optimizations
) {
    
    /**
     * Creates parameters with validation.
     */
    public VectorizedBinaryFuzzyParameters {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in [0,1], got: " + vigilance);
        }
        if (alpha <= 0.0) {
            throw new IllegalArgumentException("Alpha must be positive, got: " + alpha);
        }
        if (beta < 0.0 || beta > 1.0) {
            throw new IllegalArgumentException("Beta must be in [0,1], got: " + beta);
        }
        if (binaryThreshold < 0.0 || binaryThreshold > 1.0) {
            throw new IllegalArgumentException("Binary threshold must be in [0,1], got: " + binaryThreshold);
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
     * Creates standard binary fuzzy ART parameters.
     * Optimized for typical binary pattern recognition tasks.
     * 
     * @return Standard parameter configuration
     */
    public static VectorizedBinaryFuzzyParameters standard() {
        return new VectorizedBinaryFuzzyParameters(
            0.75,   // vigilance - moderate selectivity
            0.01,   // alpha - small choice parameter
            1.0,    // beta - fast learning
            0.1,    // binaryThreshold - 10% tolerance for binary detection
            1,      // parallelismLevel - single-threaded by default
            100,    // parallelThreshold - enable parallel for >100 categories
            1000,   // maxCacheSize - cache up to 1000 patterns
            true,   // enableSIMD - use SIMD by default
            true    // enableBinaryOptimization - use binary optimizations
        );
    }
    
    /**
     * Creates parameters optimized for strict binary patterns (exactly 0.0 or 1.0).
     * 
     * @return Strict binary parameter configuration
     */
    public static VectorizedBinaryFuzzyParameters strictBinary() {
        return new VectorizedBinaryFuzzyParameters(
            0.9,     // vigilance - high selectivity for binary patterns
            0.001,   // alpha - very small choice parameter
            1.0,     // beta - fast learning
            0.01,    // binaryThreshold - very strict binary detection
            1,       // parallelismLevel
            50,      // parallelThreshold - lower threshold for binary
            2000,    // maxCacheSize - larger cache for binary patterns
            true,    // enableSIMD
            true     // enableBinaryOptimization
        );
    }
    
    /**
     * Creates parameters for mixed binary/continuous patterns.
     * Balances binary optimization with continuous value handling.
     * 
     * @return Mixed pattern parameter configuration
     */
    public static VectorizedBinaryFuzzyParameters mixed() {
        return new VectorizedBinaryFuzzyParameters(
            0.7,    // vigilance - moderate selectivity
            0.05,   // alpha - moderate choice parameter
            0.8,    // beta - moderate learning rate
            0.2,    // binaryThreshold - relaxed binary detection
            1,      // parallelismLevel
            150,    // parallelThreshold
            500,    // maxCacheSize - smaller cache for mixed patterns
            true,   // enableSIMD
            false   // enableBinaryOptimization - disabled for mixed patterns
        );
    }
    
    /**
     * Creates parameters for high-performance binary processing.
     * Maximizes throughput for large-scale binary pattern processing.
     * 
     * @return High-performance parameter configuration
     */
    public static VectorizedBinaryFuzzyParameters highPerformance() {
        var availableProcessors = Runtime.getRuntime().availableProcessors();
        return new VectorizedBinaryFuzzyParameters(
            0.8,                      // vigilance
            0.01,                     // alpha
            1.0,                      // beta - fast learning
            0.05,                     // binaryThreshold - strict binary
            availableProcessors,      // parallelismLevel - use all cores
            20,                       // parallelThreshold - low threshold
            5000,                     // maxCacheSize - large cache
            true,                     // enableSIMD
            true                      // enableBinaryOptimization
        );
    }
    
    /**
     * Creates parameters for memory-constrained environments.
     * Minimizes memory usage while maintaining reasonable performance.
     * 
     * @return Memory-efficient parameter configuration
     */
    public static VectorizedBinaryFuzzyParameters memoryEfficient() {
        return new VectorizedBinaryFuzzyParameters(
            0.75,   // vigilance
            0.01,   // alpha
            1.0,    // beta
            0.1,    // binaryThreshold
            1,      // parallelismLevel - single-threaded
            1000,   // parallelThreshold - high threshold
            50,     // maxCacheSize - small cache
            false,  // enableSIMD - disabled to save memory
            true    // enableBinaryOptimization
        );
    }
    
    // Builder-style methods for parameter modification
    
    public VectorizedBinaryFuzzyParameters withVigilance(double vigilance) {
        return new VectorizedBinaryFuzzyParameters(vigilance, alpha, beta, binaryThreshold,
            parallelismLevel, parallelThreshold, maxCacheSize, enableSIMD, enableBinaryOptimization);
    }
    
    public VectorizedBinaryFuzzyParameters withAlpha(double alpha) {
        return new VectorizedBinaryFuzzyParameters(vigilance, alpha, beta, binaryThreshold,
            parallelismLevel, parallelThreshold, maxCacheSize, enableSIMD, enableBinaryOptimization);
    }
    
    public VectorizedBinaryFuzzyParameters withBeta(double beta) {
        return new VectorizedBinaryFuzzyParameters(vigilance, alpha, beta, binaryThreshold,
            parallelismLevel, parallelThreshold, maxCacheSize, enableSIMD, enableBinaryOptimization);
    }
    
    public VectorizedBinaryFuzzyParameters withBinaryThreshold(double binaryThreshold) {
        return new VectorizedBinaryFuzzyParameters(vigilance, alpha, beta, binaryThreshold,
            parallelismLevel, parallelThreshold, maxCacheSize, enableSIMD, enableBinaryOptimization);
    }
    
    public VectorizedBinaryFuzzyParameters withParallelismLevel(int parallelismLevel) {
        return new VectorizedBinaryFuzzyParameters(vigilance, alpha, beta, binaryThreshold,
            parallelismLevel, parallelThreshold, maxCacheSize, enableSIMD, enableBinaryOptimization);
    }
    
    public VectorizedBinaryFuzzyParameters withParallelThreshold(int parallelThreshold) {
        return new VectorizedBinaryFuzzyParameters(vigilance, alpha, beta, binaryThreshold,
            parallelismLevel, parallelThreshold, maxCacheSize, enableSIMD, enableBinaryOptimization);
    }
    
    public VectorizedBinaryFuzzyParameters withMaxCacheSize(int maxCacheSize) {
        return new VectorizedBinaryFuzzyParameters(vigilance, alpha, beta, binaryThreshold,
            parallelismLevel, parallelThreshold, maxCacheSize, enableSIMD, enableBinaryOptimization);
    }
    
    public VectorizedBinaryFuzzyParameters withEnableSIMD(boolean enableSIMD) {
        return new VectorizedBinaryFuzzyParameters(vigilance, alpha, beta, binaryThreshold,
            parallelismLevel, parallelThreshold, maxCacheSize, enableSIMD, enableBinaryOptimization);
    }
    
    public VectorizedBinaryFuzzyParameters withEnableBinaryOptimization(boolean enableBinaryOptimization) {
        return new VectorizedBinaryFuzzyParameters(vigilance, alpha, beta, binaryThreshold,
            parallelismLevel, parallelThreshold, maxCacheSize, enableSIMD, enableBinaryOptimization);
    }
    
    /**
     * Checks if a pattern should be processed with binary optimizations.
     * 
     * @param pattern Input pattern values
     * @return True if pattern is sufficiently binary
     */
    public boolean shouldUseBinaryOptimization(double[] pattern) {
        if (!enableBinaryOptimization) {
            return false;
        }
        
        int binaryCount = 0;
        for (double value : pattern) {
            if (Math.abs(value) <= binaryThreshold || 
                Math.abs(value - 1.0) <= binaryThreshold) {
                binaryCount++;
            }
        }
        
        // Use binary optimization if >= 80% of values are binary
        return (double) binaryCount / pattern.length >= 0.8;
    }
    
    /**
     * Determines if parallel processing should be used.
     * 
     * @param categoryCount Current number of categories
     * @return True if parallel processing should be enabled
     */
    public boolean shouldUseParallelProcessing(int categoryCount) {
        return parallelismLevel > 1 && categoryCount >= parallelThreshold;
    }
    
    /**
     * Gets the effective binary threshold considering SIMD alignment.
     * 
     * @return Effective binary threshold
     */
    public double getEffectiveBinaryThreshold() {
        // SIMD operations might need slightly relaxed thresholds
        return enableSIMD ? Math.max(binaryThreshold, 1e-6) : binaryThreshold;
    }
    
    @Override
    public String toString() {
        return String.format(
            "VectorizedBinaryFuzzyParameters{vigilance=%.3f, alpha=%.3f, beta=%.3f, " +
            "binaryThreshold=%.3f, parallelism=%d, SIMD=%s, binaryOpt=%s}",
            vigilance, alpha, beta, binaryThreshold, parallelismLevel, 
            enableSIMD, enableBinaryOptimization);
    }
}