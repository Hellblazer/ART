package com.hellblazer.art.algorithms;

import java.util.Objects;

/**
 * Parameters for VectorizedART with performance optimization settings.
 * 
 * This record encapsulates configuration for:
 * - Core ART parameters (vigilance, learning rate, alpha)
 * - Parallel processing settings
 * - Memory management options  
 * - Performance tuning parameters
 * 
 * All parameters are immutable and validated at construction.
 */
public record VectorizedParameters(
    double vigilanceThreshold,
    double learningRate,
    double alpha,
    int parallelismLevel,
    int parallelThreshold,
    int maxCacheSize,
    boolean enableSIMD,
    boolean enableJOML,
    double memoryOptimizationThreshold
) {
    
    /**
     * Create VectorizedParameters with validation.
     */
    public VectorizedParameters {
        if (vigilanceThreshold < 0.0 || vigilanceThreshold > 1.0) {
            throw new IllegalArgumentException("Vigilance threshold must be in [0,1], got: " + vigilanceThreshold);
        }
        if (learningRate < 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException("Learning rate must be in [0,1], got: " + learningRate);
        }
        if (alpha <= 0.0) {
            throw new IllegalArgumentException("Alpha must be > 0, got: " + alpha);
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
        if (memoryOptimizationThreshold < 0.0 || memoryOptimizationThreshold > 1.0) {
            throw new IllegalArgumentException("Memory optimization threshold must be in [0,1], got: " + memoryOptimizationThreshold);
        }
    }
    
    /**
     * Create default parameters optimized for the current system.
     */
    public static VectorizedParameters createDefault() {
        int processors = Runtime.getRuntime().availableProcessors();
        return new VectorizedParameters(
            0.75,                      // vigilanceThreshold
            0.1,                       // learningRate
            0.001,                     // alpha
            Math.max(2, processors/2), // parallelismLevel - use half available cores
            50,                        // parallelThreshold - use parallel for >50 categories
            1000,                      // maxCacheSize
            true,                      // enableSIMD
            true,                      // enableJOML
            0.8                        // memoryOptimizationThreshold
        );
    }
    
    /**
     * Create high-performance parameters for large-scale processing.
     */
    public static VectorizedParameters createHighPerformance() {
        int processors = Runtime.getRuntime().availableProcessors();
        return new VectorizedParameters(
            0.8,                       // Higher vigilance for better discrimination
            0.05,                      // Lower learning rate for stability
            0.001,                     // Standard alpha
            processors,                // Use all available cores
            20,                        // Lower threshold for parallel processing
            2000,                      // Larger cache
            true,                      // enableSIMD
            true,                      // enableJOML
            0.9                        // Aggressive memory optimization
        );
    }
    
    /**
     * Create memory-optimized parameters for resource-constrained environments.
     */
    public static VectorizedParameters createMemoryOptimized() {
        return new VectorizedParameters(
            0.7,                       // Standard vigilance
            0.15,                      // Higher learning rate for faster convergence
            0.001,                     // Standard alpha
            2,                         // Minimal parallelism
            100,                       // Higher threshold for parallel processing
            100,                       // Small cache
            false,                     // Disable SIMD to save memory
            false,                     // Disable JOML caching
            0.5                        // Conservative memory optimization
        );
    }
    
    /**
     * Create real-time parameters optimized for low-latency processing.
     */
    public static VectorizedParameters createRealTime() {
        return new VectorizedParameters(
            0.75,                      // Standard vigilance
            0.2,                       // High learning rate for quick adaptation
            0.001,                     // Standard alpha
            4,                         // Moderate parallelism
            10,                        // Very low threshold for parallel processing
            500,                       // Moderate cache size
            true,                      // enableSIMD for speed
            true,                      // enableJOML for speed
            0.95                       // Aggressive memory management
        );
    }
    
    /**
     * Create a new VectorizedParameters with modified vigilance threshold.
     */
    public VectorizedParameters withVigilance(double newVigilance) {
        return new VectorizedParameters(
            newVigilance, learningRate, alpha, parallelismLevel, parallelThreshold,
            maxCacheSize, enableSIMD, enableJOML, memoryOptimizationThreshold
        );
    }
    
    /**
     * Create a new VectorizedParameters with modified learning rate.
     */
    public VectorizedParameters withLearningRate(double newLearningRate) {
        return new VectorizedParameters(
            vigilanceThreshold, newLearningRate, alpha, parallelismLevel, parallelThreshold,
            maxCacheSize, enableSIMD, enableJOML, memoryOptimizationThreshold
        );
    }
    
    /**
     * Create a new VectorizedParameters with modified parallelism level.
     */
    public VectorizedParameters withParallelismLevel(int newParallelismLevel) {
        return new VectorizedParameters(
            vigilanceThreshold, learningRate, alpha, newParallelismLevel, parallelThreshold,
            maxCacheSize, enableSIMD, enableJOML, memoryOptimizationThreshold
        );
    }
    
    /**
     * Create a new VectorizedParameters with modified parallel threshold.
     */
    public VectorizedParameters withParallelThreshold(int newParallelThreshold) {
        return new VectorizedParameters(
            vigilanceThreshold, learningRate, alpha, parallelismLevel, newParallelThreshold,
            maxCacheSize, enableSIMD, enableJOML, memoryOptimizationThreshold
        );
    }
    
    /**
     * Create a new VectorizedParameters with modified cache settings.
     */
    public VectorizedParameters withCacheSettings(int newMaxCacheSize, boolean newEnableSIMD, boolean newEnableJOML) {
        return new VectorizedParameters(
            vigilanceThreshold, learningRate, alpha, parallelismLevel, parallelThreshold,
            newMaxCacheSize, newEnableSIMD, newEnableJOML, memoryOptimizationThreshold
        );
    }
    
    /**
     * Get estimated memory usage in bytes.
     */
    public long getEstimatedMemoryUsage(int expectedCategories, int inputDimension) {
        long baseUsage = 1000L; // Base overhead
        long categoryUsage = expectedCategories * inputDimension * 8L; // double arrays
        long cacheUsage = maxCacheSize * (inputDimension <= 4 ? 16L : 0L); // JOML cache
        return baseUsage + categoryUsage + cacheUsage;
    }
    
    /**
     * Check if this configuration is suitable for the given problem size.
     */
    public boolean isSuitableFor(int expectedCategories, int inputDimension) {
        long memoryUsage = getEstimatedMemoryUsage(expectedCategories, inputDimension);
        long availableMemory = Runtime.getRuntime().maxMemory();
        
        return memoryUsage < (availableMemory * memoryOptimizationThreshold);
    }
    
    /**
     * Get recommended configuration for the given problem characteristics.
     */
    public static VectorizedParameters recommend(int expectedCategories, int inputDimension, String priority) {
        return switch (priority.toLowerCase()) {
            case "speed", "performance" -> {
                var params = createHighPerformance();
                if (!params.isSuitableFor(expectedCategories, inputDimension)) {
                    yield params.withCacheSettings(params.maxCacheSize / 2, true, inputDimension <= 4);
                }
                yield params;
            }
            case "memory" -> createMemoryOptimized();
            case "realtime", "latency" -> {
                var params = createRealTime();
                if (!params.isSuitableFor(expectedCategories, inputDimension)) {
                    yield params.withCacheSettings(100, false, false);
                }
                yield params;
            }
            default -> createDefault();
        };
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedParameters{vigilance=%.3f, lr=%.3f, alpha=%.3f, " +
                           "parallel=%d/%d, cache=%d, simd=%s, joml=%s}",
                           vigilanceThreshold, learningRate, alpha, 
                           parallelismLevel, parallelThreshold, maxCacheSize, 
                           enableSIMD, enableJOML);
    }
}