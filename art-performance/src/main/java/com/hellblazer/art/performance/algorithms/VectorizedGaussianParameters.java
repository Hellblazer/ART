package com.hellblazer.art.performance.algorithms;

import java.util.Objects;

/**
 * Parameters for VectorizedGaussianART with performance optimization settings.
 * 
 * This record encapsulates configuration for:
 * - Core GaussianART parameters (vigilance, gamma, rho_a, rho_b)
 * - Parallel processing settings
 * - SIMD vectorization options
 * - Performance tuning parameters
 * 
 * All parameters are immutable and validated at construction.
 * 
 * GaussianART-specific parameters:
 * - vigilance: probability threshold for accepting patterns [0,1]
 * - gamma: learning rate controlling adaptation speed [0,1]
 * - rho_a: variance adjustment factor (typically 1.0)
 * - rho_b: minimum variance threshold to prevent collapse
 */
public record VectorizedGaussianParameters(
    double vigilance,
    double gamma,
    double rho_a,
    double rho_b,
    int parallelismLevel,
    boolean enableSIMD
) {
    
    /**
     * Create VectorizedGaussianParameters with validation.
     */
    public VectorizedGaussianParameters {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in [0,1], got: " + vigilance);
        }
        if (gamma < 0.0 || gamma > 1.0) {
            throw new IllegalArgumentException("Gamma (learning rate) must be in [0,1], got: " + gamma);
        }
        if (rho_a <= 0.0) {
            throw new IllegalArgumentException("Rho_a must be > 0, got: " + rho_a);
        }
        if (rho_b <= 0.0) {
            throw new IllegalArgumentException("Rho_b must be > 0, got: " + rho_b);
        }
        if (parallelismLevel < 1) {
            throw new IllegalArgumentException("Parallelism level must be >= 1, got: " + parallelismLevel);
        }
    }
    
    /**
     * Create default parameters optimized for the current system.
     */
    public static VectorizedGaussianParameters createDefault() {
        int processors = Runtime.getRuntime().availableProcessors();
        return new VectorizedGaussianParameters(
            0.75,                      // vigilance - moderate discrimination
            0.1,                       // gamma - conservative learning rate
            1.0,                       // rho_a - standard variance adjustment
            0.1,                       // rho_b - minimum variance to prevent collapse
            Math.max(2, processors/2), // parallelismLevel - use half available cores
            true                       // enableSIMD - use vectorization
        );
    }
    
    /**
     * Create high-performance parameters for large-scale processing.
     */
    public static VectorizedGaussianParameters createHighPerformance() {
        int processors = Runtime.getRuntime().availableProcessors();
        return new VectorizedGaussianParameters(
            0.8,                       // Higher vigilance for better discrimination
            0.05,                      // Lower learning rate for stability
            1.0,                       // Standard variance adjustment
            0.05,                      // Lower minimum variance for tighter clusters
            processors,                // Use all available cores
            true                       // enableSIMD for maximum speed
        );
    }
    
    /**
     * Create memory-optimized parameters for resource-constrained environments.
     */
    public static VectorizedGaussianParameters createMemoryOptimized() {
        return new VectorizedGaussianParameters(
            0.7,                       // Standard vigilance
            0.15,                      // Higher learning rate for faster convergence
            1.0,                       // Standard variance adjustment
            0.2,                       // Higher minimum variance to reduce categories
            2,                         // Minimal parallelism
            false                      // Disable SIMD to save memory
        );
    }
    
    /**
     * Create real-time parameters optimized for low-latency processing.
     */
    public static VectorizedGaussianParameters createRealTime() {
        return new VectorizedGaussianParameters(
            0.75,                      // Standard vigilance
            0.2,                       // High learning rate for quick adaptation
            1.2,                       // Slightly higher variance adjustment for flexibility
            0.15,                      // Moderate minimum variance
            4,                         // Moderate parallelism for low latency
            true                       // enableSIMD for speed
        );
    }
    
    /**
     * Create research parameters optimized for accuracy and reproducibility.
     */
    public static VectorizedGaussianParameters createResearch() {
        return new VectorizedGaussianParameters(
            0.9,                       // High vigilance for precise discrimination
            0.01,                      // Very low learning rate for stability
            1.0,                       // Standard variance adjustment
            0.01,                      // Very low minimum variance for tight clusters
            1,                         // Single-threaded for reproducibility
            false                      // Disable SIMD for exact reproducibility
        );
    }
    
    /**
     * Create a new VectorizedGaussianParameters with modified vigilance.
     */
    public VectorizedGaussianParameters withVigilance(double newVigilance) {
        return new VectorizedGaussianParameters(
            newVigilance, gamma, rho_a, rho_b, parallelismLevel, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedGaussianParameters with modified gamma (learning rate).
     */
    public VectorizedGaussianParameters withGamma(double newGamma) {
        return new VectorizedGaussianParameters(
            vigilance, newGamma, rho_a, rho_b, parallelismLevel, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedGaussianParameters with modified rho_a.
     */
    public VectorizedGaussianParameters withRhoA(double newRhoA) {
        return new VectorizedGaussianParameters(
            vigilance, gamma, newRhoA, rho_b, parallelismLevel, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedGaussianParameters with modified rho_b.
     */
    public VectorizedGaussianParameters withRhoB(double newRhoB) {
        return new VectorizedGaussianParameters(
            vigilance, gamma, rho_a, newRhoB, parallelismLevel, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedGaussianParameters with modified parallelism level.
     */
    public VectorizedGaussianParameters withParallelismLevel(int newParallelismLevel) {
        return new VectorizedGaussianParameters(
            vigilance, gamma, rho_a, rho_b, newParallelismLevel, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedGaussianParameters with modified SIMD setting.
     */
    public VectorizedGaussianParameters withSIMD(boolean newEnableSIMD) {
        return new VectorizedGaussianParameters(
            vigilance, gamma, rho_a, rho_b, parallelismLevel, newEnableSIMD
        );
    }
    
    /**
     * Get estimated memory usage in bytes for given problem size.
     */
    public long getEstimatedMemoryUsage(int expectedCategories, int inputDimension) {
        long baseUsage = 1000L; // Base overhead
        // Each Gaussian category stores: mean vector + covariance matrix + metadata
        long categoryUsage = expectedCategories * (inputDimension * 8L + inputDimension * inputDimension * 8L + 100L);
        return baseUsage + categoryUsage;
    }
    
    /**
     * Check if this configuration is suitable for the given problem size.
     */
    public boolean isSuitableFor(int expectedCategories, int inputDimension) {
        long memoryUsage = getEstimatedMemoryUsage(expectedCategories, inputDimension);
        long availableMemory = Runtime.getRuntime().maxMemory();
        
        // Use at most 50% of available memory
        return memoryUsage < (availableMemory * 0.5);
    }
    
    /**
     * Get recommended configuration for the given problem characteristics.
     */
    public static VectorizedGaussianParameters recommend(int expectedCategories, int inputDimension, String priority) {
        return switch (priority.toLowerCase()) {
            case "speed", "performance" -> {
                var params = createHighPerformance();
                if (!params.isSuitableFor(expectedCategories, inputDimension)) {
                    yield params.withParallelismLevel(Math.max(2, params.parallelismLevel() / 2));
                }
                yield params;
            }
            case "memory" -> createMemoryOptimized();
            case "realtime", "latency" -> {
                var params = createRealTime();
                if (!params.isSuitableFor(expectedCategories, inputDimension)) {
                    yield params.withSIMD(false).withParallelismLevel(2);
                }
                yield params;
            }
            case "research", "accuracy" -> createResearch();
            default -> createDefault();
        };
    }
    
    /**
     * Validate parameters for GaussianART algorithm constraints.
     */
    public void validateForGaussianART() {
        if (rho_b >= rho_a) {
            throw new IllegalArgumentException("rho_b (" + rho_b + ") must be < rho_a (" + rho_a + ") for proper variance management");
        }
        if (gamma > 0.5 && vigilance > 0.95) {
            // Note: High gamma with very high vigilance may cause instability
            // This combination should be avoided in production use
        }
    }
    
    /**
     * Get performance scaling estimate based on parallelism and SIMD settings.
     */
    public double getPerformanceScaling() {
        double baseScaling = 1.0;
        
        // Parallelism scaling (assumes diminishing returns)
        if (parallelismLevel > 1) {
            baseScaling *= Math.min(parallelismLevel * 0.8, parallelismLevel);
        }
        
        // SIMD scaling (typical 2-4x improvement for Gaussian operations)
        if (enableSIMD) {
            baseScaling *= 2.5;
        }
        
        return baseScaling;
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedGaussianParameters{vigilance=%.3f, gamma=%.3f, rho_a=%.3f, rho_b=%.3f, " +
                           "parallel=%d, simd=%s, scaling=%.1fx}",
                           vigilance, gamma, rho_a, rho_b, 
                           parallelismLevel, enableSIMD, getPerformanceScaling());
    }
}