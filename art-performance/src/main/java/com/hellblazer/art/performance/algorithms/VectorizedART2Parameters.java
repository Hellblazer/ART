package com.hellblazer.art.performance.algorithms;

import java.util.Objects;

/**
 * Parameters for VectorizedART2 with ART2-specific preprocessing controls.
 * 
 * ART2 is designed for continuous analog input patterns and includes
 * preprocessing parameters for contrast enhancement and noise suppression:
 * - vigilance: Standard ART vigilance parameter [0,1]
 * - theta: Contrast enhancement parameter [0,1]
 * - epsilon: Noise suppression parameter [0,1]
 * - parallelismLevel: Number of parallel threads for processing
 * - enableSIMD: Whether to use SIMD vectorization
 * 
 * All parameters are immutable and validated at construction.
 */
public record VectorizedART2Parameters(
    double vigilance,
    double theta,
    double epsilon,
    int parallelismLevel,
    boolean enableSIMD
) {
    
    /**
     * Create VectorizedART2Parameters with validation.
     */
    public VectorizedART2Parameters {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in [0,1], got: " + vigilance);
        }
        if (theta < 0.0 || theta > 1.0) {
            throw new IllegalArgumentException("Theta must be in [0,1], got: " + theta);
        }
        if (epsilon < 0.0 || epsilon > 1.0) {
            throw new IllegalArgumentException("Epsilon must be in [0,1], got: " + epsilon);
        }
        if (parallelismLevel < 1) {
            throw new IllegalArgumentException("Parallelism level must be >= 1, got: " + parallelismLevel);
        }
    }
    
    /**
     * Create default parameters optimized for ART2.
     */
    public static VectorizedART2Parameters createDefault() {
        int processors = Runtime.getRuntime().availableProcessors();
        return new VectorizedART2Parameters(
            0.75,                      // vigilance - moderate discrimination
            0.1,                       // theta - mild contrast enhancement
            0.001,                     // epsilon - minimal noise suppression
            Math.max(2, processors/2), // parallelismLevel - use half available cores
            true                       // enableSIMD
        );
    }
    
    /**
     * Create high-vigilance parameters for fine discrimination.
     */
    public static VectorizedART2Parameters createHighVigilance() {
        int processors = Runtime.getRuntime().availableProcessors();
        return new VectorizedART2Parameters(
            0.9,                       // High vigilance for fine categories
            0.2,                       // Moderate contrast enhancement
            0.005,                     // Low noise suppression to preserve detail
            processors,                // Use all available cores
            true                       // enableSIMD for performance
        );
    }
    
    /**
     * Create low-vigilance parameters for coarse categorization.
     */
    public static VectorizedART2Parameters createLowVigilance() {
        return new VectorizedART2Parameters(
            0.3,                       // Low vigilance for broad categories
            0.05,                      // Minimal contrast enhancement
            0.01,                      // Standard noise suppression
            2,                         // Minimal parallelism
            true                       // enableSIMD
        );
    }
    
    /**
     * Create parameters optimized for noisy input data.
     */
    public static VectorizedART2Parameters createNoiseRobust() {
        return new VectorizedART2Parameters(
            0.6,                       // Moderate vigilance
            0.3,                       // Strong contrast enhancement
            0.1,                       // Significant noise suppression
            4,                         // Moderate parallelism
            true                       // enableSIMD
        );
    }
    
    /**
     * Create parameters for high-contrast input data.
     */
    public static VectorizedART2Parameters createHighContrast() {
        return new VectorizedART2Parameters(
            0.8,                       // High vigilance to distinguish sharp features
            0.05,                      // Minimal contrast enhancement (already high)
            0.001,                     // Minimal noise suppression
            4,                         // Moderate parallelism
            true                       // enableSIMD
        );
    }
    
    /**
     * Create parameters for real-time processing.
     */
    public static VectorizedART2Parameters createRealTime() {
        int processors = Runtime.getRuntime().availableProcessors();
        return new VectorizedART2Parameters(
            0.7,                       // Balanced vigilance
            0.1,                       // Standard contrast enhancement
            0.01,                      // Standard noise suppression
            processors,                // Use all cores for speed
            true                       // enableSIMD for maximum performance
        );
    }
    
    /**
     * Create parameters for memory-constrained environments.
     */
    public static VectorizedART2Parameters createMemoryOptimized() {
        return new VectorizedART2Parameters(
            0.75,                      // Standard vigilance
            0.1,                       // Standard contrast enhancement
            0.001,                     // Minimal noise suppression
            1,                         // Single thread to save memory
            false                      // Disable SIMD to save memory
        );
    }
    
    /**
     * Create a new VectorizedART2Parameters with modified vigilance.
     */
    public VectorizedART2Parameters withVigilance(double newVigilance) {
        return new VectorizedART2Parameters(newVigilance, theta, epsilon, parallelismLevel, enableSIMD);
    }
    
    /**
     * Create a new VectorizedART2Parameters with modified theta.
     */
    public VectorizedART2Parameters withTheta(double newTheta) {
        return new VectorizedART2Parameters(vigilance, newTheta, epsilon, parallelismLevel, enableSIMD);
    }
    
    /**
     * Create a new VectorizedART2Parameters with modified epsilon.
     */
    public VectorizedART2Parameters withEpsilon(double newEpsilon) {
        return new VectorizedART2Parameters(vigilance, theta, newEpsilon, parallelismLevel, enableSIMD);
    }
    
    /**
     * Create a new VectorizedART2Parameters with modified parallelism level.
     */
    public VectorizedART2Parameters withParallelismLevel(int newParallelismLevel) {
        return new VectorizedART2Parameters(vigilance, theta, epsilon, newParallelismLevel, enableSIMD);
    }
    
    /**
     * Create a new VectorizedART2Parameters with modified SIMD setting.
     */
    public VectorizedART2Parameters withSIMD(boolean newEnableSIMD) {
        return new VectorizedART2Parameters(vigilance, theta, epsilon, parallelismLevel, newEnableSIMD);
    }
    
    /**
     * Create a new VectorizedART2Parameters with modified preprocessing settings.
     */
    public VectorizedART2Parameters withPreprocessing(double newTheta, double newEpsilon) {
        return new VectorizedART2Parameters(vigilance, newTheta, newEpsilon, parallelismLevel, enableSIMD);
    }
    
    /**
     * Check if these parameters are suitable for the given problem characteristics.
     */
    public boolean isSuitableFor(int expectedCategories, int inputDimension, boolean hasNoise) {
        // High epsilon for noisy data
        if (hasNoise && epsilon < 0.01) {
            return false;
        }
        
        // High vigilance may create too many categories
        if (vigilance > 0.9 && expectedCategories < inputDimension) {
            return false;
        }
        
        // Low vigilance may create too few categories
        if (vigilance < 0.3 && expectedCategories > inputDimension * 2) {
            return false;
        }
        
        return true;
    }
    
    /**
     * Get recommended configuration for the given problem characteristics.
     */
    public static VectorizedART2Parameters recommend(int expectedCategories, int inputDimension, 
                                                     boolean hasNoise, String priority) {
        return switch (priority.toLowerCase()) {
            case "accuracy", "precision" -> {
                if (hasNoise) {
                    yield createNoiseRobust().withVigilance(0.8);
                } else {
                    yield createHighVigilance();
                }
            }
            case "speed", "performance" -> createRealTime();
            case "memory" -> createMemoryOptimized();
            case "robustness" -> createNoiseRobust();
            case "contrast" -> createHighContrast();
            default -> {
                if (hasNoise) {
                    yield createNoiseRobust();
                } else if (expectedCategories > inputDimension * 2) {
                    yield createLowVigilance();
                } else {
                    yield createDefault();
                }
            }
        };
    }
    
    /**
     * Calculate effective contrast enhancement for given input characteristics.
     */
    public double getEffectiveContrast(double inputVariance) {
        // Higher theta is more effective for low-variance inputs
        if (inputVariance < 0.1) {
            return Math.min(1.0, theta * 2.0);
        } else if (inputVariance > 0.5) {
            return Math.max(0.01, theta * 0.5);
        } else {
            return theta;
        }
    }
    
    /**
     * Calculate effective noise suppression for given input characteristics.
     */
    public double getEffectiveNoiseSupression(double inputSignalToNoise) {
        // Higher epsilon for lower signal-to-noise ratio
        if (inputSignalToNoise < 2.0) {
            return Math.min(1.0, epsilon * 10.0);
        } else if (inputSignalToNoise > 10.0) {
            return Math.max(0.001, epsilon * 0.1);
        } else {
            return epsilon;
        }
    }
    
    /**
     * Get estimated memory usage in bytes for the given problem size.
     */
    public long getEstimatedMemoryUsage(int expectedCategories, int inputDimension) {
        long baseUsage = 1000L; // Base overhead
        long categoryUsage = expectedCategories * inputDimension * 8L; // double arrays for weights
        long parallelUsage = parallelismLevel * 1000L; // Thread overhead
        long simdUsage = enableSIMD ? (inputDimension * 4L) : 0L; // Float arrays for SIMD
        
        return baseUsage + categoryUsage + parallelUsage + simdUsage;
    }
    
    /**
     * Validate parameters against ART2 algorithm constraints.
     */
    public void validateART2Constraints() {
        // Theta should be small to avoid over-enhancement
        if (theta > 0.5) {
            throw new IllegalArgumentException("Theta > 0.5 may cause over-enhancement in ART2");
        }
        
        // Very high epsilon can suppress valid signals
        if (epsilon > 0.2) {
            throw new IllegalArgumentException("Epsilon > 0.2 may suppress valid signals in ART2");
        }
        
        // Very low vigilance may prevent category formation
        if (vigilance < 0.1) {
            throw new IllegalArgumentException("Vigilance < 0.1 may prevent proper categorization in ART2");
        }
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedART2Parameters{vigilance=%.3f, theta=%.3f, epsilon=%.3f, " +
                           "parallel=%d, simd=%s}",
                           vigilance, theta, epsilon, parallelismLevel, enableSIMD);
    }
}