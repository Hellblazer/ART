/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 * 
 * This file is part of Java ART Neural Networks.
 * 
 * Java ART Neural Networks is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Java ART Neural Networks is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with Java ART Neural Networks. If not, see <https://www.gnu.org/licenses/>.
 */
package com.hellblazer.art.performance.algorithms;

/**
 * Parameters for VectorizedGaussianARTMAP with performance optimization settings.
 * 
 * This record encapsulates configuration for supervised GaussianARTMAP learning with:
 * - Core GaussianARTMAP parameters (vigilance, gamma, rho_a, rho_b, epsilon)
 * - Parallel processing settings for large category sets
 * - SIMD vectorization control for Gaussian operations
 * - Performance tuning for match tracking and map field management
 * 
 * All parameters are immutable and validated at construction to ensure
 * proper GaussianARTMAP behavior and optimal performance characteristics.
 * 
 * @param vigilance vigilance parameter [0,1] - probability threshold for accepting patterns
 * @param gamma learning rate [0,1] - controls adaptation speed
 * @param rho_a variance adjustment factor > 0 (typically 1.0)
 * @param rho_b minimum variance threshold > 0 to prevent collapse
 * @param epsilon match tracking increment > 0 - for conflict resolution
 * @param parallelismLevel number of parallel threads >= 1
 * @param enableSIMD enable SIMD vectorization for Gaussian operations
 * 
 * @author Hal Hildebrand
 */
public record VectorizedGaussianARTMAPParameters(
    double vigilance,
    double gamma,
    double rho_a,
    double rho_b,
    double epsilon,
    int parallelismLevel,
    boolean enableSIMD
) {
    
    /**
     * Create VectorizedGaussianARTMAPParameters with validation.
     * 
     * @param vigilance vigilance parameter [0,1]
     * @param gamma learning rate [0,1]
     * @param rho_a variance adjustment factor > 0
     * @param rho_b minimum variance threshold > 0
     * @param epsilon match tracking increment > 0
     * @param parallelismLevel number of parallel threads >= 1
     * @param enableSIMD enable SIMD vectorization
     */
    public VectorizedGaussianARTMAPParameters {
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
        if (epsilon <= 0.0) {
            throw new IllegalArgumentException("Match tracking increment (epsilon) must be > 0, got: " + epsilon);
        }
        if (parallelismLevel < 1) {
            throw new IllegalArgumentException("Parallelism level must be >= 1, got: " + parallelismLevel);
        }
    }
    
    /**
     * Create default parameters optimized for the current system.
     * Uses conservative settings suitable for most GaussianARTMAP applications.
     * 
     * @return default VectorizedGaussianARTMAPParameters
     */
    public static VectorizedGaussianARTMAPParameters createDefault() {
        int processors = Runtime.getRuntime().availableProcessors();
        return new VectorizedGaussianARTMAPParameters(
            0.75,                       // vigilance - moderate discrimination
            0.1,                        // gamma - conservative learning rate
            1.0,                        // rho_a - standard variance adjustment
            0.1,                        // rho_b - minimum variance to prevent collapse
            1e-6,                       // epsilon - small increment for precise match tracking
            Math.max(2, processors/2),  // parallelismLevel - use half available cores
            true                        // enableSIMD - use vectorization by default
        );
    }
    
    /**
     * Create high-performance parameters for large-scale supervised learning.
     * Optimized for speed with aggressive parallelization and vectorization.
     * 
     * @return high-performance VectorizedGaussianARTMAPParameters
     */
    public static VectorizedGaussianARTMAPParameters createHighPerformance() {
        int processors = Runtime.getRuntime().availableProcessors();
        return new VectorizedGaussianARTMAPParameters(
            0.8,                        // vigilance - higher discrimination for better categories
            0.05,                       // gamma - lower learning rate for stability
            1.0,                        // rho_a - standard variance adjustment
            0.05,                       // rho_b - lower minimum variance for tighter clusters
            1e-6,                       // epsilon - standard match tracking
            processors,                 // parallelismLevel - use all available cores
            true                        // enableSIMD - maximum vectorization
        );
    }
    
    /**
     * Create memory-optimized parameters for resource-constrained environments.
     * Minimizes memory usage while maintaining reasonable learning performance.
     * 
     * @return memory-optimized VectorizedGaussianARTMAPParameters
     */
    public static VectorizedGaussianARTMAPParameters createMemoryOptimized() {
        return new VectorizedGaussianARTMAPParameters(
            0.7,                        // vigilance - standard discrimination
            0.15,                       // gamma - higher learning rate for faster convergence
            1.0,                        // rho_a - standard variance adjustment
            0.2,                        // rho_b - higher minimum variance to reduce categories
            1e-6,                       // epsilon - standard match tracking
            2,                          // parallelismLevel - minimal parallelism
            false                       // enableSIMD - disable vectorization to save memory
        );
    }
    
    /**
     * Create real-time parameters optimized for low-latency supervised learning.
     * Balances speed and accuracy for interactive applications.
     * 
     * @return real-time VectorizedGaussianARTMAPParameters
     */
    public static VectorizedGaussianARTMAPParameters createRealTime() {
        return new VectorizedGaussianARTMAPParameters(
            0.75,                       // vigilance - standard discrimination
            0.2,                        // gamma - high learning rate for quick adaptation
            1.2,                        // rho_a - slightly higher variance adjustment for flexibility
            0.15,                       // rho_b - moderate minimum variance
            1e-5,                       // epsilon - larger increment for faster match tracking
            4,                          // parallelismLevel - moderate parallelism
            true                        // enableSIMD - vectorization for speed
        );
    }
    
    /**
     * Create conservative parameters for high-accuracy supervised learning.
     * Prioritizes classification accuracy over speed.
     * 
     * @return conservative VectorizedGaussianARTMAPParameters
     */
    public static VectorizedGaussianARTMAPParameters createConservative() {
        return new VectorizedGaussianARTMAPParameters(
            0.9,                        // vigilance - high discrimination for precise categories
            0.01,                       // gamma - very low learning rate for stability
            1.0,                        // rho_a - standard variance adjustment
            0.01,                       // rho_b - very low minimum variance for tight clusters
            1e-7,                       // epsilon - very small increment for precise match tracking
            1,                          // parallelismLevel - single-threaded for reproducibility
            false                       // enableSIMD - disable vectorization for exact reproducibility
        );
    }
    
    /**
     * Convert to VectorizedGaussianParameters for the underlying GaussianART module.
     * 
     * @return equivalent VectorizedGaussianParameters
     */
    public VectorizedGaussianParameters toGaussianParameters() {
        return new VectorizedGaussianParameters(
            vigilance, gamma, rho_a, rho_b, parallelismLevel, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedGaussianARTMAPParameters with modified vigilance.
     * 
     * @param newVigilance new vigilance parameter [0,1]
     * @return new parameters with modified vigilance
     */
    public VectorizedGaussianARTMAPParameters withVigilance(double newVigilance) {
        return new VectorizedGaussianARTMAPParameters(
            newVigilance, gamma, rho_a, rho_b, epsilon, parallelismLevel, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedGaussianARTMAPParameters with modified gamma (learning rate).
     * 
     * @param newGamma new learning rate [0,1]
     * @return new parameters with modified learning rate
     */
    public VectorizedGaussianARTMAPParameters withGamma(double newGamma) {
        return new VectorizedGaussianARTMAPParameters(
            vigilance, newGamma, rho_a, rho_b, epsilon, parallelismLevel, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedGaussianARTMAPParameters with modified rho_a.
     * 
     * @param newRhoA new variance adjustment factor > 0
     * @return new parameters with modified rho_a
     */
    public VectorizedGaussianARTMAPParameters withRhoA(double newRhoA) {
        return new VectorizedGaussianARTMAPParameters(
            vigilance, gamma, newRhoA, rho_b, epsilon, parallelismLevel, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedGaussianARTMAPParameters with modified rho_b.
     * 
     * @param newRhoB new minimum variance threshold > 0
     * @return new parameters with modified rho_b
     */
    public VectorizedGaussianARTMAPParameters withRhoB(double newRhoB) {
        return new VectorizedGaussianARTMAPParameters(
            vigilance, gamma, rho_a, newRhoB, epsilon, parallelismLevel, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedGaussianARTMAPParameters with modified match tracking increment.
     * 
     * @param newEpsilon new match tracking increment > 0
     * @return new parameters with modified match tracking increment
     */
    public VectorizedGaussianARTMAPParameters withEpsilon(double newEpsilon) {
        return new VectorizedGaussianARTMAPParameters(
            vigilance, gamma, rho_a, rho_b, newEpsilon, parallelismLevel, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedGaussianARTMAPParameters with modified parallelism level.
     * 
     * @param newParallelismLevel new parallelism level >= 1
     * @return new parameters with modified parallelism level
     */
    public VectorizedGaussianARTMAPParameters withParallelismLevel(int newParallelismLevel) {
        return new VectorizedGaussianARTMAPParameters(
            vigilance, gamma, rho_a, rho_b, epsilon, newParallelismLevel, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedGaussianARTMAPParameters with modified SIMD setting.
     * 
     * @param newEnableSIMD new SIMD vectorization setting
     * @return new parameters with modified SIMD setting
     */
    public VectorizedGaussianARTMAPParameters withSIMD(boolean newEnableSIMD) {
        return new VectorizedGaussianARTMAPParameters(
            vigilance, gamma, rho_a, rho_b, epsilon, parallelismLevel, newEnableSIMD
        );
    }
    
    /**
     * Get estimated memory usage for given problem size.
     * 
     * @param expectedCategories estimated number of categories
     * @param inputDimension input pattern dimension
     * @param samples number of training samples
     * @return estimated memory usage in bytes
     */
    public long getEstimatedMemoryUsage(int expectedCategories, int inputDimension, int samples) {
        long baseUsage = 2000L; // Base overhead for GaussianARTMAP structure
        
        // GaussianART module memory (mean vectors + covariance matrices)
        long meanMemory = expectedCategories * inputDimension * 8L; // double arrays
        long covarianceMemory = expectedCategories * inputDimension * inputDimension * 8L; // double matrices
        long categoryMemory = meanMemory + covarianceMemory;
        
        // Map field memory (category -> label mapping)
        long mapFieldMemory = expectedCategories * 12L; // Integer map entries
        
        // Parallel processing overhead
        long parallelMemory = parallelismLevel * 1024L; // Thread pool overhead
        
        // SIMD cache if enabled (larger for Gaussian operations)
        long simdMemory = enableSIMD ? (inputDimension * 8L * 4) : 0L; // double cache
        
        return baseUsage + categoryMemory + mapFieldMemory + parallelMemory + simdMemory;
    }
    
    /**
     * Check if this configuration is suitable for the given problem size.
     * 
     * @param expectedCategories estimated number of categories
     * @param inputDimension input pattern dimension
     * @param samples number of training samples
     * @return true if configuration is suitable for problem size
     */
    public boolean isSuitableFor(int expectedCategories, int inputDimension, int samples) {
        long memoryUsage = getEstimatedMemoryUsage(expectedCategories, inputDimension, samples);
        long availableMemory = Runtime.getRuntime().maxMemory();
        
        // Should not use more than 70% of available memory (Gaussian ops are memory-intensive)
        return memoryUsage < (availableMemory * 0.7);
    }
    
    /**
     * Get recommended configuration for the given supervised learning problem.
     * 
     * @param expectedCategories estimated number of categories
     * @param inputDimension input pattern dimension
     * @param samples number of training samples
     * @param numClasses number of distinct class labels
     * @param priority optimization priority (\"speed\", \"memory\", \"accuracy\", \"realtime\")
     * @return recommended VectorizedGaussianARTMAPParameters
     */
    public static VectorizedGaussianARTMAPParameters recommend(
            int expectedCategories, int inputDimension, int samples, int numClasses, String priority) {
        
        return switch (priority.toLowerCase()) {
            case "speed", "performance" -> {
                var params = createHighPerformance();
                if (!params.isSuitableFor(expectedCategories, inputDimension, samples)) {
                    // Reduce parallelism if memory constrained
                    yield params.withParallelismLevel(Math.max(1, params.parallelismLevel / 2));
                }
                yield params;
            }
            case "memory" -> createMemoryOptimized();
            case "realtime", "latency" -> {
                var params = createRealTime();
                if (!params.isSuitableFor(expectedCategories, inputDimension, samples)) {
                    // Disable SIMD if memory constrained
                    yield params.withSIMD(false);
                }
                yield params;
            }
            case "accuracy", "precision" -> createConservative();
            default -> {
                var params = createDefault();
                if (!params.isSuitableFor(expectedCategories, inputDimension, samples)) {
                    // Fall back to memory-optimized if default won't fit
                    yield createMemoryOptimized();
                }
                yield params;
            }
        };
    }
    
    /**
     * Validate parameters for GaussianARTMAP constraints.
     * 
     * @throws IllegalStateException if parameters violate GaussianARTMAP requirements
     */
    public void validate() {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalStateException("Invalid vigilance: " + vigilance + " (must be [0,1])");
        }
        if (gamma < 0.0 || gamma > 1.0) {
            throw new IllegalStateException("Invalid learning rate: " + gamma + " (must be [0,1])");
        }
        if (rho_a <= 0.0) {
            throw new IllegalStateException("Invalid rho_a: " + rho_a + " (must be > 0)");
        }
        if (rho_b <= 0.0) {
            throw new IllegalStateException("Invalid rho_b: " + rho_b + " (must be > 0)");
        }
        if (rho_b >= rho_a) {
            throw new IllegalStateException("rho_b (" + rho_b + ") must be < rho_a (" + rho_a + ") for proper variance management");
        }
        if (epsilon <= 0.0) {
            throw new IllegalStateException("Invalid match tracking: " + epsilon + " (must be > 0)");
        }
        if (parallelismLevel < 1) {
            throw new IllegalStateException("Invalid parallelism: " + parallelismLevel + " (must be >= 1)");
        }
        
        // Note: High gamma with very high vigilance may cause instability
        // This combination should be avoided in production use
    }
    
    /**
     * Get performance scaling estimate based on parallelism and SIMD settings.
     */
    public double getPerformanceScaling() {
        double baseScaling = 1.0;
        
        // Parallelism scaling (assumes diminishing returns)
        if (parallelismLevel > 1) {
            baseScaling *= Math.min(parallelismLevel * 0.75, parallelismLevel);
        }
        
        // SIMD scaling (Gaussian operations benefit significantly)
        if (enableSIMD) {
            baseScaling *= 3.0; // Gaussian PDFs vectorize well
        }
        
        return baseScaling;
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedGaussianARTMAPParameters{vigilance=%.3f, gamma=%.3f, rho_a=%.3f, rho_b=%.3f, " +
                           "epsilon=%.2e, parallel=%d, simd=%s, scaling=%.1fx}",
                           vigilance, gamma, rho_a, rho_b, epsilon, 
                           parallelismLevel, enableSIMD, getPerformanceScaling());
    }
}