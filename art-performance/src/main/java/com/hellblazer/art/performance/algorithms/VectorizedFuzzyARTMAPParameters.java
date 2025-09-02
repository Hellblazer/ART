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

import com.hellblazer.art.core.parameters.FuzzyARTMAPParameters;

/**
 * Parameters for VectorizedFuzzyARTMAP with performance optimization settings.
 * 
 * This record encapsulates configuration for supervised FuzzyARTMAP learning with:
 * - Core FuzzyARTMAP parameters (rho, alpha, beta, epsilon)
 * - Parallel processing settings for large category sets
 * - SIMD vectorization control for fuzzy operations
 * - Performance tuning for match tracking and map field management
 * 
 * All parameters are immutable and validated at construction to ensure
 * proper FuzzyARTMAP behavior and optimal performance characteristics.
 * 
 * @param rho vigilance parameter [0,1] - controls category granularity
 * @param alpha choice parameter > 0 - influences category selection
 * @param beta learning rate [0,1] - controls weight adaptation speed
 * @param epsilon match tracking increment > 0 - for conflict resolution
 * @param parallelismLevel number of parallel threads >= 1
 * @param enableSIMD enable SIMD vectorization for fuzzy operations
 * 
 * @author Hal Hildebrand
 */
public record VectorizedFuzzyARTMAPParameters(
    double rho,
    double alpha,
    double beta,
    double epsilon,
    int parallelismLevel,
    boolean enableSIMD
) {
    
    /**
     * Create VectorizedFuzzyARTMAPParameters with validation.
     * 
     * @param rho vigilance parameter [0,1]
     * @param alpha choice parameter > 0
     * @param beta learning rate [0,1]
     * @param epsilon match tracking increment > 0
     * @param parallelismLevel number of parallel threads >= 1
     * @param enableSIMD enable SIMD vectorization
     */
    public VectorizedFuzzyARTMAPParameters {
        if (rho < 0.0 || rho > 1.0) {
            throw new IllegalArgumentException("Vigilance (rho) must be in [0,1], got: " + rho);
        }
        if (alpha <= 0.0) {
            throw new IllegalArgumentException("Choice parameter (alpha) must be > 0, got: " + alpha);
        }
        if (beta < 0.0 || beta > 1.0) {
            throw new IllegalArgumentException("Learning rate (beta) must be in [0,1], got: " + beta);
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
     * Uses conservative settings suitable for most FuzzyARTMAP applications.
     * 
     * @return default VectorizedFuzzyARTMAPParameters
     */
    public static VectorizedFuzzyARTMAPParameters createDefault() {
        int processors = Runtime.getRuntime().availableProcessors();
        return new VectorizedFuzzyARTMAPParameters(
            0.8,                        // rho - moderate vigilance for good generalization
            0.001,                      // alpha - small choice parameter for stable selection
            1.0,                        // beta - full learning rate for fast adaptation
            1e-6,                       // epsilon - small increment for precise match tracking
            Math.max(2, processors/2),  // parallelismLevel - use half available cores
            true                        // enableSIMD - use vectorization by default
        );
    }
    
    /**
     * Create high-performance parameters for large-scale supervised learning.
     * Optimized for speed with aggressive parallelization and vectorization.
     * 
     * @return high-performance VectorizedFuzzyARTMAPParameters
     */
    public static VectorizedFuzzyARTMAPParameters createHighPerformance() {
        int processors = Runtime.getRuntime().availableProcessors();
        return new VectorizedFuzzyARTMAPParameters(
            0.75,                       // rho - lower vigilance for fewer categories
            0.001,                      // alpha - standard choice parameter
            0.8,                        // beta - slightly lower learning rate for stability
            1e-6,                       // epsilon - standard match tracking
            processors,                 // parallelismLevel - use all available cores
            true                        // enableSIMD - maximum vectorization
        );
    }
    
    /**
     * Create memory-optimized parameters for resource-constrained environments.
     * Minimizes memory usage while maintaining reasonable learning performance.
     * 
     * @return memory-optimized VectorizedFuzzyARTMAPParameters
     */
    public static VectorizedFuzzyARTMAPParameters createMemoryOptimized() {
        return new VectorizedFuzzyARTMAPParameters(
            0.85,                       // rho - higher vigilance for fewer, more specific categories
            0.001,                      // alpha - standard choice parameter
            1.0,                        // beta - full learning rate for quick convergence
            1e-6,                       // epsilon - standard match tracking
            1,                          // parallelismLevel - sequential processing
            false                       // enableSIMD - disable vectorization to save memory
        );
    }
    
    /**
     * Create real-time parameters optimized for low-latency supervised learning.
     * Balances speed and accuracy for interactive applications.
     * 
     * @return real-time VectorizedFuzzyARTMAPParameters
     */
    public static VectorizedFuzzyARTMAPParameters createRealTime() {
        return new VectorizedFuzzyARTMAPParameters(
            0.7,                        // rho - lower vigilance for faster convergence
            0.01,                       // alpha - higher choice parameter for quicker selection
            0.9,                        // beta - high learning rate for rapid adaptation
            1e-5,                       // epsilon - larger increment for faster match tracking
            4,                          // parallelismLevel - moderate parallelism
            true                        // enableSIMD - vectorization for speed
        );
    }
    
    /**
     * Create conservative parameters for high-accuracy supervised learning.
     * Prioritizes classification accuracy over speed.
     * 
     * @return conservative VectorizedFuzzyARTMAPParameters
     */
    public static VectorizedFuzzyARTMAPParameters createConservative() {
        return new VectorizedFuzzyARTMAPParameters(
            0.9,                        // rho - high vigilance for precise categories
            0.0001,                     // alpha - very small choice parameter for careful selection
            0.5,                        // beta - moderate learning rate for stability
            1e-7,                       // epsilon - very small increment for precise match tracking
            2,                          // parallelismLevel - minimal parallelism for consistency
            true                        // enableSIMD - vectorization for computational precision
        );
    }
    
    /**
     * Convert to equivalent FuzzyARTMAPParameters for compatibility.
     * 
     * @return equivalent FuzzyARTMAPParameters
     */
    public FuzzyARTMAPParameters toFuzzyARTMAPParameters() {
        return new FuzzyARTMAPParameters(rho, alpha, beta, epsilon);
    }
    
    /**
     * Create a new VectorizedFuzzyARTMAPParameters with modified vigilance.
     * 
     * @param newRho new vigilance parameter [0,1]
     * @return new parameters with modified vigilance
     */
    public VectorizedFuzzyARTMAPParameters withRho(double newRho) {
        return new VectorizedFuzzyARTMAPParameters(
            newRho, alpha, beta, epsilon, parallelismLevel, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedFuzzyARTMAPParameters with modified choice parameter.
     * 
     * @param newAlpha new choice parameter > 0
     * @return new parameters with modified choice parameter
     */
    public VectorizedFuzzyARTMAPParameters withAlpha(double newAlpha) {
        return new VectorizedFuzzyARTMAPParameters(
            rho, newAlpha, beta, epsilon, parallelismLevel, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedFuzzyARTMAPParameters with modified learning rate.
     * 
     * @param newBeta new learning rate [0,1]
     * @return new parameters with modified learning rate
     */
    public VectorizedFuzzyARTMAPParameters withBeta(double newBeta) {
        return new VectorizedFuzzyARTMAPParameters(
            rho, alpha, newBeta, epsilon, parallelismLevel, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedFuzzyARTMAPParameters with modified match tracking increment.
     * 
     * @param newEpsilon new match tracking increment > 0
     * @return new parameters with modified match tracking increment
     */
    public VectorizedFuzzyARTMAPParameters withEpsilon(double newEpsilon) {
        return new VectorizedFuzzyARTMAPParameters(
            rho, alpha, beta, newEpsilon, parallelismLevel, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedFuzzyARTMAPParameters with modified parallelism level.
     * 
     * @param newParallelismLevel new parallelism level >= 1
     * @return new parameters with modified parallelism level
     */
    public VectorizedFuzzyARTMAPParameters withParallelismLevel(int newParallelismLevel) {
        return new VectorizedFuzzyARTMAPParameters(
            rho, alpha, beta, epsilon, newParallelismLevel, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedFuzzyARTMAPParameters with modified SIMD setting.
     * 
     * @param newEnableSIMD new SIMD vectorization setting
     * @return new parameters with modified SIMD setting
     */
    public VectorizedFuzzyARTMAPParameters withSIMD(boolean newEnableSIMD) {
        return new VectorizedFuzzyARTMAPParameters(
            rho, alpha, beta, epsilon, parallelismLevel, newEnableSIMD
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
        long baseUsage = 2000L; // Base overhead for ARTMAP structure
        
        // FuzzyART module memory (complement-coded dimension)
        int complementDim = inputDimension * 2;
        long categoryMemory = expectedCategories * complementDim * 8L; // double arrays
        
        // Map field memory (category -> label mapping)
        long mapFieldMemory = expectedCategories * 12L; // Integer map entries
        
        // Parallel processing overhead
        long parallelMemory = parallelismLevel * 1024L; // Thread pool overhead
        
        // SIMD cache if enabled
        long simdMemory = enableSIMD ? (complementDim * 4L * 2) : 0L; // float cache
        
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
        
        // Should not use more than 80% of available memory
        return memoryUsage < (availableMemory * 0.8);
    }
    
    /**
     * Get recommended configuration for the given supervised learning problem.
     * 
     * @param expectedCategories estimated number of categories
     * @param inputDimension input pattern dimension
     * @param samples number of training samples
     * @param numClasses number of distinct class labels
     * @param priority optimization priority ("speed", "memory", "accuracy", "realtime")
     * @return recommended VectorizedFuzzyARTMAPParameters
     */
    public static VectorizedFuzzyARTMAPParameters recommend(
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
     * Validate parameters for FuzzyARTMAP constraints.
     * 
     * @throws IllegalStateException if parameters violate FuzzyARTMAP requirements
     */
    public void validate() {
        if (rho < 0.0 || rho > 1.0) {
            throw new IllegalStateException("Invalid vigilance: " + rho + " (must be [0,1])");
        }
        if (alpha <= 0.0) {
            throw new IllegalStateException("Invalid choice parameter: " + alpha + " (must be > 0)");
        }
        if (beta < 0.0 || beta > 1.0) {
            throw new IllegalStateException("Invalid learning rate: " + beta + " (must be [0,1])");
        }
        if (epsilon <= 0.0) {
            throw new IllegalStateException("Invalid match tracking: " + epsilon + " (must be > 0)");
        }
        if (parallelismLevel < 1) {
            throw new IllegalStateException("Invalid parallelism: " + parallelismLevel + " (must be >= 1)");
        }
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedFuzzyARTMAPParameters{rho=%.3f, alpha=%.6f, beta=%.3f, " +
                           "epsilon=%.2e, parallel=%d, simd=%s}",
                           rho, alpha, beta, epsilon, parallelismLevel, enableSIMD);
    }
}