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
package com.hellblazer.art.performance;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.ARTAlgorithm;
import com.hellblazer.art.core.results.ActivationResult;

/**
 * Common interface for high-performance vectorized ART algorithm implementations.
 * 
 * This interface provides a unified API for all vectorized ART algorithms, enabling
 * polymorphism and consistent usage patterns while allowing implementation flexibility.
 * Vectorized implementations use SIMD operations, parallel processing, and other
 * performance optimizations for significant speedups over scalar implementations.
 * 
 * Key Benefits:
 * - Polymorphic usage of different vectorized ART algorithms
 * - Consistent API across all vectorized implementations  
 * - Performance monitoring and resource management
 * - Support for both supervised and unsupervised learning patterns
 * 
 * Implementation Categories:
 * - Basic ART algorithms: VectorizedFuzzyART, VectorizedART, VectorizedHypersphereART
 * - Topological algorithms: VectorizedTopoART
 * - Supervised algorithms: VectorizedARTMAP
 * - Hierarchical algorithms: VectorizedDeepARTMAP
 * 
 * @param <T> the type of performance statistics returned by this algorithm
 * @param <P> the type of parameters used by this algorithm
 * 
 * @author Hal Hildebrand
 */
public interface VectorizedARTAlgorithm<T, P> extends ARTAlgorithm<P> {
    
    // === Core Learning Interface ===
    // Methods learn(), predict(), getCategoryCount() inherited from ARTAlgorithm<P>

    // === Category Management ===
    
    /**
     * Check if the algorithm has been trained/initialized with data.
     * 
     * @return true if the algorithm has learned from at least one pattern
     */
    default boolean isTrained() {
        return getCategoryCount() > 0;
    }
    
    // === Performance Monitoring ===
    
    /**
     * Get comprehensive performance statistics for this vectorized implementation.
     * Statistics typically include SIMD operation counts, timing metrics, 
     * memory usage, and algorithm-specific performance indicators.
     * 
     * @return performance statistics object
     */
    T getPerformanceStats();
    
    /**
     * Reset performance tracking counters to zero.
     * Useful for benchmarking specific operations or time periods.
     */
    void resetPerformanceTracking();
    
    // === Resource Management ===
    
    /**
     * Release resources and perform cleanup.
     * This includes shutting down thread pools, clearing caches,
     * and releasing any other held resources.
     */
    @Override
    void close();
    
    // === Algorithm Information ===
    
    /**
     * Get the algorithm type name for identification and debugging.
     * 
     * @return a string identifying this algorithm type
     */
    default String getAlgorithmType() {
        return this.getClass().getSimpleName();
    }
    
    /**
     * Get the current parameters being used by the algorithm.
     * 
     * @return the parameter object
     */
    P getParameters();
    
    /**
     * Check if this implementation uses SIMD vectorization.
     * 
     * @return true if SIMD operations are enabled and being used
     */
    default boolean isVectorized() {
        return true; // Default assumption for vectorized implementations
    }
    
    /**
     * Get the preferred SIMD vector species length for this algorithm.
     * Returns -1 if SIMD is not used or not applicable.
     * 
     * @return the vector species length, or -1 if not applicable
     */
    default int getVectorSpeciesLength() {
        return -1; // Default: not applicable
    }
    
    // === Optional Batch Operations ===
    // Batch methods learnBatch() and predictBatch() inherited from ARTAlgorithm<P>
}