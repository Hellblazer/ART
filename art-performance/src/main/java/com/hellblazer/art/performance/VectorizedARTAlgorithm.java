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
public interface VectorizedARTAlgorithm<T, P> extends AutoCloseable {
    
    // === Core Learning Interface ===
    
    /**
     * Learn from a single pattern, updating the algorithm's internal state.
     * This is the primary learning method for unsupervised algorithms.
     * 
     * @param input the input pattern to learn from
     * @param parameters the learning parameters
     * @return the category index that was activated/created, or result object
     */
    Object learn(Pattern input, P parameters);
    
    /**
     * Process a pattern for prediction/classification without learning.
     * Returns the best matching category or prediction result.
     * 
     * @param input the input pattern to process
     * @param parameters the processing parameters
     * @return the prediction result or category index
     */
    Object predict(Pattern input, P parameters);
    
    // === Category Management ===
    
    /**
     * Get the current number of categories/clusters learned by the algorithm.
     * 
     * @return the number of categories
     */
    int getCategoryCount();
    
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
    
    /**
     * Learn from multiple patterns in a batch operation.
     * Default implementation processes patterns sequentially.
     * Vectorized implementations may override for parallel batch processing.
     * 
     * @param patterns array of input patterns
     * @param parameters the learning parameters
     * @return array of learning results
     */
    default Object[] learnBatch(Pattern[] patterns, P parameters) {
        Object[] results = new Object[patterns.length];
        for (int i = 0; i < patterns.length; i++) {
            results[i] = learn(patterns[i], parameters);
        }
        return results;
    }
    
    /**
     * Predict for multiple patterns in a batch operation.
     * Default implementation processes patterns sequentially.
     * Vectorized implementations may override for parallel batch processing.
     * 
     * @param patterns array of input patterns
     * @param parameters the processing parameters
     * @return array of prediction results
     */
    default Object[] predictBatch(Pattern[] patterns, P parameters) {
        Object[] results = new Object[patterns.length];
        for (int i = 0; i < patterns.length; i++) {
            results[i] = predict(patterns[i], parameters);
        }
        return results;
    }
}