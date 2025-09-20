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
package com.hellblazer.art.core;

import com.hellblazer.art.core.results.ActivationResult;
import java.util.List;

/**
 * Unified interface for all Adaptive Resonance Theory (ART) algorithm implementations.
 * 
 * This interface provides a common API for both vanilla and high-performance vectorized
 * implementations, ensuring consistency across the codebase and enabling easy swapping
 * between implementations for testing, benchmarking, or deployment scenarios.
 * 
 * @param <P> the type of parameters used by the algorithm (e.g., FuzzyParameters, GaussianParameters)
 * 
 * @author Hal Hildebrand
 * @since 1.0
 */
public interface ARTAlgorithm<P> extends AutoCloseable {
    
    /**
     * Train the network with a single pattern (online learning).
     * This method performs one step of the ART learning algorithm, potentially
     * creating a new category or updating existing weights.
     * 
     * @param input the input pattern to learn
     * @param parameters the algorithm-specific parameters
     * @return the result of the learning step, including category assignment and match score
     * @throws IllegalArgumentException if input or parameters are invalid
     */
    ActivationResult learn(Pattern input, P parameters);
    
    /**
     * Predict the category for an input pattern without modifying the network.
     * This performs a forward pass through the network to find the best matching category.
     * 
     * @param input the input pattern to classify
     * @param parameters the algorithm-specific parameters
     * @return the prediction result, including the matched category and activation score
     * @throws IllegalArgumentException if input or parameters are invalid
     */
    ActivationResult predict(Pattern input, P parameters);
    
    /**
     * Get the current number of categories (clusters) in the network.
     * This represents the number of distinct patterns the network has learned.
     * 
     * @return the number of categories, >= 0
     */
    int getCategoryCount();
    
    /**
     * Get all category weight vectors.
     * This provides access to the learned representations for analysis or persistence.
     * 
     * @return an unmodifiable list of weight vectors
     */
    List<WeightVector> getCategories();
    
    /**
     * Get a specific category weight vector by index.
     * 
     * @param index the category index (0-based)
     * @return the weight vector for the specified category
     * @throws IndexOutOfBoundsException if index is invalid
     */
    WeightVector getCategory(int index);
    
    /**
     * Clear all learned categories and reset the network to its initial state.
     * After this call, getCategoryCount() will return 0.
     */
    void clear();
    
    // Inherits close() from AutoCloseable
    // Vanilla implementations can use default AutoCloseable no-op
    // Vectorized implementations should override to release SIMD resources
    
    /**
     * Optional: Get performance statistics if available.
     * Implementations that don't track performance can return null.
     * 
     * @return performance statistics or null if not tracked
     */
    default Object getPerformanceStats() {
        return null;
    }
    
    /**
     * Optional: Reset performance tracking if available.
     * Implementations that don't track performance can ignore this.
     */
    default void resetPerformanceTracking() {
        // Default no-op
    }
    
    /**
     * Optional: Batch learning for multiple patterns.
     * Default implementation calls learn() for each pattern sequentially.
     * Vectorized implementations may override for better performance.
     * 
     * @param patterns the input patterns to learn
     * @param parameters the algorithm-specific parameters
     * @return results for each pattern in the same order
     */
    default List<ActivationResult> learnBatch(List<Pattern> patterns, P parameters) {
        return patterns.stream()
                      .map(p -> learn(p, parameters))
                      .toList();
    }
    
    /**
     * Optional: Batch prediction for multiple patterns.
     * Default implementation calls predict() for each pattern sequentially.
     * Vectorized implementations may override for better performance.
     * 
     * @param patterns the input patterns to classify
     * @param parameters the algorithm-specific parameters
     * @return results for each pattern in the same order
     */
    default List<ActivationResult> predictBatch(List<Pattern> patterns, P parameters) {
        return patterns.stream()
                      .map(p -> predict(p, parameters))
                      .toList();
    }
    
    /**
     * Optional: Get algorithm-specific metadata.
     * This can include information like algorithm name, version, capabilities, etc.
     * 
     * @return metadata string or null if not provided
     */
    default String getAlgorithmInfo() {
        return this.getClass().getSimpleName();
    }
    
    /**
     * Optional: Check if this implementation supports SIMD optimization.
     * 
     * @return true if SIMD is supported and enabled, false otherwise
     */
    default boolean isSIMDEnabled() {
        return false;
    }
    
    /**
     * Optional: Check if this implementation supports parallel processing.
     * 
     * @return true if parallel processing is supported and enabled, false otherwise
     */
    default boolean isParallelEnabled() {
        return false;
    }
}