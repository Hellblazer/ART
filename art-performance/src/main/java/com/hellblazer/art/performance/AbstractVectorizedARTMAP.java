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

import com.hellblazer.art.core.BaseARTMAP;
import com.hellblazer.art.core.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;

/**
 * Abstract base class for high-performance vectorized ARTMAP implementations.
 * 
 * This class provides common infrastructure for supervised learning ARTMAP variants,
 * including performance tracking, parallel processing, and map field management.
 * Eliminates code duplication across ARTMAP implementations while maintaining
 * full performance and flexibility.
 * 
 * Key features provided:
 * - Performance tracking and monitoring
 * - Thread pool management for parallel operations
 * - Map field infrastructure (category-to-label associations)
 * - Training state management
 * - Resource lifecycle management
 * 
 * @param <P> the parameter type specific to the ARTMAP variant
 * 
 * @author Hal Hildebrand
 */
public abstract class AbstractVectorizedARTMAP<P> implements BaseARTMAP, AutoCloseable {
    
    private static final Logger log = LoggerFactory.getLogger(AbstractVectorizedARTMAP.class);
    
    // Map field infrastructure
    protected final Map<Integer, Integer> mapField;       // Map field: category -> label
    protected final Map<Integer, Set<Integer>> labelCategories; // Reverse map: label -> categories
    protected final Set<Integer> knownLabels;             // Set of encountered labels
    
    // Training state
    protected boolean trained = false;
    protected int[] trainingLabels;                       // Store all training labels
    
    // Parallel processing
    protected final ForkJoinPool computePool;
    
    // Performance tracking
    protected long totalSupervisedOperations = 0;
    protected long totalMatchTrackingEvents = 0;
    protected double avgSupervisedTime = 0.0;
    protected final Map<String, Long> operationCounts;
    
    /**
     * Create a new AbstractVectorizedARTMAP with specified parallelism level.
     * 
     * @param parallelismLevel number of parallel threads to use
     */
    protected AbstractVectorizedARTMAP(int parallelismLevel) {
        this.mapField = new ConcurrentHashMap<>();
        this.labelCategories = new ConcurrentHashMap<>();
        this.knownLabels = ConcurrentHashMap.newKeySet();
        this.computePool = new ForkJoinPool(parallelismLevel);
        this.operationCounts = new ConcurrentHashMap<>();
    }
    
    /**
     * Train the ARTMAP on labeled data.
     * 
     * @param data array of input patterns
     * @param labels array of class labels corresponding to each pattern
     * @param params parameters for training
     */
    public void fit(Pattern[] data, int[] labels, P params) {
        Objects.requireNonNull(data, "Data cannot be null");
        Objects.requireNonNull(labels, "Labels cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        if (data.length == 0) {
            throw new IllegalArgumentException("Cannot fit with empty data");
        }
        
        if (data.length != labels.length) {
            throw new IllegalArgumentException(
                "Data and labels must have same length: " + data.length + " != " + labels.length);
        }
        
        validateParameters(params);
        
        long startTime = System.nanoTime();
        
        // Clear existing state
        clear();
        
        // Store training labels
        this.trainingLabels = Arrays.copyOf(labels, labels.length);
        this.knownLabels.addAll(Arrays.stream(labels).boxed().toList());
        
        // Perform algorithm-specific training
        performSupervisedTraining(data, labels, params);
        
        this.trained = true;
        
        // Update performance tracking
        long duration = System.nanoTime() - startTime;
        updatePerformanceTracking("fit", duration);
        totalSupervisedOperations++;
        
        log.info("Training completed with {} patterns, {} categories, {} labels", 
                data.length, getCategoryCount(), knownLabels.size());
    }
    
    /**
     * Make predictions on new data.
     * 
     * @param data array of input patterns
     * @param params parameters for prediction
     * @return array of predicted labels
     */
    public int[] predict(Pattern[] data, P params) {
        Objects.requireNonNull(data, "Data cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        if (!trained) {
            throw new IllegalStateException("ARTMAP must be trained before prediction");
        }
        
        if (data.length == 0) {
            return new int[0];
        }
        
        validateParameters(params);
        
        long startTime = System.nanoTime();
        
        // Perform algorithm-specific prediction
        var predictions = performSupervisedPrediction(data, params);
        
        // Update performance tracking
        long duration = System.nanoTime() - startTime;
        updatePerformanceTracking("predict", duration);
        
        return predictions;
    }
    
    /**
     * Incremental learning with new labeled data.
     * 
     * @param data array of new input patterns
     * @param labels array of corresponding labels
     * @param params parameters for incremental learning
     */
    public void partialFit(Pattern[] data, int[] labels, P params) {
        Objects.requireNonNull(data, "Data cannot be null");
        Objects.requireNonNull(labels, "Labels cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        if (data.length != labels.length) {
            throw new IllegalArgumentException(
                "Data and labels must have same length: " + data.length + " != " + labels.length);
        }
        
        validateParameters(params);
        
        long startTime = System.nanoTime();
        
        // Add new labels to known set
        this.knownLabels.addAll(Arrays.stream(labels).boxed().toList());
        
        // Perform algorithm-specific incremental learning
        performIncrementalLearning(data, labels, params);
        
        this.trained = true;
        
        // Update performance tracking
        long duration = System.nanoTime() - startTime;
        updatePerformanceTracking("partialFit", duration);
        totalSupervisedOperations++;
    }
    
    /**
     * Perform match tracking for conflict resolution.
     * 
     * @param category the category that caused conflict
     * @param label the expected label
     * @param params current parameters
     * @return updated parameters with adjusted vigilance
     */
    protected P performMatchTracking(int category, int label, P params) {
        totalMatchTrackingEvents++;
        
        // Algorithm-specific match tracking implementation
        return adjustVigilanceForMatchTracking(category, label, params);
    }
    
    /**
     * Update performance tracking with operation timing.
     */
    private void updatePerformanceTracking(String operation, long durationNanos) {
        operationCounts.merge(operation, 1L, Long::sum);
        double durationMs = durationNanos / 1_000_000.0;
        avgSupervisedTime = (avgSupervisedTime + durationMs) / 2.0;
    }
    
    /**
     * Get performance statistics.
     * 
     * @return map of performance metrics
     */
    public Map<String, Object> getPerformanceStats() {
        var stats = new HashMap<String, Object>();
        stats.put("totalSupervisedOperations", totalSupervisedOperations);
        stats.put("totalMatchTrackingEvents", totalMatchTrackingEvents);
        stats.put("avgSupervisedTime", avgSupervisedTime);
        stats.put("operationCounts", new HashMap<>(operationCounts));
        stats.put("knownLabels", knownLabels.size());
        stats.put("categoryCount", getCategoryCount());
        return stats;
    }
    
    @Override
    public boolean isTrained() {
        return trained;
    }
    
    @Override
    public void clear() {
        mapField.clear();
        labelCategories.clear();
        knownLabels.clear();
        trainingLabels = null;
        trained = false;
        totalSupervisedOperations = 0;
        totalMatchTrackingEvents = 0;
        avgSupervisedTime = 0.0;
        operationCounts.clear();
        
        // Clear algorithm-specific state
        clearAlgorithmState();
    }
    
    @Override
    public void close() throws Exception {
        computePool.shutdown();
        
        // Close algorithm-specific resources
        closeAlgorithmResources();
    }
    
    // Abstract methods for algorithm-specific implementations
    
    /**
     * Validate algorithm-specific parameters.
     * 
     * @param params parameters to validate
     * @throws IllegalArgumentException if parameters are invalid
     */
    protected abstract void validateParameters(P params);
    
    /**
     * Perform algorithm-specific supervised training.
     * 
     * @param data training patterns
     * @param labels training labels
     * @param params training parameters
     */
    protected abstract void performSupervisedTraining(Pattern[] data, int[] labels, P params);
    
    /**
     * Perform algorithm-specific prediction.
     * 
     * @param data input patterns
     * @param params prediction parameters
     * @return predicted labels
     */
    protected abstract int[] performSupervisedPrediction(Pattern[] data, P params);
    
    /**
     * Perform algorithm-specific incremental learning.
     * 
     * @param data new training patterns
     * @param labels new training labels
     * @param params learning parameters
     */
    protected abstract void performIncrementalLearning(Pattern[] data, int[] labels, P params);
    
    /**
     * Adjust vigilance parameter for match tracking.
     * 
     * @param category conflicting category
     * @param label expected label
     * @param params current parameters
     * @return updated parameters
     */
    protected abstract P adjustVigilanceForMatchTracking(int category, int label, P params);
    
    /**
     * Clear algorithm-specific state.
     */
    protected abstract void clearAlgorithmState();
    
    /**
     * Close algorithm-specific resources.
     * 
     * @throws Exception if resource cleanup fails
     */
    protected abstract void closeAlgorithmResources() throws Exception;
}