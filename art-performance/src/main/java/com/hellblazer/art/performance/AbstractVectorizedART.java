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

import com.hellblazer.art.core.BaseART;
import com.hellblazer.art.core.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Abstract base class for all vectorized ART implementations.
 * 
 * Provides common vectorization infrastructure including:
 * - SIMD vector operations setup
 * - Performance tracking and metrics
 * - Thread pool management for parallel operations
 * - Caching infrastructure
 * - Resource management
 * 
 * Eliminates code duplication across vectorized algorithms by centralizing
 * common vectorization patterns and providing template methods for 
 * algorithm-specific operations.
 * 
 * Usage Pattern:
 * 1. Extend this class
 * 2. Implement abstract template methods
 * 3. Use provided utility methods for SIMD operations
 * 4. Focus only on algorithm-specific logic
 * 
 * @param <T> the type of performance statistics returned by this algorithm
 * @param <P> the type of parameters used by this algorithm
 */
public abstract class AbstractVectorizedART<T, P> extends BaseART 
        implements VectorizedARTAlgorithm<T, P> {
    
    private static final Logger log = LoggerFactory.getLogger(AbstractVectorizedART.class);
    
    // === SIMD Infrastructure ===
    protected static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    // === Performance Tracking ===
    private final AtomicLong totalVectorOperations = new AtomicLong(0);
    private final AtomicLong totalParallelTasks = new AtomicLong(0);
    private final AtomicLong activationCalls = new AtomicLong(0);
    private final AtomicLong matchCalls = new AtomicLong(0);
    private final AtomicLong learningCalls = new AtomicLong(0);
    private volatile double avgComputeTime = 0.0;
    
    // === Parallel Processing ===
    private final ForkJoinPool computePool;
    private final P defaultParameters;
    
    // === Caching Infrastructure ===
    private final ConcurrentHashMap<Integer, float[]> inputCache = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Object> algorithmCache = new ConcurrentHashMap<>();
    
    /**
     * Initialize the vectorized ART algorithm with default parameters.
     * 
     * @param defaultParameters the default parameters for this algorithm
     */
    protected AbstractVectorizedART(P defaultParameters) {
        super();
        this.defaultParameters = Objects.requireNonNull(defaultParameters, "Parameters cannot be null");
        
        // Extract parallelism level from parameters if available
        int parallelismLevel = extractParallelismLevel(defaultParameters);
        this.computePool = new ForkJoinPool(parallelismLevel);
        
        log.info("Initialized {} with {} parallel threads, vector species: {}", 
                 getClass().getSimpleName(), parallelismLevel, SPECIES.toString());
    }
    
    // === VectorizedARTAlgorithm Implementation ===
    
    @Override
    public final Object learn(Pattern input, P parameters) {
        learningCalls.incrementAndGet();
        var effectiveParams = parameters != null ? parameters : defaultParameters;
        return performVectorizedLearning(input, effectiveParams);
    }
    
    @Override
    public final Object predict(Pattern input, P parameters) {
        activationCalls.incrementAndGet();
        var effectiveParams = parameters != null ? parameters : defaultParameters;
        return performVectorizedPrediction(input, effectiveParams);
    }
    
    @Override
    public final T getPerformanceStats() {
        return createPerformanceStats(
            totalVectorOperations.get(),
            totalParallelTasks.get(),
            activationCalls.get(),
            matchCalls.get(),
            learningCalls.get(),
            avgComputeTime
        );
    }
    
    @Override
    public final void resetPerformanceTracking() {
        totalVectorOperations.set(0);
        totalParallelTasks.set(0);
        activationCalls.set(0);
        matchCalls.set(0);
        learningCalls.set(0);
        avgComputeTime = 0.0;
    }
    
    @Override
    public final P getParameters() {
        return defaultParameters;
    }
    
    @Override
    public final int getVectorSpeciesLength() {
        return SPECIES.length();
    }
    
    @Override
    public final void close() {
        try {
            computePool.shutdown();
            inputCache.clear();
            algorithmCache.clear();
            performCleanup();
        } catch (Exception e) {
            log.warn("Error during cleanup", e);
        }
    }
    
    // === Template Methods for Subclasses ===
    
    /**
     * Perform algorithm-specific vectorized learning.
     * 
     * @param input the input pattern
     * @param parameters the learning parameters
     * @return the learning result (typically category index)
     */
    protected abstract Object performVectorizedLearning(Pattern input, P parameters);
    
    /**
     * Perform algorithm-specific vectorized prediction.
     * 
     * @param input the input pattern
     * @param parameters the prediction parameters
     * @return the prediction result
     */
    protected abstract Object performVectorizedPrediction(Pattern input, P parameters);
    
    /**
     * Create algorithm-specific performance statistics.
     * 
     * @param vectorOps total vector operations
     * @param parallelTasks total parallel tasks
     * @param activations total activation calls
     * @param matches total match calls
     * @param learnings total learning calls
     * @param avgTime average compute time
     * @return performance statistics object
     */
    protected abstract T createPerformanceStats(
        long vectorOps, long parallelTasks, long activations, 
        long matches, long learnings, double avgTime);
    
    /**
     * Extract parallelism level from algorithm parameters.
     * Default implementation returns available processors.
     * 
     * @param parameters the algorithm parameters
     * @return the parallelism level to use
     */
    protected int extractParallelismLevel(P parameters) {
        // Try reflection to find parallelismLevel() method
        try {
            var method = parameters.getClass().getMethod("parallelismLevel");
            var result = method.invoke(parameters);
            if (result instanceof Integer level) {
                return level;
            }
        } catch (Exception e) {
            log.debug("Could not extract parallelism level from parameters", e);
        }
        return Runtime.getRuntime().availableProcessors();
    }
    
    /**
     * Perform algorithm-specific cleanup.
     * Default implementation does nothing.
     */
    protected void performCleanup() {
        // Override in subclasses if needed
    }
    
    // === Protected Utility Methods ===
    
    /**
     * Track a vector operation for performance monitoring.
     */
    protected final void trackVectorOperation() {
        totalVectorOperations.incrementAndGet();
    }
    
    /**
     * Track a parallel task for performance monitoring.
     */
    protected final void trackParallelTask() {
        totalParallelTasks.incrementAndGet();
    }
    
    /**
     * Track a match operation for performance monitoring.
     */
    protected final void trackMatchOperation() {
        matchCalls.incrementAndGet();
    }
    
    /**
     * Get the compute thread pool for parallel operations.
     */
    protected final ForkJoinPool getComputePool() {
        return computePool;
    }
    
    /**
     * Get the input cache for caching converted inputs.
     */
    protected final ConcurrentHashMap<Integer, float[]> getInputCache() {
        return inputCache;
    }
    
    /**
     * Get the algorithm-specific cache for custom caching needs.
     */
    protected final ConcurrentHashMap<String, Object> getAlgorithmCache() {
        return algorithmCache;
    }
    
    /**
     * Update average compute time using exponential moving average.
     */
    protected final void updateComputeTime(double newTime) {
        avgComputeTime = avgComputeTime == 0.0 ? newTime : (0.9 * avgComputeTime + 0.1 * newTime);
    }
    
    /**
     * Get cached or convert input pattern to float array for SIMD operations.
     */
    protected final float[] getCachedFloatArray(Pattern input) {
        return inputCache.computeIfAbsent(input.hashCode(), _ -> {
            var result = new float[input.dimension()];
            for (int i = 0; i < input.dimension(); i++) {
                result[i] = (float) input.get(i);
            }
            return result;
        });
    }
}