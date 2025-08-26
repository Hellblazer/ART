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

import com.hellblazer.art.core.artmap.DeepARTMAPParameters;

/**
 * Parameters for VectorizedDeepARTMAP with SIMD optimization settings.
 * 
 * This record extends DeepARTMAP parameters with vectorization-specific options
 * for controlling parallel processing, SIMD operations, and performance optimization.
 * 
 * Key Features:
 * - Channel-level parallelism control
 * - SIMD vector operation settings
 * - Memory optimization thresholds  
 * - Performance monitoring options
 * 
 * @param baseParameters Base DeepARTMAP parameters (vigilance thresholds, learning rates, etc.)
 * @param channelParallelismLevel Number of threads for parallel channel processing
 * @param channelParallelThreshold Minimum channels to trigger parallel processing
 * @param layerParallelismLevel Number of threads for parallel layer processing
 * @param layerParallelThreshold Minimum layers to trigger parallel processing  
 * @param enableSIMD Enable SIMD vectorized operations
 * @param simdVectorSize Preferred SIMD vector size (0 for auto-detect)
 * @param predictionCacheSize Maximum number of cached prediction vectors
 * @param probabilityCacheSize Maximum number of cached probability vectors
 * @param memoryOptimizationThreshold Memory usage threshold for cache cleanup (0.0-1.0)
 * @param enablePerformanceMonitoring Enable detailed performance tracking
 * 
 * @author Hal Hildebrand
 */
public record VectorizedDeepARTMAPParameters(
    DeepARTMAPParameters baseParameters,
    int channelParallelismLevel,
    int channelParallelThreshold,
    int layerParallelismLevel,
    int layerParallelThreshold,
    boolean enableSIMD,
    int simdVectorSize,
    int predictionCacheSize,
    int probabilityCacheSize,
    double memoryOptimizationThreshold,
    boolean enablePerformanceMonitoring
) {
    
    /**
     * Create parameters with validation.
     */
    public VectorizedDeepARTMAPParameters {
        if (baseParameters == null) {
            throw new IllegalArgumentException("baseParameters cannot be null");
        }
        if (channelParallelismLevel <= 0) {
            throw new IllegalArgumentException("channelParallelismLevel must be positive");
        }
        if (channelParallelThreshold < 1) {
            throw new IllegalArgumentException("channelParallelThreshold must be at least 1");
        }
        if (layerParallelismLevel <= 0) {
            throw new IllegalArgumentException("layerParallelismLevel must be positive");
        }
        if (layerParallelThreshold < 1) {
            throw new IllegalArgumentException("layerParallelThreshold must be at least 1");
        }
        if (simdVectorSize < 0) {
            throw new IllegalArgumentException("simdVectorSize must be non-negative");
        }
        if (predictionCacheSize < 0) {
            throw new IllegalArgumentException("predictionCacheSize must be non-negative");
        }
        if (probabilityCacheSize < 0) {
            throw new IllegalArgumentException("probabilityCacheSize must be non-negative");
        }
        if (memoryOptimizationThreshold < 0.0 || memoryOptimizationThreshold > 1.0) {
            throw new IllegalArgumentException("memoryOptimizationThreshold must be in range [0.0, 1.0]");
        }
    }
    
    /**
     * Create default VectorizedDeepARTMAP parameters.
     * 
     * @param baseParameters Base DeepARTMAP parameters
     * @return Default vectorized parameters
     */
    public static VectorizedDeepARTMAPParameters defaults(DeepARTMAPParameters baseParameters) {
        return new VectorizedDeepARTMAPParameters(
            baseParameters,
            Runtime.getRuntime().availableProcessors(),  // channelParallelismLevel
            2,                                           // channelParallelThreshold
            Runtime.getRuntime().availableProcessors(),  // layerParallelismLevel  
            2,                                           // layerParallelThreshold
            true,                                        // enableSIMD
            0,                                           // simdVectorSize (auto-detect)
            1000,                                        // predictionCacheSize
            500,                                         // probabilityCacheSize
            0.8,                                         // memoryOptimizationThreshold
            true                                         // enablePerformanceMonitoring
        );
    }
    
    /**
     * Create high-performance parameters for large datasets.
     * 
     * @param baseParameters Base DeepARTMAP parameters
     * @return High-performance configuration
     */
    public static VectorizedDeepARTMAPParameters highPerformance(DeepARTMAPParameters baseParameters) {
        var availableProcessors = Runtime.getRuntime().availableProcessors();
        return new VectorizedDeepARTMAPParameters(
            baseParameters,
            Math.max(4, availableProcessors),            // channelParallelismLevel
            1,                                           // channelParallelThreshold (aggressive)
            Math.max(4, availableProcessors),            // layerParallelismLevel
            1,                                           // layerParallelThreshold (aggressive)
            true,                                        // enableSIMD
            0,                                           // simdVectorSize (auto-detect)
            5000,                                        // predictionCacheSize (larger)
            2000,                                        // probabilityCacheSize (larger)
            0.9,                                         // memoryOptimizationThreshold (higher)
            true                                         // enablePerformanceMonitoring
        );
    }
    
    /**
     * Create memory-optimized parameters for resource-constrained environments.
     * 
     * @param baseParameters Base DeepARTMAP parameters
     * @return Memory-optimized configuration
     */
    public static VectorizedDeepARTMAPParameters memoryOptimized(DeepARTMAPParameters baseParameters) {
        return new VectorizedDeepARTMAPParameters(
            baseParameters,
            2,                                           // channelParallelismLevel (limited)
            4,                                           // channelParallelThreshold (conservative)
            2,                                           // layerParallelismLevel (limited)
            4,                                           // layerParallelThreshold (conservative)
            true,                                        // enableSIMD
            0,                                           // simdVectorSize (auto-detect)
            100,                                         // predictionCacheSize (smaller)
            50,                                          // probabilityCacheSize (smaller)
            0.6,                                         // memoryOptimizationThreshold (lower)
            false                                        // enablePerformanceMonitoring (disabled)
        );
    }
    
    /**
     * Check if channel parallel processing should be used.
     * 
     * @param channelCount Number of channels to process
     * @return true if parallel processing should be used
     */
    public boolean shouldUseChannelParallelism(int channelCount) {
        return channelCount >= channelParallelThreshold;
    }
    
    /**
     * Check if layer parallel processing should be used.
     * 
     * @param layerCount Number of layers to process
     * @return true if parallel processing should be used
     */
    public boolean shouldUseLayerParallelism(int layerCount) {
        return layerCount >= layerParallelThreshold;
    }
    
    /**
     * Get the effective SIMD vector size.
     * 
     * @return Actual vector size to use (auto-detected if simdVectorSize is 0)
     */
    public int getEffectiveVectorSize() {
        if (simdVectorSize > 0) {
            return simdVectorSize;
        }
        
        // Auto-detect preferred vector size
        try {
            var species = jdk.incubator.vector.FloatVector.SPECIES_PREFERRED;
            return species.length();
        } catch (Exception e) {
            // Fallback to reasonable default
            return 8;
        }
    }
    
    /**
     * Create a copy with modified base parameters.
     * 
     * @param newBaseParameters New base parameters
     * @return Updated parameters
     */
    public VectorizedDeepARTMAPParameters withBaseParameters(DeepARTMAPParameters newBaseParameters) {
        return new VectorizedDeepARTMAPParameters(
            newBaseParameters,
            channelParallelismLevel,
            channelParallelThreshold,
            layerParallelismLevel,
            layerParallelThreshold,
            enableSIMD,
            simdVectorSize,
            predictionCacheSize,
            probabilityCacheSize,
            memoryOptimizationThreshold,
            enablePerformanceMonitoring
        );
    }
    
    @Override
    public String toString() {
        return String.format(
            "VectorizedDeepARTMAPParameters{channelThreads=%d, layerThreads=%d, SIMD=%s, vectorSize=%d, caches=pred:%d/prob:%d}",
            channelParallelismLevel, layerParallelismLevel, enableSIMD, getEffectiveVectorSize(),
            predictionCacheSize, probabilityCacheSize
        );
    }
}