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
package com.hellblazer.art.algorithms;

/**
 * Performance statistics for VectorizedDeepARTMAP implementations.
 * 
 * This record captures comprehensive metrics for SIMD and parallel processing
 * performance in vectorized DeepARTMAP operations.
 * 
 * @param totalVectorOperations Total number of vector operations performed
 * @param totalChannelParallelTasks Total number of parallel channel processing tasks
 * @param totalLayerParallelTasks Total number of parallel layer processing tasks
 * @param totalSIMDOperations Total number of SIMD-specific operations
 * @param avgComputeTimeMs Average computation time per operation in milliseconds
 * @param activeChannelThreads Number of currently active channel processing threads
 * @param activeLayerThreads Number of currently active layer processing threads
 * @param predictionCacheSize Current size of prediction cache
 * @param probabilityCacheSize Current size of probability cache
 * @param categoryCount Total number of categories across all layers
 * @param operationCount Total number of operations performed
 * 
 * @author Hal Hildebrand
 */
public record VectorizedDeepARTMAPPerformanceStats(
    long totalVectorOperations,
    long totalChannelParallelTasks,
    long totalLayerParallelTasks,
    long totalSIMDOperations,
    double avgComputeTimeMs,
    int activeChannelThreads,
    int activeLayerThreads,
    int predictionCacheSize,
    int probabilityCacheSize,
    int categoryCount,
    long operationCount
) {
    
    /**
     * Calculate total parallel tasks (channel + layer).
     * 
     * @return Total parallel task count
     */
    public long totalParallelTasks() {
        return totalChannelParallelTasks + totalLayerParallelTasks;
    }
    
    /**
     * Calculate channel parallelism efficiency.
     * 
     * @return Channel parallelism ratio (0.0 to 1.0)
     */
    public double channelParallelismEfficiency() {
        if (operationCount == 0) return 0.0;
        return (double) totalChannelParallelTasks / operationCount;
    }
    
    /**
     * Calculate layer parallelism efficiency.
     * 
     * @return Layer parallelism ratio (0.0 to 1.0)
     */
    public double layerParallelismEfficiency() {
        if (operationCount == 0) return 0.0;
        return (double) totalLayerParallelTasks / operationCount;
    }
    
    /**
     * Calculate SIMD operation efficiency.
     * 
     * @return SIMD operation ratio (0.0 to 1.0)
     */
    public double simdEfficiency() {
        if (operationCount == 0) return 0.0;
        return (double) totalSIMDOperations / operationCount;
    }
    
    /**
     * Calculate operations per second based on average compute time.
     * 
     * @return Operations per second
     */
    public double operationsPerSecond() {
        if (avgComputeTimeMs == 0.0) return 0.0;
        return 1000.0 / avgComputeTimeMs;
    }
    
    /**
     * Calculate total cache size (prediction + probability).
     * 
     * @return Total cache entries
     */
    public int totalCacheSize() {
        return predictionCacheSize + probabilityCacheSize;
    }
    
    /**
     * Calculate prediction cache hit ratio estimate.
     * 
     * @return Estimated cache efficiency (0.0 to 1.0)
     */
    public double cacheEfficiency() {
        if (operationCount == 0) return 0.0;
        var totalCache = totalCacheSize();
        if (totalCache == 0) return 0.0;
        
        // Estimate based on cache size relative to operations
        return Math.min(1.0, (double) totalCache / Math.max(operationCount, 1));
    }
    
    /**
     * Get total active threads across both pools.
     * 
     * @return Total active thread count
     */
    public int totalActiveThreads() {
        return activeChannelThreads + activeLayerThreads;
    }
    
    /**
     * Calculate memory usage indicator based on cache sizes.
     * 
     * @return Relative memory usage indicator
     */
    public double memoryUsageIndicator() {
        // Estimate memory usage based on cache sizes
        var totalEntries = totalCacheSize();
        var estimatedKB = totalEntries * 0.1; // Rough estimate: 100 bytes per cache entry
        return estimatedKB;
    }
    
    /**
     * Get a formatted summary of key performance metrics.
     * 
     * @return Human-readable performance summary
     */
    public String getSummary() {
        return String.format(
            "VectorizedDeepARTMAP Performance: %.1f ops/sec, %.1f%% channel parallel, %.1f%% layer parallel, %.1f%% SIMD, %d threads active",
            operationsPerSecond(),
            channelParallelismEfficiency() * 100,
            layerParallelismEfficiency() * 100,
            simdEfficiency() * 100,
            totalActiveThreads()
        );
    }
    
    @Override
    public String toString() {
        return String.format(
            "VectorizedDeepARTMAPPerformanceStats{" +
            "vectorOps=%d, channelTasks=%d, layerTasks=%d, simdOps=%d, " +
            "avgMs=%.3f, threads=%d/%d, caches=%d/%d, categories=%d, ops=%d}",
            totalVectorOperations, totalChannelParallelTasks, totalLayerParallelTasks, 
            totalSIMDOperations, avgComputeTimeMs, activeChannelThreads, activeLayerThreads,
            predictionCacheSize, probabilityCacheSize, categoryCount, operationCount
        );
    }
}