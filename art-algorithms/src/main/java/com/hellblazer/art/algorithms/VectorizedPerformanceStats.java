package com.hellblazer.art.algorithms;

/**
 * Performance statistics for VectorizedART operations.
 * 
 * This record captures key metrics for monitoring and optimizing
 * the performance of vectorized ART implementations.
 */
public record VectorizedPerformanceStats(
    long totalVectorOperations,
    long totalParallelTasks,
    double avgComputeTimeMs,
    int activeThreads,
    int cacheSize,
    int categoryCount
) {
    
    /**
     * Calculate operations per second.
     */
    public double getOperationsPerSecond() {
        return avgComputeTimeMs > 0.0 ? 1000.0 / avgComputeTimeMs : 0.0;
    }
    
    /**
     * Calculate parallel efficiency (0.0 to 1.0).
     */
    public double getParallelEfficiency() {
        return totalVectorOperations > 0 ? 
               (double) totalParallelTasks / totalVectorOperations : 0.0;
    }
    
    /**
     * Calculate cache hit ratio estimate.
     */
    public double getCacheEfficiency() {
        return categoryCount > 0 ? 
               Math.min(1.0, (double) cacheSize / categoryCount) : 0.0;
    }
    
    /**
     * Get performance summary string.
     */
    public String getPerformanceSummary() {
        return String.format(
            "VectorizedPerformanceStats: %.1f ops/sec, %.1f%% parallel, %.1f%% cache efficiency, %d threads",
            getOperationsPerSecond(),
            getParallelEfficiency() * 100,
            getCacheEfficiency() * 100,
            activeThreads
        );
    }
    
    @Override
    public String toString() {
        return String.format(
            "VectorizedPerformanceStats{ops=%d, parallel=%d, avgMs=%.3f, threads=%d, cache=%d, categories=%d}",
            totalVectorOperations, totalParallelTasks, avgComputeTimeMs, 
            activeThreads, cacheSize, categoryCount
        );
    }
}