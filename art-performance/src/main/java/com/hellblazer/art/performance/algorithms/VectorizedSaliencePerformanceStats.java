package com.hellblazer.art.performance.algorithms;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Performance statistics for VectorizedSalienceART algorithm.
 * Tracks SIMD operations, salience computations, memory efficiency, and other metrics.
 */
public record VectorizedSaliencePerformanceStats(
    long totalOperations,
    long simdOperations,
    double averageProcessingTime,
    double averageSalienceComputationTime,
    long statisticsUpdateCount,
    double averageCategoryUtilization,
    Map<Integer, Double> categorySalienceScores,
    long sparseVectorOperations,
    double memoryEfficiencyRatio
) {
    
    /**
     * Constructor that ensures immutability of the salience scores map
     */
    public VectorizedSaliencePerformanceStats {
        // Make defensive copy and wrap in unmodifiable map
        categorySalienceScores = Collections.unmodifiableMap(
            new HashMap<>(categorySalienceScores != null ? categorySalienceScores : Map.of())
        );
    }
    
    /**
     * Create empty statistics instance
     */
    public static VectorizedSaliencePerformanceStats empty() {
        return new VectorizedSaliencePerformanceStats(
            0L,     // totalOperations
            0L,     // simdOperations
            0.0,    // averageProcessingTime
            0.0,    // averageSalienceComputationTime
            0L,     // statisticsUpdateCount
            0.0,    // averageCategoryUtilization
            Map.of(), // categorySalienceScores (empty)
            0L,     // sparseVectorOperations
            0.0     // memoryEfficiencyRatio
        );
    }
    
    /**
     * Merge two statistics instances
     * @param other the other statistics to merge with
     * @return merged statistics with combined metrics
     */
    public VectorizedSaliencePerformanceStats merge(VectorizedSaliencePerformanceStats other) {
        if (other == null) {
            return this;
        }
        
        long newTotalOps = this.totalOperations + other.totalOperations;
        long newSimdOps = this.simdOperations + other.simdOperations;
        long newStatsCount = this.statisticsUpdateCount + other.statisticsUpdateCount;
        long newSparseOps = this.sparseVectorOperations + other.sparseVectorOperations;
        
        // Weighted average for times
        double totalWeight = this.totalOperations + other.totalOperations;
        double newAvgProcessingTime = totalWeight > 0 ?
            (this.averageProcessingTime * this.totalOperations + 
             other.averageProcessingTime * other.totalOperations) / totalWeight : 0.0;
        
        double newAvgSalienceTime = totalWeight > 0 ?
            (this.averageSalienceComputationTime * this.totalOperations + 
             other.averageSalienceComputationTime * other.totalOperations) / totalWeight : 0.0;
        
        // Average of utilization rates
        double newAvgUtilization = (this.averageCategoryUtilization + other.averageCategoryUtilization) / 2.0;
        double newMemoryRatio = (this.memoryEfficiencyRatio + other.memoryEfficiencyRatio) / 2.0;
        
        // Merge salience scores maps
        Map<Integer, Double> mergedScores = new HashMap<>(this.categorySalienceScores);
        other.categorySalienceScores.forEach((key, value) -> 
            mergedScores.merge(key, value, (v1, v2) -> (v1 + v2) / 2.0)
        );
        
        return new VectorizedSaliencePerformanceStats(
            newTotalOps,
            newSimdOps,
            newAvgProcessingTime,
            newAvgSalienceTime,
            newStatsCount,
            newAvgUtilization,
            mergedScores,
            newSparseOps,
            newMemoryRatio
        );
    }
    
    /**
     * Calculate SIMD utilization ratio
     * @return ratio of SIMD operations to total operations
     */
    public double getSimdUtilizationRatio() {
        return totalOperations > 0 ? (double) simdOperations / totalOperations : 0.0;
    }
    
    /**
     * Calculate salience computation overhead
     * @return ratio of salience computation time to total processing time
     */
    public double getSalienceOverheadRatio() {
        return averageProcessingTime > 0 ? 
            averageSalienceComputationTime / averageProcessingTime : 0.0;
    }
    
    /**
     * Calculate sparse operation ratio
     * @return ratio of sparse operations to total operations
     */
    public double getSparseOperationRatio() {
        return totalOperations > 0 ? (double) sparseVectorOperations / totalOperations : 0.0;
    }
    
    /**
     * Get formatted performance summary
     */
    @Override
    public String toString() {
        return String.format(
            "VectorizedSaliencePerformanceStats{" +
            "totalOperations=%d, simdOperations=%d (%.1f%%), " +
            "avgProcessingTime=%.3fms, avgSalienceTime=%.3fms (%.1f%% overhead), " +
            "statsUpdates=%d, categoryUtilization=%.1f%%, " +
            "sparseOps=%d (%.1f%%), memoryEfficiencyRatio=%.2f, " +
            "categoriesTracked=%d}",
            totalOperations, simdOperations, getSimdUtilizationRatio() * 100,
            averageProcessingTime, averageSalienceComputationTime, getSalienceOverheadRatio() * 100,
            statisticsUpdateCount, averageCategoryUtilization * 100,
            sparseVectorOperations, getSparseOperationRatio() * 100,
            memoryEfficiencyRatio,
            categorySalienceScores.size()
        );
    }
}