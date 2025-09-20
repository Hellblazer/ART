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
package com.hellblazer.art.performance.reinforcement;

/**
 * Performance statistics tracking for VectorizedFALCON.
 * 
 * Tracks:
 * - SIMD operations count
 * - Parallel action evaluations
 * - Training and prediction times
 * - FusionART performance metrics
 * - Memory usage
 * - Throughput metrics
 * 
 * @author Hal Hildebrand
 */
public class VectorizedFALCONPerformanceStats {
    
    // SIMD metrics
    private long simdOperations = 0;
    private long vectorizedComputations = 0;
    
    // Parallel processing metrics
    private long parallelEvaluations = 0;
    private long parallelBatches = 0;
    
    // Timing metrics (nanoseconds)
    private long totalTrainingTime = 0;
    private long totalActionSelectionTime = 0;
    private long totalPredictionTime = 0;
    
    // Throughput metrics
    private long patternsProcessed = 0;
    private long actionsEvaluated = 0;
    
    // FusionART sub-component stats
    private Object fusionARTStats;
    
    // Memory metrics
    private long peakMemoryUsage = 0;
    private long currentMemoryUsage = 0;
    
    /**
     * Record time spent in training.
     * 
     * @param nanos Time in nanoseconds
     */
    public void recordTrainingTime(long nanos) {
        totalTrainingTime += nanos;
    }
    
    /**
     * Record time spent in action selection.
     * 
     * @param nanos Time in nanoseconds
     */
    public void recordActionSelectionTime(long nanos) {
        totalActionSelectionTime += nanos;
    }
    
    /**
     * Record time spent in prediction.
     * 
     * @param nanos Time in nanoseconds
     */
    public void recordPredictionTime(long nanos) {
        totalPredictionTime += nanos;
    }
    
    /**
     * Set SIMD operations count.
     * 
     * @param count Number of SIMD operations
     */
    public void setSIMDOperations(long count) {
        this.simdOperations = count;
    }
    
    /**
     * Set parallel evaluations count.
     * 
     * @param count Number of parallel evaluations
     */
    public void setParallelEvaluations(long count) {
        this.parallelEvaluations = count;
    }
    
    /**
     * Set FusionART performance statistics.
     * 
     * @param stats FusionART stats
     */
    public void setFusionARTStats(Object stats) {
        this.fusionARTStats = stats;
    }
    
    /**
     * Increment patterns processed counter.
     * 
     * @param count Number of patterns
     */
    public void addPatternsProcessed(long count) {
        this.patternsProcessed += count;
    }
    
    /**
     * Increment actions evaluated counter.
     * 
     * @param count Number of actions
     */
    public void addActionsEvaluated(long count) {
        this.actionsEvaluated += count;
    }
    
    /**
     * Update memory usage.
     */
    public void updateMemoryUsage() {
        Runtime runtime = Runtime.getRuntime();
        currentMemoryUsage = runtime.totalMemory() - runtime.freeMemory();
        peakMemoryUsage = Math.max(peakMemoryUsage, currentMemoryUsage);
    }
    
    /**
     * Reset all statistics.
     */
    public void reset() {
        simdOperations = 0;
        vectorizedComputations = 0;
        parallelEvaluations = 0;
        parallelBatches = 0;
        totalTrainingTime = 0;
        totalActionSelectionTime = 0;
        totalPredictionTime = 0;
        patternsProcessed = 0;
        actionsEvaluated = 0;
        peakMemoryUsage = 0;
        currentMemoryUsage = 0;
        fusionARTStats = null;
    }
    
    /**
     * Get average training time per pattern.
     * 
     * @return Average time in milliseconds
     */
    public double getAverageTrainingTimeMs() {
        if (patternsProcessed == 0) return 0.0;
        return (totalTrainingTime / 1_000_000.0) / patternsProcessed;
    }
    
    /**
     * Get average action selection time.
     * 
     * @return Average time in milliseconds
     */
    public double getAverageActionSelectionTimeMs() {
        if (actionsEvaluated == 0) return 0.0;
        return (totalActionSelectionTime / 1_000_000.0) / actionsEvaluated;
    }
    
    /**
     * Get training throughput.
     * 
     * @return Patterns per second
     */
    public double getTrainingThroughput() {
        if (totalTrainingTime == 0) return 0.0;
        return patternsProcessed * 1_000_000_000.0 / totalTrainingTime;
    }
    
    /**
     * Get action evaluation throughput.
     * 
     * @return Actions per second
     */
    public double getActionEvaluationThroughput() {
        if (totalActionSelectionTime == 0) return 0.0;
        return actionsEvaluated * 1_000_000_000.0 / totalActionSelectionTime;
    }
    
    /**
     * Get SIMD efficiency ratio.
     * 
     * @return Ratio of SIMD operations to total computations
     */
    public double getSIMDEfficiency() {
        long totalOps = simdOperations + vectorizedComputations;
        if (totalOps == 0) return 0.0;
        return (double) simdOperations / totalOps;
    }
    
    /**
     * Get parallel processing efficiency.
     * 
     * @return Ratio of parallel to total evaluations
     */
    public double getParallelEfficiency() {
        if (actionsEvaluated == 0) return 0.0;
        return (double) parallelEvaluations / actionsEvaluated;
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("VectorizedFALCON Performance Statistics:\n");
        sb.append("========================================\n");
        
        // SIMD metrics
        sb.append("SIMD Operations: ").append(simdOperations).append("\n");
        sb.append("SIMD Efficiency: ").append(String.format("%.2f%%", getSIMDEfficiency() * 100)).append("\n");
        
        // Parallel metrics
        sb.append("Parallel Evaluations: ").append(parallelEvaluations).append("\n");
        sb.append("Parallel Efficiency: ").append(String.format("%.2f%%", getParallelEfficiency() * 100)).append("\n");
        
        // Timing metrics
        sb.append("Total Training Time: ").append(String.format("%.3f ms", totalTrainingTime / 1_000_000.0)).append("\n");
        sb.append("Avg Training Time/Pattern: ").append(String.format("%.3f ms", getAverageTrainingTimeMs())).append("\n");
        sb.append("Avg Action Selection Time: ").append(String.format("%.3f ms", getAverageActionSelectionTimeMs())).append("\n");
        
        // Throughput metrics
        sb.append("Training Throughput: ").append(String.format("%.0f patterns/sec", getTrainingThroughput())).append("\n");
        sb.append("Action Eval Throughput: ").append(String.format("%.0f actions/sec", getActionEvaluationThroughput())).append("\n");
        
        // Memory metrics
        sb.append("Peak Memory Usage: ").append(String.format("%.2f MB", peakMemoryUsage / (1024.0 * 1024.0))).append("\n");
        sb.append("Current Memory Usage: ").append(String.format("%.2f MB", currentMemoryUsage / (1024.0 * 1024.0))).append("\n");
        
        // FusionART stats
        if (fusionARTStats != null) {
            sb.append("\nFusionART Statistics:\n");
            sb.append("---------------------\n");
            sb.append(fusionARTStats.toString());
        }
        
        return sb.toString();
    }
    
    // Getters for all metrics
    
    public long getSIMDOperations() {
        return simdOperations;
    }
    
    public long getVectorizedComputations() {
        return vectorizedComputations;
    }
    
    public long getParallelEvaluations() {
        return parallelEvaluations;
    }
    
    public long getParallelBatches() {
        return parallelBatches;
    }
    
    public long getTotalTrainingTime() {
        return totalTrainingTime;
    }
    
    public long getTotalActionSelectionTime() {
        return totalActionSelectionTime;
    }
    
    public long getTotalPredictionTime() {
        return totalPredictionTime;
    }
    
    public long getPatternsProcessed() {
        return patternsProcessed;
    }
    
    public long getActionsEvaluated() {
        return actionsEvaluated;
    }
    
    public long getPeakMemoryUsage() {
        return peakMemoryUsage;
    }
    
    public long getCurrentMemoryUsage() {
        return currentMemoryUsage;
    }
    
    public Object getFusionARTStats() {
        return fusionARTStats;
    }
}