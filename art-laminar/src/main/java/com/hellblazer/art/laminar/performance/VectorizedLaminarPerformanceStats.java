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
package com.hellblazer.art.laminar.performance;

/**
 * Performance statistics for vectorized laminar circuit operations.
 *
 * Tracks SIMD vectorization metrics specific to laminar cortical processing:
 * - Layer-wise computation efficiency
 * - Shunting dynamics vectorization
 * - Pathway parallelization metrics
 * - Cross-layer communication overhead
 */
public record VectorizedLaminarPerformanceStats(
    long totalVectorOperations,
    long totalParallelTasks,
    long activationCalls,
    long matchCalls,
    long learningCalls,
    double avgComputeTimeMs,
    int vectorLaneWidth
) {

    /**
     * Calculate the theoretical speedup based on vectorization.
     *
     * @return estimated speedup factor
     */
    public double getEstimatedSpeedup() {
        if (totalVectorOperations == 0) {
            return 1.0;
        }

        // Estimate based on vector lane width and operation count
        var vectorizedOps = totalVectorOperations * vectorLaneWidth;
        var scalarOps = activationCalls + matchCalls + learningCalls;

        if (scalarOps == 0) {
            return vectorLaneWidth;
        }

        return (double) vectorizedOps / scalarOps;
    }

    /**
     * Get the average operations per millisecond.
     *
     * @return operations/ms throughput
     */
    public double getThroughput() {
        if (avgComputeTimeMs == 0) {
            return 0.0;
        }

        var totalOps = activationCalls + matchCalls + learningCalls;
        return totalOps / avgComputeTimeMs;
    }

    /**
     * Get vectorization efficiency as a percentage.
     *
     * @return efficiency percentage (0-100)
     */
    public double getVectorizationEfficiency() {
        var totalOps = activationCalls + matchCalls + learningCalls;
        if (totalOps == 0) {
            return 0.0;
        }

        // Efficiency based on how many operations were vectorized
        return (totalVectorOperations * 100.0) / totalOps;
    }

    @Override
    public String toString() {
        return String.format(
            "VectorizedLaminarPerformanceStats{" +
            "vectorOps=%d, parallelTasks=%d, " +
            "activations=%d, matches=%d, learning=%d, " +
            "avgTimeMs=%.2f, vectorWidth=%d, " +
            "speedup=%.2fx, throughput=%.0f ops/ms, efficiency=%.1f%%}",
            totalVectorOperations, totalParallelTasks,
            activationCalls, matchCalls, learningCalls,
            avgComputeTimeMs, vectorLaneWidth,
            getEstimatedSpeedup(), getThroughput(), getVectorizationEfficiency()
        );
    }
}