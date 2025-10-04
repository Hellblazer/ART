package com.hellblazer.art.laminar.batch;

/**
 * Performance statistics from batch processing operation.
 *
 * <p>Provides detailed timing breakdown and performance metrics for analyzing
 * and optimizing batch processing performance.
 *
 * <h2>Key Metrics</h2>
 * <ul>
 *   <li>Total and per-layer timing</li>
 *   <li>Category learning statistics</li>
 *   <li>SIMD operation counts</li>
 *   <li>Parallel execution metrics</li>
 *   <li>Cache utilization estimates</li>
 * </ul>
 *
 * @param batchSize number of patterns processed
 * @param totalTimeNanos total processing time in nanoseconds
 * @param layer4TimeNanos time spent in Layer 4 processing
 * @param layer23TimeNanos time spent in Layer 2/3 processing
 * @param layer5TimeNanos time spent in Layer 5 processing
 * @param layer6TimeNanos time spent in Layer 6 processing
 * @param artTimeNanos time spent in ART module
 * @param categoriesCreated number of new categories created
 * @param categorySearches total category searches performed
 * @param simdOperations count of SIMD operations executed
 * @param parallelTasks count of parallel tasks executed
 * @param cacheHitRate estimated cache hit rate (0.0-1.0, -1.0 if not tracked)
 *
 * @author Hal Hildebrand
 */
public record BatchStatistics(
    int batchSize,
    long totalTimeNanos,
    long layer4TimeNanos,
    long layer23TimeNanos,
    long layer5TimeNanos,
    long layer6TimeNanos,
    long artTimeNanos,
    int categoriesCreated,
    int categorySearches,
    long simdOperations,
    long parallelTasks,
    double cacheHitRate
) {
    /**
     * Validate statistics fields.
     *
     * @throws IllegalArgumentException if values are invalid
     */
    public BatchStatistics {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive");
        }
        if (totalTimeNanos < 0) {
            throw new IllegalArgumentException("totalTimeNanos cannot be negative");
        }
        if (cacheHitRate < -1.0 || cacheHitRate > 1.0) {
            throw new IllegalArgumentException("cacheHitRate must be in [-1.0, 1.0]");
        }
    }

    /**
     * Calculate patterns per second throughput.
     *
     * @return throughput in patterns/second
     */
    public double getPatternsPerSecond() {
        if (totalTimeNanos == 0) return Double.POSITIVE_INFINITY;
        return (batchSize * 1e9) / totalTimeNanos;
    }

    /**
     * Calculate average time per pattern in microseconds.
     *
     * @return microseconds per pattern
     */
    public double getMicrosecondsPerPattern() {
        return (totalTimeNanos / 1000.0) / batchSize;
    }

    /**
     * Calculate average time per pattern in milliseconds.
     *
     * @return milliseconds per pattern
     */
    public double getMillisecondsPerPattern() {
        return (totalTimeNanos / 1e6) / batchSize;
    }

    /**
     * Calculate speedup vs baseline single-pattern processing.
     *
     * @param baselineNanosPerPattern baseline single-pattern time in nanoseconds
     * @return speedup factor (>1.0 means faster)
     */
    public double getSpeedup(long baselineNanosPerPattern) {
        if (baselineNanosPerPattern <= 0) {
            throw new IllegalArgumentException("baselineNanosPerPattern must be positive");
        }
        var batchAvgNanos = (double) totalTimeNanos / batchSize;
        return baselineNanosPerPattern / batchAvgNanos;
    }

    /**
     * Calculate percentage of time spent in each layer.
     *
     * @return array [layer4%, layer23%, layer5%, layer6%, art%]
     */
    public double[] getTimeBreakdownPercentages() {
        var totalLayerTime = (double) (layer4TimeNanos + layer23TimeNanos +
                                      layer5TimeNanos + layer6TimeNanos + artTimeNanos);
        if (totalLayerTime == 0) {
            return new double[]{0, 0, 0, 0, 0};
        }
        return new double[]{
            (layer4TimeNanos / totalLayerTime) * 100.0,
            (layer23TimeNanos / totalLayerTime) * 100.0,
            (layer5TimeNanos / totalLayerTime) * 100.0,
            (layer6TimeNanos / totalLayerTime) * 100.0,
            (artTimeNanos / totalLayerTime) * 100.0
        };
    }

    /**
     * Get SIMD operations per pattern.
     *
     * @return average SIMD operations per pattern
     */
    public double getSIMDOperationsPerPattern() {
        return (double) simdOperations / batchSize;
    }

    /**
     * Get parallel efficiency metric.
     * Measures how effectively parallelism was utilized.
     *
     * @return efficiency (0.0-1.0, higher is better, -1.0 if not applicable)
     */
    public double getParallelEfficiency() {
        if (parallelTasks == 0) return -1.0;
        var idealParallelism = batchSize;
        return Math.min(1.0, (double) parallelTasks / idealParallelism);
    }

    /**
     * Check if detailed statistics were collected.
     *
     * @return true if per-layer timing is available
     */
    public boolean hasDetailedStats() {
        return layer4TimeNanos > 0 || layer23TimeNanos > 0 ||
               layer5TimeNanos > 0 || layer6TimeNanos > 0;
    }

    /**
     * Create minimal statistics (just batch size and total time).
     *
     * @param batchSize number of patterns
     * @param totalTimeNanos total time
     * @return minimal statistics
     */
    public static BatchStatistics minimal(int batchSize, long totalTimeNanos) {
        return new BatchStatistics(
            batchSize, totalTimeNanos,
            0, 0, 0, 0, 0,  // No per-layer timing
            0, 0,           // No category stats
            0, 0,           // No SIMD/parallel stats
            -1.0            // Cache not tracked
        );
    }

    @Override
    public String toString() {
        return String.format(
            "BatchStatistics[batch=%d, total=%.2fms, throughput=%.1f patterns/sec, " +
            "categories=%d, simd=%d, parallel=%d]",
            batchSize,
            totalTimeNanos / 1e6,
            getPatternsPerSecond(),
            categoriesCreated,
            simdOperations,
            parallelTasks
        );
    }

    /**
     * Generate detailed report string.
     *
     * @return multi-line detailed statistics report
     */
    public String toDetailedString() {
        var sb = new StringBuilder();
        sb.append("=== Batch Processing Statistics ===\n");
        sb.append(String.format("Batch size:      %d patterns\n", batchSize));
        sb.append(String.format("Total time:      %.2f ms\n", totalTimeNanos / 1e6));
        sb.append(String.format("Per pattern:     %.2f μs\n", getMicrosecondsPerPattern()));
        sb.append(String.format("Throughput:      %.1f patterns/sec\n", getPatternsPerSecond()));

        if (hasDetailedStats()) {
            sb.append("\nTiming Breakdown:\n");
            var breakdown = getTimeBreakdownPercentages();
            sb.append(String.format("  Layer 4:       %.2f ms (%.1f%%)\n",
                layer4TimeNanos / 1e6, breakdown[0]));
            sb.append(String.format("  Layer 2/3:     %.2f ms (%.1f%%)\n",
                layer23TimeNanos / 1e6, breakdown[1]));
            sb.append(String.format("  Layer 5:       %.2f ms (%.1f%%)\n",
                layer5TimeNanos / 1e6, breakdown[2]));
            sb.append(String.format("  Layer 6:       %.2f ms (%.1f%%)\n",
                layer6TimeNanos / 1e6, breakdown[3]));
            sb.append(String.format("  ART Module:    %.2f ms (%.1f%%)\n",
                artTimeNanos / 1e6, breakdown[4]));
        }

        sb.append("\nCategory Learning:\n");
        sb.append(String.format("  New categories:  %d\n", categoriesCreated));
        sb.append(String.format("  Total searches:  %d\n", categorySearches));

        if (simdOperations > 0) {
            sb.append("\nOptimization Metrics:\n");
            sb.append(String.format("  SIMD ops:        %d (%.1f/pattern)\n",
                simdOperations, getSIMDOperationsPerPattern()));
        }

        if (parallelTasks > 0) {
            sb.append(String.format("  Parallel tasks:  %d (%.1f%% efficiency)\n",
                parallelTasks, getParallelEfficiency() * 100));
        }

        if (cacheHitRate >= 0) {
            sb.append(String.format("  Cache hit rate:  %.1f%%\n", cacheHitRate * 100));
        }

        return sb.toString();
    }
}
